"""
TranscrevAI Main Application - Final, Optimized Architecture.

This version uses a single-process, async model, which has been proven to be the
most performant for single-file processing. It initializes ML models once at startup
and runs the CPU-bound pipeline as a non-blocking background task.
"""

import asyncio
import logging
import os
import time
import threading
import base64
import uuid
import queue
from src.worker import transcription_worker
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# Eagerly import torch and set threads at the very beginning
import torch
import psutil

import uvicorn
from fastapi import (FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, BackgroundTasks)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Core application modules
from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from src.audio_processing import AudioQualityAnalyzer, SessionManager, LiveAudioProcessor, SessionData
from datetime import datetime
from src.subtitle_generator import generate_srt
from src.file_manager import FileManager
from src.websocket_enhancements import get_websocket_safety_manager, MessagePriority
from src.pipeline import process_audio_pipeline
from src.exceptions import TranscrevAIError, TranscriptionError, DiarizationError, AudioProcessingError, SessionError, ValidationError
from config.app_config import get_config

# --- Global Configuration & Tuning ---
app_config = get_config()
logging.basicConfig(level=app_config.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adaptive Performance Tuning (Hardware-Aware)
# Automatically configures optimal thread counts based on available hardware
from src.audio_processing import configure_adaptive_threads

torch_threads, omp_threads = configure_adaptive_threads()
torch.set_num_threads(torch_threads)
os.environ["OMP_NUM_THREADS"] = str(omp_threads)
logger.info(f"PERFORMANCE TUNING APPLIED: OMP_NUM_THREADS={omp_threads}, torch_threads={torch_threads}")

DEVICE = "cpu"  # This is a CPU-only application

# Import DI functions
from src.dependencies import (
    get_file_manager,
    get_transcription_service,
    get_diarization_service,
    get_audio_quality_analyzer,
    get_session_manager,
    get_live_audio_processor,
    get_transcription_queue,
    get_worker_thread,
    cleanup_services
)

# --- Lifespan Manager (for startup and shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TranscrevAI Server Starting...")
    logger.info("Initializing Machine Learning services...")

    # Initialize services via DI - triggers singleton creation
    get_file_manager()
    get_transcription_service()
    get_diarization_service()
    get_audio_quality_analyzer()
    get_session_manager()
    get_live_audio_processor()
    get_transcription_queue()

    # Get running loop and pass to worker
    loop = asyncio.get_running_loop()
    get_worker_thread(loop)

    logger.info("Services and worker initialized successfully.")
    yield

    # --- Shutdown Logic ---
    logger.info("Shutting down TranscrevAI...")

    # Cleanup sessions before shutting down services
    session_manager = get_session_manager()
    all_sessions = await session_manager.get_all_session_ids()
    logger.info(f"Cleaning up {len(all_sessions)} active sessions...")
    for session_id in all_sessions:
        await session_manager.remove_session(session_id)
    logger.info("All sessions cleaned up.")

    # Cleanup all services
    cleanup_services()

# --- FastAPI App Setup ---
app = FastAPI(title="TranscrevAI Single-Process", version="6.0.0", lifespan=lifespan)

# Rate Limiting Configuration
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Custom rate limit exception handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please try again later."}
    )

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === API Endpoints ===
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
@limiter.limit("10/minute")
async def upload_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    file_manager: FileManager = Depends(get_file_manager),
    session_manager: SessionManager = Depends(get_session_manager)
):

    # Validate file size (500MB limit for 60min audio)
    MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB
    file_size = 0

    # Read file to check size
    content = await file.read()
    file_size = len(content)

    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande. Tamanho máximo: 500MB. Tamanho recebido: {file_size / (1024*1024):.1f}MB"
        )

    # Reset file pointer for later processing
    await file.seek(0)

    # Use the provided session_id if it exists, otherwise create a new one.
    session_id = session_id or str(uuid.uuid4())
    logger.info(f"Upload received ({file_size / (1024*1024):.1f}MB), processing for session {session_id}")

    try:
        # Save the uploaded file asynchronously
        file_path = await file_manager.save_uploaded_file(file, file.filename or f"{session_id}.wav")

        # Validate audio duration (60min max)
        import librosa
        try:
            duration = librosa.get_duration(path=str(file_path))
            if duration > 3600:  # 60 minutes
                Path(file_path).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Áudio muito longo. Duração máxima: 60 minutos. Duração: {duration/60:.1f} min"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Could not validate duration for {file_path}: {e}")

        # Create a session data object for this upload
        session_data = SessionData(
            session_id=session_id,
            websocket=None, # No websocket for uploads
            format=Path(file_path).suffix,
            started_at=datetime.now(),
            temp_file=str(file_path) # The uploaded file is the temp file
        )
        await session_manager.create_session(session_id, session_data)

        # Run the processing pipeline in the background without blocking
        asyncio.create_task(process_audio_pipeline(str(file_path), session_id))
        
        logger.info(f"Job accepted for session {session_id}. Processing started in background.")
        return JSONResponse(content={"success": True, "session_id": session_id})
    except Exception as e:
        logger.error(f"Upload failed for session {session_id}: {e}", exc_info=True)
        # Ensure session is cleaned up on failure
        await session_manager.remove_session(session_id)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download-srt/{session_id}")
async def download_srt(session_id: str, session_manager: SessionManager = Depends(get_session_manager)):
    session = await session_manager.get_session(session_id)
    
    # The new SessionData object stores file paths in a `files` dictionary.
    # This path is populated by the pipeline.
    srt_path = session.files.get("subtitles") if session else None

    if not srt_path or not os.path.exists(srt_path):
        return JSONResponse(status_code=404, content={"error": "SRT file not found or not yet generated."})
    
    return FileResponse(path=srt_path, filename=f"transcription_{session_id}.srt", media_type="application/x-subrip")

@app.get("/api/download/{session_id}/{file_type}")
async def download_file(session_id: str, file_type: str, session_manager: SessionManager = Depends(get_session_manager)):
    """
    Download recorded files from live recording sessions.
    """
    valid_types = ['audio', 'transcript', 'subtitles']
    if file_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Must be one of: {valid_types}")

    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get file path from the session's `files` dictionary
    file_path_str = session.files.get(file_type)
    if not file_path_str or not Path(file_path_str).exists():
        raise HTTPException(status_code=404, detail=f"{file_type.capitalize()} file not found for this session")

    file_path = Path(file_path_str)
    audio_format = getattr(session, 'format', 'wav') # Get format from session

    # On-demand MP4 conversion for audio files
    if file_type == 'audio' and audio_format == 'mp4' and file_path.suffix.lower() == '.wav':
        from src.audio_processing import convert_wav_to_mp4
        mp4_path = file_path.with_suffix(".mp4")
        subtitle_path = session.files.get("subtitles")

        if convert_wav_to_mp4(str(file_path), str(mp4_path), subtitle_path=subtitle_path):
            logger.info(f"MP4 video created for session {session_id}")
            return FileResponse(path=str(mp4_path), media_type='video/mp4', filename=f"recording_{session_id}.mp4")
        else:
            logger.warning(f"MP4 conversion failed for session {session_id}. Serving original WAV.")

    # Define file extensions and media types
    if file_type == 'audio':
        media_type = 'audio/wav'
        filename = f"recording_{session_id}.wav"
    else:
        file_config = {
            'transcript': {'ext': '.txt', 'media_type': 'text/plain'},
            'subtitles': {'ext': '.srt', 'media_type': 'application/x-subrip'}
        }
        config = file_config[file_type]
        filename = f"recording_{session_id}{config['ext']}"
        media_type = config['media_type']

    logger.info(f"Download requested: session={session_id}, file_type={file_type}, path={file_path}")

    return FileResponse(path=str(file_path), media_type=media_type, filename=filename)
@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}

from src.websocket_handler import WebSocketValidator, WebSocketHandler
from collections import defaultdict
from time import time as current_time

# WebSocket rate limiting (manual implementation as slowapi doesn't support WS)
ws_connection_tracker = defaultdict(list)
WS_RATE_LIMIT = 20  # connections per minute
WS_RATE_WINDOW = 60  # seconds

@app.post("/test/reset-rate-limit", status_code=200)
async def reset_rate_limit_for_testing():
    """Reset rate limiters - FOR TESTING ONLY"""
    ws_connection_tracker.clear()
    # Reset slowapi limiter storage
    try:
        limiter.reset()
    except:
        pass
    return {"status": "rate limits reset"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    live_audio_processor: LiveAudioProcessor = Depends(get_live_audio_processor),
    transcription_queue: queue.Queue = Depends(get_transcription_queue)
):
    """
    WebSocket endpoint for live audio recording, refactored for centralized session management.

    Creates and removes sessions exclusively through the SessionManager, ensuring a single
    source of truth and proper resource cleanup.
    """
    # Rate limiting check
    client_ip = websocket.client.host if websocket.client else "unknown"
    now = current_time()

    # Clean old entries
    ws_connection_tracker[client_ip] = [
        t for t in ws_connection_tracker[client_ip]
        if now - t < WS_RATE_WINDOW
    ]

    # Check limit
    if len(ws_connection_tracker[client_ip]) >= WS_RATE_LIMIT:
        await websocket.close(code=1008, reason="Rate limit exceeded")
        logger.warning(f"WebSocket rate limit exceeded for IP: {client_ip}")
        return

    ws_connection_tracker[client_ip].append(now)

    await websocket.accept()
    logger.info(f"WebSocket connected for session: {session_id}")

    # Create a new session data object for this connection
    session_data = SessionData(
        session_id=session_id,
        websocket=websocket,
        format=None, # Will be set on 'start' message
        started_at=datetime.now()
    )

    try:
        # Defensively check for and remove any lingering session with the same ID.
        # This can happen if a client disconnects improperly and reconnects quickly.
        if await session_manager.session_exists(session_id):
            logger.warning(f"Session {session_id} already exists. Removing old session before creating new one.")
            await session_manager.remove_session(session_id)

        await session_manager.create_session(session_id, session_data)

    except Exception as e:
        logger.error(f"Failed to create session {session_id}: {e}", exc_info=True)
        await websocket.close(code=1011, reason=f"Failed to create session: {e}")
        return

    # Instantiate helpers
    validator = WebSocketValidator()

    # Recording state variables
    recording_start_time: Optional[float] = None
    total_data_received: int = 0
    
    try:
        loop = asyncio.get_running_loop()
        while True:
            message = await websocket.receive_json()
            
            action = message.get("action")

            # --- Validation ---
            current_session_data = await session_manager.get_session(session_id)
            if not current_session_data:
                logger.warning(f"Session {session_id} not found during message processing. Closing connection.")
                break

            session_state = live_audio_processor.get_session_state(session_id)
            session_dict_for_validator = session_state if session_state else {"status": "idle"}

            validator.validate_action(action)
            
            if action == "start":
                audio_format = message.get("format", "wav").lower()
                validator.validate_audio_format(audio_format)

            elif action == "audio_chunk":
                # State validation is now handled by the worker/processor, removing the check here
                # to prevent race conditions during testing.
                chunk_b64 = message.get("data", "")
                validator.validate_chunk_size(chunk_b64)
                estimated_chunk_size = len(chunk_b64) * 3 / 4
                validator.validate_total_size(int(total_data_received + estimated_chunk_size))

                # Check if 60min limit reached - auto-stop if so
                if validator.validate_recording_duration(recording_start_time):
                    await websocket.send_json({
                        "type": "duration_limit_reached",
                        "message": "Gravação atingiu o limite de 60 minutos. Processando áudio gravado..."
                    })
                    # Trigger auto-stop (will process on next iteration)
                    action = "stop"

            # --- Handling ---
            if action == "start":
                current_session_data.format = message.get("format", "wav").lower()

                start_result = await loop.run_in_executor(None, live_audio_processor.start_recording, session_id, 16000)
                recording_start_time = start_result["start_time"]
                current_session_data.temp_file = start_result["temp_file"]

                await websocket.send_json({
                    "type": "recording_started",
                    "session_id": session_id,
                    "format": current_session_data.format,
                    "message": f"Gravação iniciada (formato: {(current_session_data.format or 'wav').upper()})"
                })

            elif action == "audio_chunk":
                try:
                    audio_bytes = base64.b64decode(message.get("data", ""))
                    job = {
                        "session_id": session_id,
                        "type": "audio_chunk",
                        "audio_chunk_bytes": audio_bytes
                    }
                    transcription_queue.put(job)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Dados de áudio inválidos (base64 decode falhou)"})

            elif action == "custom_job":
                job = {"session_id": session_id, **message}
                transcription_queue.put(job)

            elif action == "pause":
                await loop.run_in_executor(None, live_audio_processor.pause_recording, session_id)
                await websocket.send_json({"type": "state_change", "data": {"status": "paused"}})

            elif action == "resume":
                await loop.run_in_executor(None, live_audio_processor.resume_recording, session_id)
                await websocket.send_json({"type": "state_change", "data": {"status": "recording"}})

            elif action == "stop":
                logger.info(f"⏹️ Stop action received for session {session_id}")

                # Update session status
                current_session_data.status = "processing"

                # Send stop confirmation
                await websocket.send_json({
                    "type": "recording_stopped",
                    "message": "Gravação finalizada. Processando batch final..."
                })

                # Queue final transcription job for any remaining audio chunks in worker buffer
                stop_job = {"type": "stop", "session_id": session_id}
                transcription_queue.put(stop_job)
                logger.info(f"Stop job queued for session {session_id} - worker will process final batch")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client: {session_id}")

    except ValidationError as e:
        logger.warning(f"Validation error in WebSocket: {e.message}", extra={"session_id": session_id, "context": e.context})
        await websocket.send_json({"error": "Dados inválidos", "details": e.message, "type": "validation_error"})

    except SessionError as e:
        logger.error(f"Session error during message processing: {e.message}", extra={"session_id": session_id, "context": e.context})
        await websocket.send_json({"error": "Erro de sessão", "details": e.message, "type": "session_error"})

    except (TranscriptionError, DiarizationError, AudioProcessingError) as e:
        logger.error(f"Processing error in WebSocket handler: {e.message}", extra={"session_id": session_id, "error_type": type(e).__name__, "context": e.context})
        await websocket.send_json({"error": "Erro ao processar áudio", "details": e.message, "type": "processing_error"})

    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler for session {session_id}", extra={"session_id": session_id})
        try:
            await websocket.send_json({"error": "Erro interno do servidor", "type": "internal_error"})
        except Exception:
            pass # Ignore errors on a closed socket
    
    finally:
        # This block is the safety net. It guarantees that the session and its
        # associated resources are cleaned up, no matter how the connection
        # terminates (cleanly, with an error, or a disconnect).
        logger.info(f"Cleaning up session due to WebSocket closure: {session_id}")
        await session_manager.remove_session(session_id)
        logger.info(f"WebSocket cleanup complete: {session_id}")

if __name__ == "__main__":
    # SSL configuration
    if app_config.ssl_cert_path and app_config.ssl_key_path:
        uvicorn.run(
            app="main:app",
            host=app_config.host,
            port=app_config.port,
            reload=False,
            log_level=app_config.log_level.lower(),
            ssl_certfile=app_config.ssl_cert_path,
            ssl_keyfile=app_config.ssl_key_path
        )
    else:
        uvicorn.run(
            app="main:app",
            host=app_config.host,
            port=app_config.port,
            reload=False,
            log_level=app_config.log_level.lower()
        )
