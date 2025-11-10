import asyncio
import base64
import logging
import os
import time
import uuid
import hashlib
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
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
from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from src.audio_processing import AudioQualityAnalyzer, SessionManager, LiveAudioProcessor, SessionData
from datetime import datetime
from src.file_manager import FileManager
from src.pipeline import process_audio_pipeline
from src.exceptions import TranscrevAIError, TranscriptionError, DiarizationError, AudioProcessingError, SessionError, ValidationError
from config.app_config import get_config

# --- Global Configuration & Tuning
app_config = get_config()
logging.basicConfig(level=app_config.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Dynamic cache busting
def calculate_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of file content for cache busting."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except:
        return str(int(time.time()))  # Fallback to timestamp

# Calculate static file hashes at startup
STATIC_HASHES = {
    'app.js': calculate_file_hash(Path('static/app.js')),
    'styles.css': calculate_file_hash(Path('static/styles.css'))
}

# Hardware-aware adaptive performance tuning
from src.audio_processing import configure_adaptive_threads

torch_threads, omp_threads = configure_adaptive_threads()
torch.set_num_threads(torch_threads)
os.environ["OMP_NUM_THREADS"] = str(omp_threads)
logger.info(f"PERFORMANCE TUNING APPLIED: OMP_NUM_THREADS={omp_threads}, torch_threads={torch_threads}")
DEVICE = "cpu"

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

# --- Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TranscrevAI Server Starting...")
    logger.info("Initializing Machine Learning services...")

    # Initialize services via DI - triggers singleton creation
    get_file_manager()

    # Get transcription service and pre-load whisper model
    transcription_service = get_transcription_service()
    logger.info("Pre-loading Whisper model...")
    await transcription_service.initialize()
    logger.info("Whisper model loaded successfully")

    get_diarization_service()
    get_audio_quality_analyzer()
    get_session_manager()
    get_live_audio_processor()
    get_transcription_queue()

    # Get running loop - pass to worker
    loop = asyncio.get_running_loop()
    get_worker_thread(loop)

    logger.info("Services and worker initialized successfully.")
    yield

    # --- Shutdown logic
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

# --- FastAPI app setup
app = FastAPI(title="TranscrevAI Single-Process", version="6.0.0", lifespan=lifespan)

# Rate limiting configuration
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

# --- API endpoints
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "static_hashes": STATIC_HASHES
    })

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

    # File size validation
    MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB
    file_size = 0
    content = await file.read()
    file_size = len(content)

    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande. Tamanho máximo: 500MB. Tamanho recebido: {file_size / (1024*1024):.1f}MB"
        )

    # Reset file pointer for later processing
    await file.seek(0)

    # Use the provided session_id if it exists - otherwise create new
    session_id = session_id or str(uuid.uuid4())
    logger.info(f"Upload received ({file_size / (1024*1024):.1f}MB), processing for session {session_id}")

    try:
        file_path = await file_manager.save_uploaded_file(file, file.filename or f"{session_id}.wav")

        # Validate audio duration
        import librosa
        try:
            duration = librosa.get_duration(path=str(file_path))
            if duration > 3600:  # 60 min
                Path(file_path).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Áudio muito longo. Duração máxima: 60 minutos. Duração: {duration/60:.1f} min"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Could not validate duration for {file_path}: {e}")

        # Get existing session or create new one
        session = await session_manager.get_session(session_id)
        if not session:
            # Create a session data object for this upload
            session_data = SessionData(
                session_id=session_id,
                websocket=None, # No websocket for uploads
                format=Path(file_path).suffix,
                started_at=datetime.now(),
                temp_file=str(file_path) # Uploaded file
            )
            await session_manager.create_session(session_id, session_data)
        else:
            # Update existing session with file path
            session.temp_file = str(file_path)
            session.format = Path(file_path).suffix

        # Run the processing pipeline in background
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
    
    # The new SessionData object stores file paths in a 'files' dictionary
    srt_path = session.files.get("subtitles") if session else None

    if not srt_path or not os.path.exists(srt_path):
        return JSONResponse(status_code=404, content={"error": "SRT file not found or not yet generated."})
    
    return FileResponse(path=srt_path, filename=f"transcription_{session_id}.srt", media_type="application/x-subrip")

@app.get("/api/download/{session_id}/{file_type}")
async def download_file(session_id: str, file_type: str, session_manager: SessionManager = Depends(get_session_manager)):
    valid_types = ['audio', 'transcript', 'subtitles']
    if file_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Must be one of: {valid_types}")

    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get file path from the session's 'files' dictionary
    file_path_str = session.files.get(file_type)
    if not file_path_str or not Path(file_path_str).exists():
        raise HTTPException(status_code=404, detail=f"{file_type.capitalize()} file not found for this session")

    file_path = Path(file_path_str)
    audio_format = getattr(session, 'format', 'wav') # Get format from session

    # MP4 conversion for audio files
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

from src.websocket_handler import WebSocketValidator
from collections import defaultdict
from time import time as current_time

# WebSocket rate limiting - manual implementation - slowapi doesn't support direct implementation
ws_connection_tracker = defaultdict(list)
WS_RATE_LIMIT = 20  # connections per min
WS_RATE_WINDOW = 60  # sec

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
    live_audio_processor: LiveAudioProcessor = Depends(get_live_audio_processor)
):
    """
    WebSocket endpoint for live audio recording - creates and removes sessions exclusively through the SessionManager
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
        # Defensively check for and remove any lingering session with the same ID
        if await session_manager.session_exists(session_id):
            logger.warning(f"Session {session_id} already exists. Removing old session before creating new one.")
            await session_manager.remove_session(session_id)

        await session_manager.create_session(session_id, session_data)

    except Exception as e:
        logger.error(f"Failed to create session {session_id}: {e}", exc_info=True)
        await websocket.close(code=1011, reason=f"Failed to create session: {e}")
        return

    # Instantiate helpers - validator resulted from class instantiation, validates actions and formats
    validator = WebSocketValidator()

    try:
        loop = asyncio.get_running_loop()

        while True:
            message = await websocket.receive_json()
            action = message.get("action")

            # Handle heartbeat ping (keep connection alive)
            if action == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # --- Validation
            current_session_data = await session_manager.get_session(session_id)

            if not current_session_data:
                logger.warning(f"Session {session_id} not found during message processing. Closing connection.")
                break

            session_state = live_audio_processor.get_session_state(session_id)
            validator.validate_action(action)

            if action == "start":
                audio_format = message.get("format", "wav").lower()
                validator.validate_audio_format(audio_format)

            # --- Handling
            if action == "start":
                current_session_data.format = message.get("format", "wav").lower()

                start_result = await loop.run_in_executor(None, live_audio_processor.start_recording, session_id, 16000)

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
                    await loop.run_in_executor(None, live_audio_processor.process_audio_chunk, session_id, audio_bytes)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Dados de áudio inválidos (base64 decode falhou)"})

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
                    "message": "Gravação finalizada. Processando áudio gravado..."
                })

                # Stop recording and get WAV path (converts WebM → WAV)
                wav_path = await loop.run_in_executor(None, live_audio_processor.stop_recording, session_id)
                current_session_data.temp_file = wav_path

                if not wav_path or not os.path.exists(wav_path):
                    logger.error(f"Audio file not found for session {session_id} at path: {wav_path}")
                    raise AudioProcessingError("Arquivo de áudio gravado não encontrado para processamento.", context={"session_id": session_id})

                # Run the processing pipeline in background 
                asyncio.create_task(process_audio_pipeline(wav_path, session_id))
                logger.info(f"Audio processing started for session {session_id}.")

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
            pass # ignore errors on a closed socket
    
    finally:
        # Guarantees session and associated resources clean-up
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
