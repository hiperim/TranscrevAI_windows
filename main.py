# Final Architecture: Single-Process, Async Model
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
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# Eagerly import torch and set threads at the very beginning
import torch
import psutil

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Core application modules
from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from src.audio_processing import AudioQualityAnalyzer, SessionManager
from src.subtitle_generator import generate_srt
from src.file_manager import FileManager
from src.websocket_enhancements import get_websocket_safety_manager, MessagePriority
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

# --- Application State ---
class AppState:
    """Holds the global state and initialized services for the application."""
    def __init__(self):
        self._lock = threading.RLock()
        self.sessions: Dict[str, Dict] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.websocket_safety_manager = get_websocket_safety_manager()
        self.file_manager = FileManager()

        # Services will be initialized during startup
        self.transcription_service: Optional[TranscriptionService] = None
        self.diarization_service: Optional[PyannoteDiarizer] = None
        self.audio_quality_analyzer: Optional[AudioQualityAnalyzer] = None
        self.session_manager: Optional[SessionManager] = None

    async def send_message(self, session_id: str, message: Dict, priority: MessagePriority = MessagePriority.NORMAL):
        await self.websocket_safety_manager.safe_send_message(self, session_id, message, priority)

app_state = AppState()

# --- Lifespan Manager (for startup and shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TranscrevAI Server Starting...")
    logger.info("Initializing Machine Learning services...")

    # Initialize services once and store them in the global state
    app_state.transcription_service = TranscriptionService(model_name="medium", device=DEVICE)
    await app_state.transcription_service.initialize()
    app_state.diarization_service = PyannoteDiarizer(device=DEVICE)
    app_state.audio_quality_analyzer = AudioQualityAnalyzer()

    # Initialize SessionManager for live recording
    app_state.session_manager = SessionManager(session_timeout_hours=24)

    # Start background cleanup task
    cleanup_task = asyncio.create_task(app_state.session_manager.cleanup_old_sessions())
    logger.info("SessionManager initialized and cleanup task started")

    logger.info("Services initialized successfully.")
    yield

    # Shutdown: cleanup all active sessions
    logger.info("Shutting down TranscrevAI...")
    if app_state.session_manager:
        session_count = app_state.session_manager.get_active_session_count()
        logger.info(f"Cleaning up {session_count} active live recording sessions...")

        for session_id in app_state.session_manager.get_all_session_ids():
            app_state.session_manager.delete_session(session_id)

        logger.info("All live recording sessions cleaned up")

    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

# --- Audio Processing Pipeline ---
async def process_audio_pipeline(audio_path: str, session_id: str):
    """The complete, non-blocking pipeline for processing a single audio file."""
    try:
        # Ensure services are initialized
        if not all([app_state.transcription_service, app_state.diarization_service, app_state.audio_quality_analyzer]):
            raise RuntimeError("Application services are not initialized.")

        import librosa
        audio_duration = librosa.get_duration(path=audio_path)
        pipeline_start_time = time.time()

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'start', 'percentage': 5, 'message': 'Analisando qualidade do áudio...'})
        
        # --- Run Pipeline Steps ---
        # 1. Transcription
        transcription_result = await app_state.transcription_service.transcribe_with_enhancements(audio_path)

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'diarization', 'percentage': 50, 'message': 'Transcrição concluída. Identificando falantes...'})

        # 2. Diarization
        diarization_result = await app_state.diarization_service.diarize(audio_path, transcription_result.segments)

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'srt', 'percentage': 80, 'message': 'Gerando legendas...'})

        # 3. Subtitle Generation
        srt_path = await generate_srt(diarization_result["segments"], output_path=app_state.file_manager.get_data_path("temp"), filename=f"{session_id}.srt")

        # --- Finalize and Report --- 
        pipeline_end_time = time.time()
        actual_processing_time = pipeline_end_time - pipeline_start_time
        processing_ratio = actual_processing_time / audio_duration if audio_duration > 0 else 0

        logger.info(f"Processing complete for {session_id}: {actual_processing_time:.2f}s for {audio_duration:.2f}s audio (ratio: {processing_ratio:.2f}x)")

        final_result = {
            "segments": diarization_result["segments"],
            "num_speakers": diarization_result["num_speakers"],
            "processing_time": round(actual_processing_time, 2),
            "processing_ratio": round(processing_ratio, 2),
            "audio_duration": round(audio_duration, 2)
        }

        with app_state._lock:
            if session_id not in app_state.sessions: app_state.sessions[session_id] = {}
            app_state.sessions[session_id]['srt_file_path'] = srt_path

        await app_state.send_message(session_id, {'type': 'complete', 'result': final_result}, MessagePriority.CRITICAL)

    except Exception as e:
        logger.error(f"Audio pipeline failed for session {session_id}: {e}", exc_info=True)
        await app_state.send_message(session_id, {'type': 'error', 'message': str(e)}, MessagePriority.CRITICAL)

# --- FastAPI App Setup ---
app = FastAPI(title="TranscrevAI Single-Process", version="6.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === API Endpoints ===
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    try:
        if not session_id: session_id = f"upload_{int(time.time() * 1000)}"
        file_path = app_state.file_manager.save_uploaded_file(file.file, file.filename or f"{session_id}.wav")
        
        # BEST PRACTICE: Run the heavy CPU-bound task in the background
        # This allows the endpoint to return immediately, keeping the server responsive.
        asyncio.create_task(process_audio_pipeline(str(file_path), session_id))
        
        logger.info(f"Job accepted for session {session_id}. Processing started in background.")
        return JSONResponse(content={"success": True, "session_id": session_id})
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download-srt/{session_id}")
async def download_srt(session_id: str):
    with app_state._lock:
        session = app_state.sessions.get(session_id)
    if not session or not session.get("srt_file_path") or not os.path.exists(session["srt_file_path"]):
        return JSONResponse(status_code=404, content={"error": "Arquivo SRT não encontrado."})
    return FileResponse(path=session["srt_file_path"], filename=f"transcricao_{session_id}.srt", media_type="application/x-subrip")

@app.get("/api/download/{session_id}/{file_type}")
async def download_file(session_id: str, file_type: str):
    """
    Download recorded files from live recording sessions.

    Args:
        session_id: Session ID from SessionManager
        file_type: One of ['audio', 'transcript', 'subtitles']

    Returns:
        FileResponse with the requested file

    Raises:
        HTTPException 400: Invalid file type
        HTTPException 404: Session or file not found
    """
    # Validate file type
    valid_types = ['audio', 'transcript', 'subtitles']
    if file_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Must be one of: {', '.join(valid_types)}"
        )

    # Get session from SessionManager
    if not app_state.session_manager:
        raise HTTPException(status_code=503, detail="SessionManager not initialized")

    session = app_state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get file path from session
    file_path = session.get("files", {}).get(file_type)
    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"{file_type.capitalize()} file not found for this session"
        )

    # Define file extensions and media types
    # For audio, detect format from actual file extension (WAV or MP4)
    if file_type == 'audio':
        file_ext = Path(file_path).suffix.lower()  # .wav or .mp4
        media_type = 'video/mp4' if file_ext == '.mp4' else 'audio/wav'
        filename = f"recording_{session_id}{file_ext}"
    else:
        file_config = {
            'transcript': {'ext': '.txt', 'media_type': 'text/plain'},
            'subtitles': {'ext': '.srt', 'media_type': 'application/x-subrip'}
        }
        config = file_config[file_type]
        filename = f"recording_{session_id}{config['ext']}"
        media_type = config['media_type']

    logger.info(f"Download requested: session={session_id}, file_type={file_type}, path={file_path}")

    return FileResponse(
        path=file_path,
        media_type=config['media_type'],
        filename=filename
    )

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}

# Security limits for live recording
MAX_RECORDING_DURATION = 3600  # 1 hour in seconds
MAX_CHUNK_SIZE = 1 * 1024 * 1024  # 1MB per chunk
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB total

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for live audio recording.

    Handles:
    - Real-time audio chunk streaming
    - Start/stop recording commands
    - Progress updates during processing
    - Error handling and validation

    Message format:
    - Start: {"action": "start", "format": "wav|mp4"}
    - Chunk: {"action": "audio_chunk", "data": "<base64-encoded-audio>"}
    - Stop: {"action": "stop"}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    # Register connection
    with app_state._lock:
        app_state.connections[session_id] = websocket
    await app_state.websocket_safety_manager.handle_connection_established(session_id)

    # Get or create session from SessionManager
    if not app_state.session_manager:
        await websocket.send_json({"type": "error", "message": "SessionManager not initialized"})
        await websocket.close()
        return

    session = app_state.session_manager.get_session(session_id)
    if not session:
        session_id = app_state.session_manager.create_session()
        session = app_state.session_manager.get_session(session_id)
        logger.info(f"Created new session: {session_id}")

    # Get LiveAudioProcessor from session
    processor = session.get("processor")
    if not processor:
        await websocket.send_json({"type": "error", "message": "Processor not found"})
        await websocket.close()
        return

    # Recording state
    recording_start = None
    total_data_received = 0
    audio_format = "wav"  # default

    try:
        while True:
            # Receive message from client
            message = await websocket.receive_json()
            action = message.get("action")

            # Handle "start" action
            if action == "start":
                logger.info(f"▶️ Starting recording for session {session_id}")

                # Get format choice (wav or mp4)
                audio_format = message.get("format", "wav").lower()
                if audio_format not in ["wav", "mp4"]:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid format. Use 'wav' or 'mp4'"
                    })
                    continue

                # Start recording
                await processor.start_recording(session_id)
                session["status"] = "recording"
                session["audio_format"] = audio_format
                recording_start = time.time()
                total_data_received = 0

                await websocket.send_json({
                    "type": "recording_started",
                    "session_id": session_id,
                    "format": audio_format,
                    "message": f"Gravação iniciada (formato: {audio_format.upper()})"
                })

            # Handle "audio_chunk" action
            elif action == "audio_chunk":
                # Validate recording state
                if session.get("status") != "recording":
                    await websocket.send_json({
                        "type": "error",
                        "message": "Gravação não está ativa. Use 'start' primeiro."
                    })
                    continue

                # Validate chunk size
                chunk_b64 = message.get("data", "")
                if len(chunk_b64) > MAX_CHUNK_SIZE * 1.5:  # Base64 adds ~33% overhead
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Chunk muito grande (max {MAX_CHUNK_SIZE // 1024 // 1024}MB)"
                    })
                    break

                # Decode base64
                try:
                    audio_data = base64.b64decode(chunk_b64)
                except Exception as e:
                    logger.error(f"Failed to decode audio chunk: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Dados de áudio inválidos (base64 decode falhou)"
                    })
                    continue

                # Check total size limit
                total_data_received += len(audio_data)
                if total_data_received > MAX_FILE_SIZE:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Arquivo muito grande (max {MAX_FILE_SIZE // 1024 // 1024}MB)"
                    })
                    break

                # Check duration limit
                elapsed_time = time.time() - recording_start
                if elapsed_time > MAX_RECORDING_DURATION:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Gravação muito longa (max {MAX_RECORDING_DURATION // 60} minutos)"
                    })
                    break

                # Process chunk
                await processor.process_audio_chunk(session_id, audio_data)

                # Send periodic progress updates (every 10MB)
                if total_data_received % (10 * 1024 * 1024) < len(audio_data):
                    await websocket.send_json({
                        "type": "recording_progress",
                        "data_received_mb": round(total_data_received / (1024 * 1024), 2),
                        "duration_sec": round(elapsed_time, 2)
                    })

            # Handle "stop" action
            elif action == "stop":
                logger.info(f"⏹️ Stopping recording for session {session_id}")

                # Stop recording and get WAV file
                wav_path = await processor.stop_recording(session_id)
                session["status"] = "processing"

                await websocket.send_json({
                    "type": "recording_stopped",
                    "message": "Gravação finalizada. Processando..."
                })

                # Convert to MP4 if requested
                if audio_format == "mp4":
                    # TODO: Implement WAV to MP4 conversion
                    # For now, just use WAV
                    logger.warning("MP4 conversion not yet implemented, using WAV")
                    audio_path = wav_path
                else:
                    audio_path = wav_path

                # Store audio file path
                session["files"]["audio"] = str(audio_path)

                # Process audio (transcription + diarization)
                await websocket.send_json({
                    "type": "progress",
                    "stage": "transcription",
                    "percentage": 10,
                    "message": "Iniciando transcrição..."
                })

                # Run audio pipeline in background
                asyncio.create_task(process_audio_pipeline(str(audio_path), session_id))

                await websocket.send_json({
                    "type": "processing_started",
                    "message": "Processamento iniciado em background"
                })

            # Unknown action
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Ação desconhecida: '{action}'. Use: start, audio_chunk, stop"
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": f"Erro interno: {str(e)}"})
        except:
            pass
    finally:
        await app_state.websocket_safety_manager.handle_connection_lost(session_id)
        with app_state._lock:
            if session_id in app_state.connections:
                del app_state.connections[session_id]
        logger.info(f"WebSocket cleanup complete: {session_id}")

if __name__ == "__main__":
    # SSL configuration
    ssl_config = {}
    if app_config.ssl_cert_path and app_config.ssl_key_path:
        ssl_config = {
            "ssl_certfile": app_config.ssl_cert_path,
            "ssl_keyfile": app_config.ssl_key_path
        }
        logger.info(f"Starting server with SSL on https://{app_config.host}:{app_config.port}")
    else:
        logger.info(f"Starting server without SSL on http://{app_config.host}:{app_config.port}")

    uvicorn.run(
        app="main:app",
        host=app_config.host,
        port=app_config.port,
        reload=False,
        log_level=app_config.log_level.lower(),
        **ssl_config
    )
