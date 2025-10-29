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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Core application modules
from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from src.audio_processing import AudioQualityAnalyzer, SessionManager, LiveAudioProcessor
from src.subtitle_generator import generate_srt
from src.file_manager import FileManager
from src.websocket_enhancements import get_websocket_safety_manager, MessagePriority
from src.pipeline import run_pipeline_sync
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



# --- FastAPI App Setup ---
app = FastAPI(title="TranscrevAI Single-Process", version="6.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === API Endpoints ===
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    try:
        # Ensure a session exists in the SessionManager
        if not app_state.session_manager:
            raise RuntimeError("SessionManager not initialized - application startup failed")

        if not session_id:
            session_id = app_state.session_manager.create_session()
        else:
            # If a session_id is provided, ensure it is registered
            if not app_state.session_manager.get_session(session_id):
                app_state.session_manager.sessions[session_id] = {
                    "id": session_id,
                    "created_at": time.time(),
                    "last_activity": time.time(),
                    "processor": None,  # No live processor for uploads
                    "files": {},
                    "status": "starting"
                }

        if not app_state.file_manager:
            raise RuntimeError("FileManager not initialized - application startup failed")
        file_path = app_state.file_manager.save_uploaded_file(file.file, file.filename or f"{session_id}.wav")
        
        # Store the audio file path in the session
        session = app_state.session_manager.get_session(session_id)
        if session:
            session["files"]["audio"] = str(file_path)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, run_pipeline_sync, app_state, str(file_path), session_id)
        
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

    if not app_state.session_manager:
        raise RuntimeError("SessionManager not initialized - application startup failed")
    session = app_state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get file path from session
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    file_path_str = session.get("files", {}).get(file_type)
    if not file_path_str or not Path(file_path_str).exists():
        raise HTTPException(
            status_code=404,
            detail=f"{file_type.capitalize()} file not found for this session"
        )

    file_path = Path(file_path_str)

    # On-demand MP4 conversion for audio files
    if file_type == 'audio' and session.get("audio_format") == 'mp4' and file_path.suffix.lower() == '.wav':
        from src.audio_processing import convert_wav_to_mp4
        mp4_path = file_path.with_suffix(".mp4")

        # Get subtitle file path if available
        subtitle_path_str = session.get("files", {}).get("subtitles")
        subtitle_path = subtitle_path_str if subtitle_path_str and Path(subtitle_path_str).exists() else None

        # Convert WAV to MP4 video with black background and embedded subtitles (if available)
        if convert_wav_to_mp4(str(file_path), str(mp4_path), subtitle_path=subtitle_path):
            logger.info(f"MP4 video created successfully for session {session_id} (with subtitles: {subtitle_path is not None})")
            return FileResponse(
                path=str(mp4_path),
                media_type='video/mp4',
                filename=f"recording_{session_id}.mp4"
            )
        else:
            # Fallback to serving the WAV if conversion fails
            logger.warning(f"MP4 conversion failed for session {session_id}. Serving original WAV file.")

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

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}

from src.websocket_handler import WebSocketValidator, WebSocketHandler

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for live audio recording.
    This endpoint acts as a dispatcher, delegating validation and handling
    to the WebSocketValidator and WebSocketHandler classes.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    # Register connection and session
    with app_state._lock:
        app_state.connections[session_id] = websocket
    await app_state.websocket_safety_manager.handle_connection_established(session_id)

    if not app_state.session_manager:
        await websocket.close(code=1011, reason="SessionManager not initialized")
        return
    
    session = app_state.session_manager.get_session(session_id)
    if not session:
        session = app_state.session_manager.create_session(session_id)
        logger.info(f"Created new session with client ID: {session_id}")

    if not session or not session.get("processor"):
        await websocket.close(code=1011, reason="Failed to get or create a valid session/processor")
        return

    # Instantiate helpers
    handler = WebSocketHandler(app_state)
    validator = WebSocketValidator()

    # Recording state variables
    recording_start_time: Optional[float] = None
    total_data_received: int = 0
    
    try:
        while True:
            message = await websocket.receive_json()
            action = message.get("action")

            # --- Validation ---
            error_msg = validator.validate_action(action)
            if error_msg:
                await websocket.send_json({"type": "error", "message": error_msg})
                continue
            
            if action == "start":
                audio_format = message.get("format", "wav").lower()
                error_msg = validator.validate_audio_format(audio_format)
            elif action == "audio_chunk":
                error_msg = validator.validate_session_state_for_chunk(session)
                if not error_msg:
                    chunk_b64 = message.get("data", "")
                    error_msg = validator.validate_chunk_size(chunk_b64)
                if not error_msg:
                    estimated_chunk_size = len(chunk_b64) * 3 / 4
                    error_msg = validator.validate_total_size(total_data_received + estimated_chunk_size)
                if not error_msg:
                    error_msg = validator.validate_recording_duration(recording_start_time)
            
            if error_msg:
                await websocket.send_json({"type": "error", "message": error_msg})
                if action in ["audio_chunk", "start"]:
                    break
                continue

            # --- Handling ---
            if action == "start":
                start_time, data_received = await handler.handle_start(message, session_id, websocket)
                recording_start_time = start_time
                total_data_received = data_received

            elif action == "audio_chunk":
                bytes_received, error_occurred = await handler.handle_chunk(message, session_id, websocket)
                if error_occurred:
                    continue
                total_data_received += bytes_received

            elif action == "stop":
                await handler.handle_stop(session_id, websocket)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": f"Erro interno: {str(e)}"})
        except:
            pass
    finally:
        # Cleanup
        await app_state.websocket_safety_manager.handle_connection_lost(session_id)
        with app_state._lock:
            if session_id in app_state.connections:
                del app_state.connections[session_id]
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
