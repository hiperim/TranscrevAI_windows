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
from src.audio_processing import AudioQualityAnalyzer, SessionManager, LiveAudioProcessor, SessionData
from datetime import datetime
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
    """Central application state container.

    Manages references to core services without duplicating session storage.
    Session data is exclusively managed by SessionManager.
    """    
    def __init__(self):
        self._lock = threading.RLock()
        self.websocket_safety_manager = get_websocket_safety_manager()
        self.file_manager = FileManager()

    # Services will be initialized during startup
    transcription_service: Optional[TranscriptionService] = None
    diarization_service: Optional[PyannoteDiarizer] = None
    audio_quality_analyzer: Optional[AudioQualityAnalyzer] = None
    session_manager: Optional[SessionManager] = None

app_state = AppState()

# --- Lifespan Manager (for startup and shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TranscrevAI Server Starting...")
    logger.info("Initializing Machine Learning services...")

    # Initialize services once and store them in the global state
    app_state.transcription_service = TranscriptionService(model_name="medium", device=DEVICE)
    app_state.diarization_service = PyannoteDiarizer(device=DEVICE)
    app_state.audio_quality_analyzer = AudioQualityAnalyzer()
    app_state.session_manager = SessionManager(session_timeout_hours=24)

    logger.info("SessionManager initialized.")
    logger.info("Services initialized successfully.")
    yield

    # --- Shutdown Logic ---
    logger.info("Shutting down TranscrevAI...")
    if app_state.session_manager:
        all_sessions = await app_state.session_manager.get_all_session_ids()
        logger.info(f"Cleaning up {len(all_sessions)} active sessions...")
        for session_id in all_sessions:
            await app_state.session_manager.remove_session(session_id)
        logger.info("All sessions cleaned up.")

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
    if not app_state.session_manager or not app_state.file_manager:
        raise HTTPException(status_code=503, detail="Core services not initialized")

    # Use the provided session_id if it exists, otherwise create a new one.
    session_id = session_id or str(uuid.uuid4())
    logger.info(f"Upload received, processing for session {session_id}")

    try:
        # Save the uploaded file first
        file_path = app_state.file_manager.save_uploaded_file(file.file, file.filename or f"{session_id}.wav")

        # Create a session data object for this upload
        session_data = SessionData(
            session_id=session_id,
            websocket=None, # No websocket for uploads
            format=Path(file_path).suffix,
            started_at=datetime.now(),
            temp_file=str(file_path) # The uploaded file is the temp file
        )
        await app_state.session_manager.create_session(session_id, session_data)

        # Run the processing pipeline in the background
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, run_pipeline_sync, app_state, str(file_path), session_id)
        
        logger.info(f"Job accepted for session {session_id}. Processing started in background.")
        return JSONResponse(content={"success": True, "session_id": session_id})
    except Exception as e:
        logger.error(f"Upload failed for session {session_id}: {e}", exc_info=True)
        # Ensure session is cleaned up on failure
        await app_state.session_manager.remove_session(session_id)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download-srt/{session_id}")
async def download_srt(session_id: str):
    if not app_state.session_manager:
        raise HTTPException(status_code=503, detail="SessionManager not initialized")

    session = await app_state.session_manager.get_session(session_id)
    
    # The new SessionData object stores file paths in a `files` dictionary.
    # This path is populated by the pipeline.
    srt_path = session.files.get("subtitles") if session else None

    if not srt_path or not os.path.exists(srt_path):
        return JSONResponse(status_code=404, content={"error": "SRT file not found or not yet generated."})
    
    return FileResponse(path=srt_path, filename=f"transcription_{session_id}.srt", media_type="application/x-subrip")

@app.get("/api/download/{session_id}/{file_type}")
async def download_file(session_id: str, file_type: str):
    """
    Download recorded files from live recording sessions.
    """
    valid_types = ['audio', 'transcript', 'subtitles']
    if file_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Must be one of: {valid_types}")

    if not app_state.session_manager:
        raise HTTPException(status_code=503, detail="SessionManager not initialized")

    session = await app_state.session_manager.get_session(session_id)
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

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for live audio recording, refactored for centralized session management.
    
    Creates and removes sessions exclusively through the SessionManager, ensuring a single
    source of truth and proper resource cleanup.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for session: {session_id}")

    if not app_state.session_manager:
        await websocket.close(code=1011, reason="SessionManager not initialized")
        return

    # Create a new session data object for this connection
    session_data = SessionData(
        session_id=session_id,
        websocket=websocket,
        format=None, # Will be set on 'start' message
        started_at=datetime.now()
    )

    try:
        # Register the new session. If a session with this ID somehow still exists, 
        # remove the old one to prevent conflicts.
        if await app_state.session_manager.session_exists(session_id):
            logger.warning(f"Session {session_id} already exists. Removing old session before creating new one.")
            await app_state.session_manager.remove_session(session_id)
        
        await app_state.session_manager.create_session(session_id, session_data)

    except Exception as e:
        logger.error(f"Failed to create session {session_id}: {e}", exc_info=True)
        await websocket.close(code=1011, reason=f"Failed to create session: {e}")
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
            # The validator needs the session object, get it from the manager
            current_session_data = await app_state.session_manager.get_session(session_id)
            if not current_session_data:
                logger.warning(f"Session {session_id} not found during validation. Closing connection.")
                break # Exit loop, finally block will handle cleanup

            # Convert SessionData object to a dict for the validator if it expects a dict
            # NOTE: This part might need adjustment if validator is also refactored
            session_dict_for_validator = {
                "status": current_session_data.status if hasattr(current_session_data, 'status') else 'recording'
            }

            error_msg = validator.validate_action(action)
            if error_msg:
                await websocket.send_json({"type": "error", "message": error_msg})
                continue
            
            if action == "start":
                audio_format = message.get("format", "wav").lower()
                error_msg = validator.validate_audio_format(audio_format)
            elif action == "audio_chunk":
                error_msg = validator.validate_session_state_for_chunk(session_dict_for_validator)
                if not error_msg:
                    chunk_b64 = message.get("data", "")
                    error_msg = validator.validate_chunk_size(chunk_b64)
                if not error_msg:
                    estimated_chunk_size = len(chunk_b64) * 3 / 4
                    error_msg = validator.validate_total_size(int(total_data_received + estimated_chunk_size))
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
        logger.info(f"WebSocket disconnected by client: {session_id}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in WebSocket for session {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": f"Internal server error: {str(e)}"})
        except Exception:
            pass # Ignore errors on a closed socket
    finally:
        # Guaranteed cleanup: remove the session from the manager
        logger.info(f"Cleaning up session due to WebSocket closure: {session_id}")
        await app_state.session_manager.remove_session(session_id)
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
