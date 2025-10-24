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
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# Eagerly import torch and set threads at the very beginning
import torch
import psutil

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Core application modules
from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from src.audio_processing import AudioQualityAnalyzer
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
    
    logger.info("Services initialized successfully.")
    yield
    logger.info("Shutting down TranscrevAI...")

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

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    with app_state._lock: app_state.connections[session_id] = websocket
    await app_state.websocket_safety_manager.handle_connection_established(session_id)
    try:
        # Keep the connection alive to receive progress updates
        while True: await asyncio.sleep(3600)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    finally:
        await app_state.websocket_safety_manager.handle_connection_lost(session_id)
        with app_state._lock:
            if session_id in app_state.connections: del app_state.connections[session_id]

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
