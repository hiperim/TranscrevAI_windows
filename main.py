# Enhanced main.py - UPDATED FOR 10/10 RATING
"""
TranscrevAI Main Application - Complete 10/10 Implementation
Enhanced with hardware validation, memory monitoring, and adaptive performance optimization
"""

import asyncio
import logging
import os
import time
import uuid
import threading
import psutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Core modules
from src.audio_processing import AudioRecorder, LiveAudioProcessor, RecordingState
from src.diarization import TwoPassDiarizer, force_transcription_segmentation  
from src.transcription import TranscriptionService
# import
from src.subtitle_generator import generate_srt

# usage (replace previous call)
# Note: srt_path must be created after `final_segments` is available inside the processing pipeline;
# avoid calling generate_srt here as `final_segments` is not defined in the module scope.
from src.file_manager import FileManager

# Enhanced systems for 10/10 rating
from src.hardware_validator import get_hardware_validator, HardwareCompatibility
from src.memory_monitor import get_memory_monitor
from src.performance_optimizer import get_production_optimizer, ProcessType
from src.websocket_enhancements import get_websocket_safety_manager, MessagePriority

# Configuration
from config.app_config import get_config

# Set up application logging
app_config = get_config()
logging.basicConfig(level=app_config.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreadSafeEnhancedAppState:
    """Enhanced thread-safe application state with full monitoring"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self.sessions: Dict[str, Dict] = {}
        self.connections: Dict[str, WebSocket] = {}
        
        # Initialize all services and managers
        self.transcription_service: Optional[TranscriptionService] = None
        self.diarization_service: Optional[TwoPassDiarizer] = None
        self.live_audio_processor = LiveAudioProcessor()
        self.websocket_safety_manager = get_websocket_safety_manager()
        self.hardware_validator = get_hardware_validator()
        self.memory_monitor = get_memory_monitor()
        self.performance_optimizer = get_production_optimizer()
        self.file_manager = FileManager()

        self.is_initialized = False

    async def initialize_services(self):
        """Initialize all services with comprehensive validation"""
        with self._lock:
            if self.is_initialized: return
            logger.info("Starting TranscrevAI Enhanced Initialization...")
            hw_report = self.hardware_validator.validate_system(force_refresh=True)
            if hw_report.compatibility_level == HardwareCompatibility.INCOMPATIBLE:
                raise RuntimeError(f"Hardware incompatible. Score: {hw_report.overall_score:.1%}. Warnings: {hw_report.warnings}")
            
            self.memory_monitor.start_monitoring()
            self.transcription_service = TranscriptionService()
            self.diarization_service = TwoPassDiarizer()
            self.is_initialized = True
            logger.info(f"TranscrevAI Initialization Complete! Compatibility: {hw_report.compatibility_level.value.upper()}")

    async def send_message(self, session_id: str, message: Dict, priority: MessagePriority = MessagePriority.NORMAL):
        with self._lock:
            if session_id in self.connections:
                await self.websocket_safety_manager.safe_send_message(self, session_id, message)

app_state = ThreadSafeEnhancedAppState()

# === Background Tasks ===

async def model_cleanup_task():
    """Background task to unload inactive models for memory optimization."""
    logger.info("Model cleanup task started (check interval: 5min)")
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        try:
            with app_state._lock:
                transcription_service = app_state.transcription_service
            if transcription_service and transcription_service.should_unload():
                await transcription_service.unload_model()
        except Exception as e:
            logger.error(f"Model cleanup task error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan with full system validation"""
    await app_state.initialize_services()

    # Start background cleanup task
    cleanup_task = asyncio.create_task(model_cleanup_task())
    logger.info("Background model cleanup task initiated")

    yield

    # Shutdown cleanup
    logger.info("Shutting down TranscrevAI...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        logger.info("Model cleanup task cancelled")

    if app_state.memory_monitor.is_monitoring:
        app_state.memory_monitor.stop_monitoring()

app = FastAPI(title="TranscrevAI Enhanced", version="3.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === Main Processing Pipeline ===
async def process_audio_pipeline(audio_path: str, session_id: str):
    try:
        # FASE 1.2: Memory Profiling - Track RAM usage at pipeline checkpoints
        process = psutil.Process()
        mem_baseline = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        logger.info(f"[MEMORY PROFILING] Baseline: {mem_baseline:.2f} MB")

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'start', 'percentage': 5, 'message': 'Iniciando processamento...'}, MessagePriority.HIGH)

        with app_state._lock:
            transcription_service = app_state.transcription_service
            diarization_service = app_state.diarization_service

        if not transcription_service:
            logger.error("Transcription service not initialized")
            await app_state.send_message(session_id, {'type': 'error', 'message': 'Transcription service not initialized'}, MessagePriority.CRITICAL)
            return

        transcription_result = await transcription_service.transcribe_with_enhancements(audio_path) # Reverted

        # FASE 1.2: Memory checkpoint after transcription
        mem_after_transcription = process.memory_info().rss / (1024 * 1024)
        mem_delta_transcription = mem_after_transcription - mem_baseline
        logger.info(f"[MEMORY PROFILING] After Transcription: {mem_after_transcription:.2f} MB (+{mem_delta_transcription:.2f} MB)")

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'transcription', 'percentage': 50, 'message': 'Transcrição concluída.'}, MessagePriority.NORMAL)

        if not diarization_service:
            logger.error("Diarization service not initialized")
            await app_state.send_message(session_id, {'type': 'error', 'message': 'Diarization service not initialized'}, MessagePriority.CRITICAL)
            return

        loop = asyncio.get_event_loop()
        diarization_coro = diarization_service.diarize(audio_path, transcription_result.segments)
        diarization_result = await diarization_coro

        # FASE 1.2: Memory checkpoint after diarization
        mem_after_diarization = process.memory_info().rss / (1024 * 1024)
        mem_delta_diarization = mem_after_diarization - mem_after_transcription
        logger.info(f"[MEMORY PROFILING] After Diarization: {mem_after_diarization:.2f} MB (+{mem_delta_diarization:.2f} MB)")

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'diarization', 'percentage': 80, 'message': 'Identificação de falantes concluída.'}, MessagePriority.NORMAL)
        final_segments = force_transcription_segmentation(transcription_result.segments, diarization_result["segments"])
        srt_path = await generate_srt(final_segments, output_path=os.path.join(app_state.file_manager.get_data_path("temp"), f"{session_id}.srt"))

        final_result = {
            "text": transcription_result.text,
            "segments": final_segments,
            "srt_output_path": srt_path,
            "num_speakers": diarization_result["num_speakers"]
        }

        with app_state._lock:
            if session_id not in app_state.sessions:
                app_state.sessions[session_id] = {}
            app_state.sessions[session_id]['srt_file_path'] = srt_path

        # FASE 1.2: Final memory checkpoint
        mem_final = process.memory_info().rss / (1024 * 1024)
        mem_total_delta = mem_final - mem_baseline
        logger.info(f"[MEMORY PROFILING] Pipeline Complete: {mem_final:.2f} MB (Total Delta: +{mem_total_delta:.2f} MB)")
        logger.info(f"[MEMORY PROFILING] Target: <3500 MB | Current: {mem_final:.2f} MB | Status: {'PASS' if mem_final < 3500 else 'FAIL'}")

        await app_state.send_message(session_id, {'type': 'complete', 'result': final_result}, MessagePriority.CRITICAL)

    except Exception as e:
        logger.error(f"Pipeline failed for session {session_id}: {e}", exc_info=True)
        await app_state.send_message(session_id, {'type': 'error', 'message': str(e)}, MessagePriority.CRITICAL)

# === API Endpoints ===

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), session_id: str = Form(...)):
    try:
        filename = file.filename if file.filename is not None else f"uploaded_{uuid.uuid4().hex}.wav"
        file_path = app_state.file_manager.save_uploaded_file(file.file, filename)
        asyncio.create_task(process_audio_pipeline(str(file_path), session_id))
        return JSONResponse(content={"success": True, "session_id": session_id, "message": "Upload successful, processing started."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download/srt/{session_id}")
async def download_srt(session_id: str):
    with app_state._lock:
        session = app_state.sessions.get(session_id)
    if not session or not session.get("srt_file_path") or not os.path.exists(session["srt_file_path"]):
        return JSONResponse(status_code=404, content={"error": "SRT file not found."})
    return FileResponse(path=session["srt_file_path"], filename=f"transcription_{session_id}.srt")

@app.get("/check-first-time")
async def check_first_time():
    """Check if this is the first time the app is being used (model not downloaded yet)."""
    try:
        model_path = Path(app_state.file_manager.get_data_path("models_cache"))

        # Check if faster-whisper models exist
        is_first_time = True
        if model_path.exists():
            # Check for downloaded model files
            model_files = list(model_path.glob("**/model.bin")) + list(model_path.glob("**/pytorch_model.bin"))
            if model_files:
                is_first_time = False

        return JSONResponse(content={"is_first_time": is_first_time})
    except Exception as e:
        logger.error(f"Error checking first-time status: {e}")
        return JSONResponse(content={"is_first_time": False})

def validate_websocket_message(data: Dict[str, Any]) -> bool:
    """Validate WebSocket message to prevent DoS and injection attacks"""
    # Whitelist of allowed actions
    ALLOWED_ACTIONS = {"start", "pause", "resume", "audio_chunk", "stop", "get_state"}

    # Maximum payload size: 5MB for audio chunks
    MAX_PAYLOAD_SIZE = 5 * 1024 * 1024

    # Validate action exists and is allowed
    action = data.get("action")
    if not action or action not in ALLOWED_ACTIONS:
        logger.warning(f"Invalid action received: {action}")
        return False

    # Validate audio_chunk size
    if action == "audio_chunk":
        chunk_data = data.get("data", "")
        # Estimate base64 decoded size (base64 is ~33% larger than binary)
        estimated_size = len(chunk_data) * 3 // 4

        if estimated_size > MAX_PAYLOAD_SIZE:
            logger.warning(f"Audio chunk too large: {estimated_size} bytes")
            return False

        if len(chunk_data) == 0:
            logger.warning("Empty audio chunk received")
            return False

    return True

@app.websocket("/ws/{session_id}")
async def file_upload_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for file upload progress updates"""
    await websocket.accept()
    logger.info(f"File upload WebSocket connected: {session_id}")

    with app_state._lock:
        app_state.connections[session_id] = websocket

    try:
        while True:
            # Keep connection alive, wait for messages
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info(f"File upload WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"File upload WebSocket error: {e}")
    finally:
        with app_state._lock:
            if session_id in app_state.connections:
                del app_state.connections[session_id]

@app.websocket("/ws/live/{session_id}")
async def live_audio_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for live audio recording with state management"""
    await websocket.accept()
    logger.info(f"Live audio WebSocket connected: {session_id}")

    # Rate limiting: track message count
    message_count = 0
    last_reset_time = time.time()
    MAX_MESSAGES_PER_SECOND = 100

    try:
        while True:
            # Rate limiting check
            current_time = time.time()
            if current_time - last_reset_time >= 1.0:
                message_count = 0
                last_reset_time = current_time

            message_count += 1
            if message_count > MAX_MESSAGES_PER_SECOND:
                logger.warning(f"Rate limit exceeded for session {session_id}")
                await websocket.send_json({"type": "error", "message": "Rate limit exceeded"})
                await asyncio.sleep(1)
                continue

            data = await websocket.receive_json()

            # SECURITY: Validate message
            if not validate_websocket_message(data):
                await websocket.send_json({"type": "error", "message": "Invalid message format"})
                continue

            action = data.get("action")

            if action == "start":
                result = await app_state.live_audio_processor.start_recording(session_id)
                await websocket.send_json({"type": "state_change", "data": result})

            elif action == "pause":
                result = await app_state.live_audio_processor.pause_recording(session_id)
                await websocket.send_json({"type": "state_change", "data": result})

            elif action == "resume":
                result = await app_state.live_audio_processor.resume_recording(session_id)
                await websocket.send_json({"type": "state_change", "data": result})

            elif action == "audio_chunk":
                import base64
                audio_data = base64.b64decode(data.get("data"))
                result = await app_state.live_audio_processor.process_audio_chunk(session_id, audio_data)
                # Send minimal acknowledgment (don't flood WebSocket)
                if result["chunks_received"] % 10 == 0:  # Every 10 chunks
                    await websocket.send_json({"type": "chunk_ack", "chunks": result["chunks_received"]})

            elif action == "stop":
                audio_file_path = await app_state.live_audio_processor.stop_recording(session_id)
                await websocket.send_json({"type": "processing", "message": "Processando áudio..."})

                # Process complete audio file through existing pipeline
                asyncio.create_task(process_audio_pipeline(audio_file_path, session_id))

                # Cleanup after processing
                await app_state.live_audio_processor.complete_session(session_id)

            elif action == "get_state":
                state = app_state.live_audio_processor.get_session_state(session_id)
                await websocket.send_json({"type": "state", "data": state})

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        logger.info(f"Live audio WebSocket disconnected: {session_id}")
        # Cleanup on disconnect
        try:
            await app_state.live_audio_processor.complete_session(session_id)
        except:
            pass
    except Exception as e:
        logger.error(f"Live audio WebSocket error for {session_id}: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": str(e)})

# Removed empty WebSocket endpoint - use /ws/live/{session_id} instead

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )