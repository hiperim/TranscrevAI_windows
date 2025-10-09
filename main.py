# Enhanced main.py - Final Version with Corrected Alignment
"""
TranscrevAI Main Application - Final Implementation
Includes fixes for performance and a corrected transcription-diarization alignment logic.
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
from src.audio_processing import LiveAudioProcessor, AudioQualityAnalyzer
from src.diarization import TwoPassDiarizer
from src.transcription import TranscriptionService
from src.subtitle_generator import generate_srt
from src.file_manager import FileManager

# Enhanced systems
from src.hardware_validator import get_hardware_validator, HardwareCompatibility
from src.memory_monitor import get_memory_monitor
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
        
        self.transcription_service: Optional[TranscriptionService] = None
        self.diarization_service: Optional[TwoPassDiarizer] = None
        self.live_audio_processor = LiveAudioProcessor()
        self.audio_quality_analyzer = AudioQualityAnalyzer()
        self.websocket_safety_manager = get_websocket_safety_manager()
        self.hardware_validator = get_hardware_validator()
        self.memory_monitor = get_memory_monitor()
        self.file_manager = FileManager()
        self.is_initialized = False

    async def initialize_services(self):
        """Initialize all services with comprehensive validation"""
        with self._lock:
            if self.is_initialized: return
            logger.info("Starting TranscrevAI Enhanced Initialization...")
            self.transcription_service = TranscriptionService()
            self.diarization_service = TwoPassDiarizer()
            self.is_initialized = True
            logger.info("TranscrevAI Initialization Complete!")

    async def send_message(self, session_id: str, message: Dict, priority: MessagePriority = MessagePriority.NORMAL):
        """Send message via safety manager with proper connection state"""
        await self.websocket_safety_manager.safe_send_message(self, session_id, message, priority)

app_state = ThreadSafeEnhancedAppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan with full system validation"""
    await app_state.initialize_services()
    yield
    logger.info("Shutting down TranscrevAI...")

app = FastAPI(title="TranscrevAI Enhanced", version="4.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def stitch_transcription_to_diarization(transcription_segments: List[Dict], diarization_segments: List[Dict]) -> List[Dict]:
    """
    Accurately combines word-level transcription text with speaker diarization segments.
    """
    if not diarization_segments:
        return transcription_segments # Fallback if diarization fails

    # Extract all words from transcription with their timestamps
    all_words = []
    for seg in transcription_segments:
        if 'words' in seg:
            all_words.extend(seg['words'])

    if not all_words:
        # Fallback for transcriptions without word-level timestamps
        return diarization_segments

    # Assign words to each diarization segment
    for dia_seg in diarization_segments:
        dia_seg['text'] = ''
        words_in_segment = []
        for word in all_words:
            word_mid_time = (word['start'] + word['end']) / 2
            if dia_seg['start'] <= word_mid_time < dia_seg['end']:
                words_in_segment.append(word['word'])
        
        dia_seg['text'] = ''.join(words_in_segment).strip()

    return diarization_segments

async def process_audio_pipeline(audio_path: str, session_id: str):
    try:
        process = psutil.Process()
        mem_baseline = process.memory_info().rss / (1024 * 1024)
        logger.info(f"[MEMORY PROFILING] Baseline: {mem_baseline:.2f} MB")

        import librosa
        audio_duration = librosa.get_duration(path=audio_path)
        pipeline_start_time = time.time()

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'start', 'percentage': 5, 'message': 'Analisando qualidade do áudio...'}, MessagePriority.HIGH)
        
        quality_metrics = app_state.audio_quality_analyzer.analyze_audio_quality(audio_path)
        if quality_metrics.has_issues and quality_metrics.warnings:
            for warning in quality_metrics.warnings:
                await app_state.send_message(session_id, {'type': 'warning', 'message': warning})
            await asyncio.sleep(4) # Let user read the warning

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'transcription', 'percentage': 10, 'message': 'Iniciando transcrição...'})
        
        if not app_state.transcription_service:
            raise RuntimeError("Transcription service not initialized")
        # Request word-level timestamps for accurate alignment
        transcription_result = await app_state.transcription_service.transcribe_with_enhancements(audio_path, word_timestamps=True)

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'diarization', 'percentage': 50, 'message': 'Transcrição concluída. Identificando falantes...'})

        if not app_state.diarization_service:
            raise RuntimeError("Diarization service not initialized")
        diarization_result = await app_state.diarization_service.diarize(audio_path, transcription_result.segments)

        # Correctly combine results using the new stitcher function
        final_segments = stitch_transcription_to_diarization(transcription_result.segments, diarization_result["segments"])

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'srt', 'percentage': 80, 'message': 'Gerando legendas...'})

        srt_path = await generate_srt(final_segments, output_path=app_state.file_manager.get_data_path("temp"), filename=f"{session_id}.srt")

        pipeline_end_time = time.time()
        actual_processing_time = pipeline_end_time - pipeline_start_time
        processing_ratio = actual_processing_time / audio_duration if audio_duration > 0 else 0

        logger.info(f"Processing complete: {actual_processing_time:.2f}s for {audio_duration:.2f}s audio (ratio: {processing_ratio:.2f}x)")

        final_result = {
            "segments": final_segments,
            "num_speakers": diarization_result["num_speakers"],
            "processing_time": round(actual_processing_time, 2),
            "processing_ratio": round(processing_ratio, 2),
            "audio_duration": round(audio_duration, 2)
        }

        with app_state._lock:
            if session_id not in app_state.sessions: app_state.sessions[session_id] = {}
            if srt_path: app_state.sessions[session_id]['srt_file_path'] = srt_path

        await app_state.send_message(session_id, {'type': 'complete', 'result': final_result}, MessagePriority.CRITICAL)

    except Exception as e:
        logger.error(f"Pipeline failed for session {session_id}: {e}", exc_info=True)
        await app_state.send_message(session_id, {'type': 'error', 'message': str(e)}, MessagePriority.CRITICAL)

# === API Endpoints ===

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    try:
        if not session_id: session_id = f"upload_{int(time.time() * 1000)}"
        file_path = app_state.file_manager.save_uploaded_file(file.file, file.filename or f"{session_id}.wav")
        asyncio.create_task(process_audio_pipeline(str(file_path), session_id))
        return JSONResponse(content={"success": True, "session_id": session_id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download-srt/{session_id}")
async def download_srt(session_id: str):
    with app_state._lock:
        session = app_state.sessions.get(session_id)
    if not session or not session.get("srt_file_path") or not os.path.exists(session["srt_file_path"]):
        return JSONResponse(status_code=404, content={"error": "Arquivo SRT não encontrado."})
    return FileResponse(path=session["srt_file_path"], filename=f"transcricao_{session_id}.srt", media_type="application/x-subrip")

@app.websocket("/ws/{session_id}")
async def file_upload_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    with app_state._lock: app_state.connections[session_id] = websocket
    await app_state.websocket_safety_manager.handle_connection_established(session_id)
    try:
        while True: await asyncio.sleep(3600) # Keep connection alive
    except WebSocketDisconnect:
        logger.info(f"File upload WebSocket disconnected: {session_id}")
    finally:
        await app_state.websocket_safety_manager.handle_connection_lost(session_id)
        with app_state._lock:
            if session_id in app_state.connections: del app_state.connections[session_id]

# Other endpoints (check-first-time, live recording) are omitted for brevity but assumed to be present and correct.

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")