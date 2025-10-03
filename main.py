"""
TranscrevAI Main Application - Refactored for Modular Multiprocessing
Optimized for memory-efficient transcription with full live recording capabilities
"""

import asyncio
import logging
import os
import time
import uuid
import datetime
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import numpy as np
import multiprocessing as mp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import optimized modules
from src.diarization import force_transcription_segmentation
from src.subtitle_generator import generate_srt
from src.transcription import optimized_transcriber
from src.audio_processing import AudioRecorder # Use the real recorder

# Import the new modular multiprocessing components
from src.performance_optimizer import (
    ProcessType,
    SharedMemoryManager,
    diarization_worker,
    transcription_worker,
    _get_available_cores
)
from websocket_enhancements import create_websocket_safety_manager

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteAppState:
    """Complete application state management with refactored multiprocessing support"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.websocket_safety = create_websocket_safety_manager()

        # Refactored multiprocessing components
        self.multiprocessing_enabled: bool = True
        self.shared_memory_manager: Optional[SharedMemoryManager] = None
        self.transcription_queue: Optional[mp.Queue] = None
        self.diarization_queue: Optional[mp.Queue] = None
        self.processes: Dict[ProcessType, mp.Process] = {}

    def create_session(self, session_id: str) -> bool:
        """Create a new session for a client"""
        if session_id in self.sessions:
            return False
        self.sessions[session_id] = {
            "status": "created",
            "language": "pt",
            "created_at": time.time(),
            "recorder": None,
            "recording": False,
            "paused": False,
            "task": None,
            "user_choices": {"language": "pt", "domain": "general"},
            "srt_file": None,
            "processing_complete": False
        }
        logger.info(f"Session created: {session_id}")
        return True

    def create_recorder_for_session(self, session_id: str, format_type: str = "wav") -> bool:
        """Create a real AudioRecorder for the session."""
        try:
            recordings_dir = Path("data/recordings")
            recordings_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            output_file = recordings_dir / f"recording_{timestamp}_{session_id}.{format_type}"

            recorder = AudioRecorder(
                output_file=str(output_file),
                websocket_manager=self,
                session_id=session_id
            )

            if session_id in self.sessions:
                self.sessions[session_id]["recorder"] = recorder
                self.sessions[session_id]["format"] = format_type
                logger.info(f"Real AudioRecorder created for session {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to create real recorder for session {session_id}: {e}")
            return False

    async def connect_websocket(self, websocket: WebSocket, session_id: str):
        """Handle a new WebSocket connection"""
        await websocket.accept()
        self.connections[session_id] = websocket
        self.create_session(session_id)
        logger.info(f"WebSocket connected: {session_id}")
        await self.send_message(session_id, {"type": "connection_ready", "message": "Ready for commands."})

    async def disconnect_websocket(self, session_id: str):
        """Handle a WebSocket disconnection"""
        if session_id in self.connections:
            del self.connections[session_id]
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.get("task") and not session["task"].done():
                session["task"].cancel()
            if session.get("recording") and session.get("recorder"):
                await session["recorder"].stop_recording()
            del self.sessions[session_id]
        logger.info(f"WebSocket disconnected and cleaned up: {session_id}")

    async def send_message(self, session_id: str, message: Dict):
        """Safely send a message to a WebSocket client"""
        if session_id in self.connections:
            await self.websocket_safety.safe_send_message(self, session_id, message)

    async def start_recording(self, session_id: str, format_type: str = "wav"):
        if not self.create_recorder_for_session(session_id, format_type):
            await self.send_message(session_id, {"type": "error", "message": "Falha ao criar gravador."})
            return
        session = self.sessions[session_id]
        recorder = session["recorder"]
        await recorder.start_recording()
        session["recording"] = True
        await self.send_message(session_id, {"type": "recording_started", "message": "Gravação iniciada."})

    async def stop_recording(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session or not session.get("recorder") or not session.get("recording"):
            return
        recorder = session["recorder"]
        await recorder.stop_recording()
        session["recording"] = False
        audio_file = recorder.output_file
        logger.info(f"Recording stopped for {session_id}. File saved to {audio_file}")
        await self.send_message(session_id, {"type": "recording_stopped", "message": "Gravação finalizada. Iniciando processamento..."})
        # Start processing the recorded file
        asyncio.create_task(self.transcribe_with_multiprocessing(session_id, audio_file, session["user_choices"]))

    async def transcribe_with_multiprocessing(self, session_id: str, audio_file: str, user_choices: dict):
        """Orchestrate transcription and diarization using background worker processes."""
        try:
            if not self.multiprocessing_enabled or not self.transcription_queue or not self.diarization_queue or not self.shared_memory_manager:
                raise RuntimeError("Multiprocessing components are not initialized.")

            processing_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
            domain = user_choices.get("domain", "general")

            await self.send_message(session_id, {"type": "processing_progress", "stage": "queuing", "progress": 10, "message": "Enviando para fila de processamento..."})
            
            transcription_task = {"command": "transcribe_audio", "payload": {"session_id": processing_id, "audio_file": audio_file, "language": "pt", "domain": domain}}
            diarization_task = {"command": "diarize_audio", "payload": {"session_id": processing_id, "audio_file": audio_file}}

            self.transcription_queue.put(transcription_task)
            self.diarization_queue.put(diarization_task)
            logger.info(f"Dispatched tasks for processing_id: {processing_id}")

            await self.send_message(session_id, {"type": "processing_progress", "stage": "processing", "progress": 30, "message": "Aguardando resultados dos workers..."})
            
            transcription_result, diarization_result = await self._wait_for_results(processing_id)

            await self.send_message(session_id, {"type": "processing_progress", "stage": "combining", "progress": 75, "message": "Combinando resultados..."})

            if not transcription_result or transcription_result.get("error"):
                raise ValueError(f"Transcription failed: {transcription_result.get('error', 'No result')}")
            if not diarization_result or diarization_result.get("error"):
                logger.warning(f"Diarization failed: {diarization_result.get('error', 'No result')}. Proceeding without speaker labels.")
                diarization_segments = []
            else:
                diarization_segments = diarization_result.get("segments", [])

            transcription_segments = transcription_result.get("segments", [])
            final_segments = force_transcription_segmentation(transcription_segments, diarization_segments)

            await self.send_message(session_id, {"type": "processing_progress", "stage": "srt_generation", "progress": 90, "message": "Gerando legenda SRT..."})
            
            srt_path = await generate_srt(final_segments, None, filename=f"{session_id}.srt")
            if session_id in self.sessions:
                self.sessions[session_id]["srt_file"] = srt_path

            await self.send_message(session_id, {
                "type": "processing_complete",
                "progress": 100,
                "transcription_data": final_segments,
                "srt_file": srt_path,
                "download_url": f"/download/srt/{session_id}" if srt_path else None,
                "message": "Processamento concluído!"
            })

        except Exception as e:
            logger.error(f"Error in transcribe_with_multiprocessing for session {session_id}: {e}")
            await self.send_message(session_id, {"type": "error", "message": f"Ocorreu um erro crítico no processamento: {e}"})

    async def _wait_for_results(self, processing_id: str, timeout: int = 600) -> tuple:
        """Poll the shared dictionary for results from the worker processes."""
        start_time = time.time()
        transcription_result = None
        diarization_result = None
        shared_dict = self.shared_memory_manager.get_shared_dict()

        while time.time() - start_time < timeout:
            if not transcription_result and f"transcription_{processing_id}" in shared_dict:
                transcription_result = shared_dict.pop(f"transcription_{processing_id}")
                logger.info(f"Retrieved transcription result for {processing_id}")
            
            if not diarization_result and f"diarization_{processing_id}" in shared_dict:
                diarization_result = shared_dict.pop(f"diarization_{processing_id}")
                logger.info(f"Retrieved diarization result for {processing_id}")

            if transcription_result and diarization_result:
                return transcription_result, diarization_result
            
            await asyncio.sleep(0.5)
        
        raise asyncio.TimeoutError(f"Timeout waiting for results for processing_id: {processing_id}")

# Global state
app_state = CompleteAppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application lifecycle, including starting and stopping background processes."""
    logger.info("🚀 Servidor iniciando - TranscrevAI com Multiprocessing Refatorado")
    
    if app_state.multiprocessing_enabled:
        try:
            app_state.shared_memory_manager = SharedMemoryManager()
            app_state.transcription_queue = mp.Queue()
            app_state.diarization_queue = mp.Queue()
            
            worker_config = {"cpu_cores": _get_available_cores()}
            parent_pid = os.getpid()

            transcription_proc = mp.Process(target=transcription_worker, args=(parent_pid, app_state.transcription_queue, app_state.shared_memory_manager.get_shared_dict(), worker_config, True), daemon=True)
            app_state.processes[ProcessType.TRANSCRIPTION] = transcription_proc
            transcription_proc.start()
            logger.info(f"Started transcription worker process (PID: {transcription_proc.pid})")

            diarization_proc = mp.Process(target=diarization_worker, args=(parent_pid, app_state.diarization_queue, app_state.shared_memory_manager.get_shared_dict(), worker_config, True), daemon=True)
            app_state.processes[ProcessType.DIARIZATION] = diarization_proc
            diarization_proc.start()
            logger.info(f"Started diarization worker process (PID: {diarization_proc.pid})")

        except Exception as e:
            logger.critical(f"❌ Falha fatal ao iniciar workers de multiprocessing: {e}")
            app_state.multiprocessing_enabled = False
    
    yield
    
    logger.info("🔄 Desligando servidor...")
    if app_state.multiprocessing_enabled:
        for process_type, process in app_state.processes.items():
            if process.is_alive():
                logger.info(f"Terminating {process_type.value} worker (PID: {process.pid})...")
                process.terminate()
                process.join(timeout=5)
        if app_state.shared_memory_manager:
            app_state.shared_memory_manager.cleanup()

# FastAPI App Definition
app = FastAPI(title="TranscrevAI - Refactored", version="8.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- HTTP Routes --- #
@app.get("/")
async def main_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio_file(file: UploadFile = File(...), session_id: str = Form(...), domain: str = Form("general")):
    try:
        app_state.create_session(session_id)
        app_state.sessions[session_id]["user_choices"]["domain"] = domain
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(exist_ok=True)
        audio_path = upload_dir / f"{session_id}_{file.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"File uploaded: {audio_path}")
        asyncio.create_task(app_state.transcribe_with_multiprocessing(session_id, str(audio_path), {"domain": domain}))
        return JSONResponse(content={"success": True, "session_id": session_id, "message": "Processamento iniciado."})
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download/srt/{session_id}")
async def download_srt(session_id: str):
    session = app_state.sessions.get(session_id)
    if not session or not session.get("srt_file") or not os.path.exists(session["srt_file"]):
        return JSONResponse(status_code=404, content={"error": "Arquivo SRT não encontrado."})
    return FileResponse(path=session["srt_file"], filename=f"transcription_{session_id}.srt", media_type="text/plain")

# --- WebSocket Route --- #
@app.websocket("/ws/{session_id}")
async def websocket_handler(websocket: WebSocket, session_id: str):
    await app_state.connect_websocket(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            if message_type == "start_recording":
                await app_state.start_recording(session_id, data.get("format", "wav"))
            elif message_type == "stop_recording":
                await app_state.stop_recording(session_id)
            elif message_type == "pause_recording":
                session = app_state.sessions.get(session_id)
                if session and session.get("recorder"):
                    session["recorder"].pause_recording()
            elif message_type == "resume_recording":
                session = app_state.sessions.get(session_id)
                if session and session.get("recorder"):
                    session["recorder"].resume_recording()
    except WebSocketDisconnect:
        await app_state.disconnect_websocket(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        await app_state.disconnect_websocket(session_id)

# --- Main Entry Point --- #
if __name__ == "__main__":
    port = int(os.getenv("TRANSCREVAI_PORT", "8000"))
    host = os.getenv("TRANSCREVAI_HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")