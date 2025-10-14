# Enhanced main.py - Final Version with Multiprocessing Pool

"""
TranscrevAI Main Application - Final Implementation
Uses a multiprocessing.Pool for efficient, concurrent pipeline execution.
"""

import asyncio
import logging
import os
import time
import multiprocessing
import queue
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, TYPE_CHECKING
from multiprocessing.managers import SyncManager, DictProxy


import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.worker import process_audio_task, init_worker
from src.file_manager import FileManager
from src.websocket_enhancements import get_websocket_safety_manager, MessagePriority
from config.app_config import get_config

# --- Global Configuration ---
app_config = get_config()

# Set OMP_NUM_THREADS for CTranslate2 performance optimization
os.environ["OMP_NUM_THREADS"] = "2"

logging.basicConfig(level=app_config.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() and not app_config.force_cpu else "cpu"

# Define worker config in a broader scope so it can be used in the lifespan manager
worker_config = {"model_name": "medium", "device": DEVICE}

class AppState:
    """Thread-safe application state for managing server resources."""
    def __init__(self):
        self._lock = threading.RLock()
        self.sessions: Dict[str, Dict] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.queues: 'DictProxy'  # This will be a manager.dict()
        self.process_pool: Any = None
        self.manager: Optional['SyncManager'] = None
        self.websocket_safety_manager = get_websocket_safety_manager()
        self.file_manager = FileManager()

    async def send_message(self, session_id: str, message: Dict):
        # The websocket_manager is now passed correctly
        await self.websocket_safety_manager.safe_send_message(self, session_id, message, MessagePriority.HIGH)

app_state = AppState()

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TranscrevAI Web Server Starting...")
    # Create and manage the global process pool and manager
    cpu_cores = os.cpu_count() or 4 # Fallback to 4 cores if undetectable
    num_workers = max(1, cpu_cores - 2)
    logger.info(f"Initializing process pool with {num_workers} workers.")
    
    # Correctly initialize manager and shared state
    app_state.manager = multiprocessing.Manager()
    app_state.queues = app_state.manager.dict()
    
    # Create the pool with the worker initializer
    app_state.process_pool = multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(worker_config,)
    )
    
    yield
    
    logger.info("Shutting down TranscrevAI...")
    if app_state.process_pool:
        app_state.process_pool.close()
        app_state.process_pool.join()
    logger.info("Process pool shut down.")

# --- FastAPI App Setup ---
app = FastAPI(title="TranscrevAI Enhanced", version="5.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Queue Listener Thread ---
def queue_listener(session_id: str, q: 'queue.Queue'):
    logger.info(f"[Listener-{session_id}] Started for session.")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        try:
            message = q.get(timeout=3600)
            if message is None: break
            if message.get('type') == 'complete':
                srt_path = message.get('result', {}).get('srt_path')
                if srt_path:
                    with app_state._lock:
                        if session_id not in app_state.sessions: app_state.sessions[session_id] = {}
                        app_state.sessions[session_id]['srt_file_path'] = srt_path
            loop.run_until_complete(app_state.send_message(session_id, message))
        except (queue.Empty, BrokenPipeError):
            logger.warning(f"[Listener-{session_id}] Queue empty or broken, shutting down.")
            break
        except Exception as e:
            logger.error(f"[Listener-{session_id}] Error: {e}", exc_info=True)
            break
    logger.info(f"[Listener-{session_id}] Stopped.")

# === API Endpoints ===
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    try:
        if not session_id: session_id = f"upload_{int(time.time() * 1000)}"
        file_path = app_state.file_manager.save_uploaded_file(file.file, file.filename or f"{session_id}.wav")
        
        if not app_state.manager:
            raise RuntimeError("Process manager not initialized.")
        
        q = app_state.manager.Queue()
        app_state.queues[session_id] = q

        worker_config = {"model_name": "medium", "device": DEVICE}

        if not app_state.process_pool:
            raise RuntimeError("Process pool not initialized.")

        app_state.process_pool.apply_async(
            process_audio_task,
            args=(str(file_path), session_id, worker_config, q)
        )
        logger.info(f"Submitted job to process pool for session {session_id}")

        listener_thread = threading.Thread(target=queue_listener, args=(session_id, q), daemon=True)
        listener_thread.start()

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
    """
    Health check endpoint for monitoring systems.
    """
    return {"status": "ok"}

@app.websocket("/ws/{session_id}")
async def file_upload_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    with app_state._lock: app_state.connections[session_id] = websocket
    await app_state.websocket_safety_manager.handle_connection_established(session_id)
    try:
        while True: await asyncio.sleep(3600)
    except WebSocketDisconnect:
        logger.info(f"File upload WebSocket disconnected: {session_id}")
    finally:
        await app_state.websocket_safety_manager.handle_connection_lost(session_id)
        with app_state._lock:
            if session_id in app_state.connections: del app_state.connections[session_id]
            q = app_state.queues.pop(session_id, None)
            if q: 
                try:
                    q.put(None)
                except Exception:
                    pass # Ignore errors on shutdown

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.freeze_support()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
