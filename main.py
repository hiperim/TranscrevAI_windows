# TranscrevAI Main Application - Production Ready
import asyncio
import logging
import os
import time
import uuid
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.audio_processing import AudioRecorder
from src.transcription import transcribe_audio_with_progress
from src.speaker_diarization import SpeakerDiarization
from src.file_manager import FileManager
from src.subtitle_generator import generate_srt
from config.app_config import MODEL_DIR
from src.logging_setup import setup_app_logging

# Setup logging
logger = setup_app_logging()

app = FastAPI(title="TranscrevAI", description="Audio Transcription with Diarization")

# Session-based state management
class AppState:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.lock = asyncio.Lock()

    async def create_session(self, session_id: str):
        async with self.lock:
            recorder = AudioRecorder()
            self.sessions[session_id] = {
                "recording": False,
                "paused": False,
                "progress": {"transcription": 0, "diarization": 0},
                "current_task": None,
                "recorder": recorder,
                "websocket": None,
                "audio_levels": [],  # For waveform visualization
                "start_time": None,
                "duration": 0
            }

    async def get_session(self, session_id: str) -> Optional[dict]:
        async with self.lock:
            return self.sessions.get(session_id)

    async def update_session(self, session_id: str, updates: dict):
        async with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(updates)

    async def cleanup_session(self, session_id: str):
        async with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                
                # Cleanup recorder resources
                if session.get("recorder"):
                    try:
                        await session["recorder"].stop_recording()
                        await session["recorder"].cleanup_resources()
                    except Exception as e:
                        logger.warning(f"Error during recorder cleanup: {e}")
                
                # Cleanup async tasks
                if session.get("current_task"):
                    task = session["current_task"]
                    if not task.done() and not task.cancelled():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        
                del self.sessions[session_id]
                logger.info(f"Session cleaned up: {session_id}")

# WebSocket connection manager with enhanced error handling
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.lock = asyncio.Lock()
        self.reconnect_attempts: Dict[str, int] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        async with self.lock:
            self.active_connections[session_id] = websocket
            self.reconnect_attempts[session_id] = 0
        logger.info(f"WebSocket connected: {session_id}")

    async def disconnect(self, session_id: str):
        async with self.lock:
            if session_id in self.active_connections:
                del self.active_connections[session_id]
            if session_id in self.reconnect_attempts:
                del self.reconnect_attempts[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")

    async def send_personal_message(self, message: dict, session_id: str):
        async with self.lock:
            if session_id in self.active_connections:
                websocket = self.active_connections[session_id]
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to {session_id}: {e}")
                    await self.remove_connection(websocket)

    async def remove_connection(self, websocket: WebSocket):
        async with self.lock:
            for session_id, ws in list(self.active_connections.items()):
                if ws == websocket:
                    del self.active_connections[session_id]
                    await app_state.cleanup_session(session_id)
                    break

# Global instances
app_state = AppState()
manager = ConnectionManager()

# Dependency injection
def get_app_state():
    return app_state

def get_connection_manager():
    return manager

# WebSocket endpoint with enhanced features
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    session_id: str,
    state: AppState = Depends(get_app_state),
    manager: ConnectionManager = Depends(get_connection_manager)
):
    await manager.connect(websocket, session_id)
    await state.create_session(session_id)
    await state.update_session(session_id, {"websocket": websocket})
    
    # Start sending audio levels for visualization
    audio_level_task = asyncio.create_task(send_audio_levels(session_id))
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            message_data = data.get("data", {})
            
            session = await state.get_session(session_id)
            if not session:
                await manager.send_personal_message({
                    "type": "error", 
                    "message": "Session not found"
                }, session_id)
                continue
                
            recorder = session.get("recorder")
            if not recorder:
                await manager.send_personal_message({
                    "type": "error", 
                    "message": "Recorder not initialized"
                }, session_id)
                continue
            
            # Handle different message types
            if message_type == "start_recording":
                if not session["recording"]:
                    try:
                        language = message_data.get("language", "en")
                        await recorder.start_recording()
                        await state.update_session(session_id, {
                            "recording": True,
                            "start_time": time.time()
                        })
                        await manager.send_personal_message({
                            "type": "recording_started",
                            "message": "Recording started successfully"
                        }, session_id)
                        
                        # Start processing pipeline
                        task = asyncio.create_task(
                            process_audio_pipeline(session_id, language)
                        )
                        await state.update_session(session_id, {"current_task": task})
                        
                    except Exception as e:
                        logger.error(f"Error starting recording: {e}")
                        await manager.send_personal_message({
                            "type": "error", 
                            "message": f"Failed to start recording: {str(e)}"
                        }, session_id)
            
            elif message_type == "stop_recording":
                if session["recording"]:
                    try:
                        success = await recorder.stop_recording()
                        duration = time.time() - session["start_time"] if session["start_time"] else 0
                        await state.update_session(session_id, {
                            "recording": False,
                            "duration": duration
                        })
                        if success:
                            await manager.send_personal_message({
                                "type": "recording_stopped",
                                "message": "Recording stopped successfully",
                                "duration": duration
                            }, session_id)
                    except Exception as e:
                        logger.error(f"Error stopping recording: {e}")
                        await manager.send_personal_message({
                            "type": "error", 
                            "message": f"Failed to stop recording: {str(e)}"
                        }, session_id)
            
            elif message_type == "pause_recording":
                if session["recording"] and not session["paused"]:
                    try:
                        recorder.pause_recording()
                        await state.update_session(session_id, {"paused": True})
                        await manager.send_personal_message({
                            "type": "recording_paused",
                            "message": "Recording paused"
                        }, session_id)
                    except Exception as e:
                        logger.error(f"Error pausing recording: {e}")
                        await manager.send_personal_message({
                            "type": "error", 
                            "message": f"Failed to pause recording: {str(e)}"
                        }, session_id)
            
            elif message_type == "resume_recording":
                if session["recording"] and session["paused"]:
                    try:
                        recorder.resume_recording()
                        await state.update_session(session_id, {"paused": False})
                        await manager.send_personal_message({
                            "type": "recording_resumed",
                            "message": "Recording resumed"
                        }, session_id)
                    except Exception as e:
                        logger.error(f"Error resuming recording: {e}")
                        await manager.send_personal_message({
                            "type": "error", 
                            "message": f"Failed to resume recording: {str(e)}"
                        }, session_id)
            
            elif message_type == "ping":
                # Heartbeat to keep connection alive
                await manager.send_personal_message({"type": "pong"}, session_id)
                
    except WebSocketDisconnect:
        audio_level_task.cancel()
        await manager.disconnect(session_id)
        await state.cleanup_session(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        audio_level_task.cancel()
        await manager.disconnect(session_id)
        await state.cleanup_session(session_id)

# Send audio levels for waveform visualization
async def send_audio_levels(session_id: str):
    try:
        while True:
            session = await app_state.get_session(session_id)
            if session and session["recording"] and not session["paused"]:
                # Simulate audio levels (replace with actual audio level detection)
                import random
                levels = [random.uniform(0.1, 1.0) for _ in range(20)]
                await manager.send_personal_message({
                    "type": "audio_levels",
                    "levels": levels
                }, session_id)
            await asyncio.sleep(0.1)  # Update 10 times per second
    except asyncio.CancelledError:
        pass

# Audio processing pipeline with enhanced progress tracking
async def process_audio_pipeline(session_id: str, audio_file_or_language: str):
    try:
        session = await app_state.get_session(session_id)
        if not session:
            return
            
        # If it's a language code, we're starting the recording pipeline
        if len(audio_file_or_language) == 2:
            language = audio_file_or_language
            
            # Wait for recording to complete
            while session and session["recording"]:
                if session["paused"]:
                    await asyncio.sleep(0.1)
                    session = await app_state.get_session(session_id)
                    continue
                    
                await asyncio.sleep(0.1)
                session = await app_state.get_session(session_id)
            
            if session and session.get("recorder"):
                audio_file = session["recorder"].output_file
            else:
                return
        else:
            audio_file = audio_file_or_language
            language = "en"

        # Transcription with smooth progress updates
        transcription_data = []
        async for progress, data in transcribe_audio_with_progress(
            audio_file, MODEL_DIR, language
        ):
            await app_state.update_session(session_id, {
                "progress": {"transcription": progress, "diarization": 0}
            })
            await manager.send_personal_message({
                "type": "progress", 
                "transcription": progress, 
                "diarization": 0,
                "smooth": True  # Enable smooth animations on frontend
            }, session_id)
            transcription_data = data

        # Speaker diarization with progress
        diarizer = SpeakerDiarization()
        diarization_segments = await diarizer.diarize_audio(audio_file)
        
        await app_state.update_session(session_id, {
            "progress": {"transcription": 100, "diarization": 100}
        })
        await manager.send_personal_message({
            "type": "progress", 
            "transcription": 100, 
            "diarization": 100,
            "smooth": True
        }, session_id)

        # Generate subtitles
        srt_file = await generate_srt(transcription_data, diarization_segments)

        # Send completion message with enhanced data
        await manager.send_personal_message({
            "type": "processing_complete",
            "srt_file": srt_file,
            "transcription_data": transcription_data,
            "diarization_segments": diarization_segments,
            "duration": session.get("duration", 0)
        }, session_id)

    except Exception as e:
        logger.error(f"Processing pipeline error for session {session_id}: {e}")
        await manager.send_personal_message({
            "type": "error", 
            "message": f"Processing failed: {str(e)}"
        }, session_id)

# TranscrevAI Main Application - Production Ready
import asyncio
import logging
import os
import time
import uuid
import json
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.audio_processing import AudioRecorder
from src.transcription import transcribe_audio_with_progress
from src.speaker_diarization import SpeakerDiarization
from src.file_manager import FileManager
from src.subtitle_generator import generate_srt
from config.app_config import MODEL_DIR
from src.logging_setup import setup_app_logging

# Setup logging
logger = setup_app_logging()

app = FastAPI(
    title="TranscrevAI", 
    description="Real-time Audio Transcription with Speaker Diarization",
    version="1.0.0"
)

# Session-based state management with enhanced features
class AppState:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.lock = asyncio.Lock()

    async def create_session(self, session_id: str):
        async with self.lock:
            recorder = AudioRecorder()
            self.sessions[session_id] = {
                "recording": False,
                "paused": False,
                "progress": {"transcription": 0, "diarization": 0},
                "current_task": None,
                "recorder": recorder,
                "websocket": None,
                "start_time": None,
                "duration": 0,
                "audio_levels": [],
                "speakers_detected": 0,
                "created_at": datetime.now()
            }

    async def get_session(self, session_id: str) -> Optional[dict]:
        async with self.lock:
            return self.sessions.get(session_id)

    async def update_session(self, session_id: str, updates: dict):
        async with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(updates)

    async def cleanup_session(self, session_id: str):
        async with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                
                if session.get("recorder"):
                    try:
                        await session["recorder"].stop_recording()
                        await session["recorder"].cleanup_resources()
                    except Exception as e:
                        logger.warning(f"Error during recorder cleanup: {e}")
                
                if session.get("current_task"):
                    task = session["current_task"]
                    if not task.done() and not task.cancelled():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        
                del self.sessions[session_id]
                logger.info(f"Session cleaned up: {session_id}")

# Enhanced WebSocket connection manager with reconnection support
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.lock = asyncio.Lock()
        self.reconnect_attempts: Dict[str, int] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        async with self.lock:
            self.active_connections[session_id] = websocket
            self.reconnect_attempts[session_id] = 0
        logger.info(f"WebSocket connected: {session_id}")

    async def disconnect(self, session_id: str):
        async with self.lock:
            if session_id in self.active_connections:
                del self.active_connections[session_id]
            if session_id in self.reconnect_attempts:
                del self.reconnect_attempts[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")

    async def send_personal_message(self, message: dict, session_id: str):
        async with self.lock:
            if session_id in self.active_connections:
                websocket = self.active_connections[session_id]
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to {session_id}: {e}")
                    await self.handle_failed_connection(session_id)

    async def handle_failed_connection(self, session_id: str):
        if session_id in self.reconnect_attempts:
            self.reconnect_attempts[session_id] += 1
            if self.reconnect_attempts[session_id] > 3:
                await self.remove_connection(session_id)

    async def remove_connection(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            await app_state.cleanup_session(session_id)

# Global instances
app_state = AppState()
manager = ConnectionManager()

# WebSocket endpoint with enhanced features
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    await app_state.create_session(session_id)
    await app_state.update_session(session_id, {"websocket": websocket})
    
    # Send initial configuration
    await manager.send_personal_message({
        "type": "config",
        "data": {
            "session_id": session_id,
            "features": ["waveform", "diarization", "realtime_progress"],
            "max_duration": 3600  # 1 hour max recording
        }
    }, session_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            message_data = data.get("data", {})
            
            session = await app_state.get_session(session_id)
            if not session:
                await manager.send_personal_message({
                    "type": "error", 
                    "message": "Session not found"
                }, session_id)
                continue
                
            recorder = session.get("recorder")
            if not recorder:
                await manager.send_personal_message({
                    "type": "error", 
                    "message": "Recorder not initialized"
                }, session_id)
                continue
            
            # Handle different message types
            if message_type == "start_recording":
                if not session["recording"]:
                    try:
                        language = message_data.get("language", "en")
                        await recorder.start_recording()
                        start_time = time.time()
                        await app_state.update_session(session_id, {
                            "recording": True,
                            "start_time": start_time
                        })
                        await manager.send_personal_message({
                            "type": "recording_started",
                            "message": "Recording started successfully",
                            "timestamp": start_time
                        }, session_id)
                        
                        # Start audio level monitoring
                        asyncio.create_task(monitor_audio_levels(session_id))
                        
                        # Start processing pipeline
                        task = asyncio.create_task(
                            process_audio_pipeline(session_id, language)
                        )
                        await app_state.update_session(session_id, {"current_task": task})
                        
                    except Exception as e:
                        logger.error(f"Error starting recording: {e}")
                        await manager.send_personal_message({
                            "type": "error", 
                            "message": f"Failed to start recording: {str(e)}"
                        }, session_id)
            
            elif message_type == "stop_recording":
                if session["recording"]:
                    try:
                        success = await recorder.stop_recording()
                        duration = time.time() - session["start_time"] if session["start_time"] else 0
                        await app_state.update_session(session_id, {
                            "recording": False,
                            "duration": duration
                        })
                        if success:
                            await manager.send_personal_message({
                                "type": "recording_stopped",
                                "message": "Recording stopped successfully",
                                "duration": duration
                            }, session_id)
                    except Exception as e:
                        logger.error(f"Error stopping recording: {e}")
                        await manager.send_personal_message({
                            "type": "error", 
                            "message": f"Failed to stop recording: {str(e)}"
                        }, session_id)
            
            elif message_type == "pause_recording":
                if session["recording"] and not session["paused"]:
                    try:
                        recorder.pause_recording()
                        await app_state.update_session(session_id, {"paused": True})
                        await manager.send_personal_message({
                            "type": "recording_paused",
                            "message": "Recording paused"
                        }, session_id)
                    except Exception as e:
                        logger.error(f"Error pausing recording: {e}")
                        await manager.send_personal_message({
                            "type": "error", 
                            "message": f"Failed to pause recording: {str(e)}"
                        }, session_id)
            
            elif message_type == "resume_recording":
                if session["recording"] and session["paused"]:
                    try:
                        recorder.resume_recording()
                        await app_state.update_session(session_id, {"paused": False})
                        await manager.send_personal_message({
                            "type": "recording_resumed",
                            "message": "Recording resumed"
                        }, session_id)
                    except Exception as e:
                        logger.error(f"Error resuming recording: {e}")
                        await manager.send_personal_message({
                            "type": "error", 
                            "message": f"Failed to resume recording: {str(e)}"
                        }, session_id)
            
            elif message_type == "ping":
                # Heartbeat mechanism
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": time.time()
                }, session_id)
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await manager.send_personal_message({
                    "type": "error", 
                    "message": f"Unknown action: {message_type}"
                }, session_id)
                
    except WebSocketDisconnect:
        await manager.disconnect(session_id)
        await app_state.cleanup_session(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await manager.disconnect(session_id)
        await app_state.cleanup_session(session_id)

# Audio level monitoring for waveform visualization
async def monitor_audio_levels(session_id: str):
    """Monitor audio levels for real-time waveform visualization"""
    try:
        session = await app_state.get_session(session_id)
        while session and session["recording"]:
            if not session["paused"]:
                # Simulate audio level data (replace with actual audio level reading)
                import random
                audio_level = random.uniform(0.1, 1.0) if random.random() > 0.3 else 0.0
                
                await manager.send_personal_message({
                    "type": "audio_level",
                    "level": audio_level,
                    "timestamp": time.time()
                }, session_id)
            
            await asyncio.sleep(0.1)  # Update every 100ms
            session = await app_state.get_session(session_id)
    except Exception as e:
        logger.error(f"Error monitoring audio levels: {e}")

# Enhanced audio processing pipeline
async def process_audio_pipeline(session_id: str, language: str = "en"):
    try:
        session = await app_state.get_session(session_id)
        if not session:
            return
            
        # Wait for recording to complete
        while session and session["recording"]:
            if session["paused"]:
                await asyncio.sleep(0.1)
                session = await app_state.get_session(session_id)
                continue
            await asyncio.sleep(0.1)
            session = await app_state.get_session(session_id)
        
        # Get the recorded audio file
        if session and session.get("recorder"):
            audio_file = session["recorder"].output_file
        else:
            return

        # Transcription with smooth progress updates
        transcription_data = []
        async for progress, data in transcribe_audio_with_progress(
            audio_file, MODEL_DIR, language
        ):
            await app_state.update_session(session_id, {
                "progress": {"transcription": progress, "diarization": session["progress"]["diarization"]}
            })
            await manager.send_personal_message({
                "type": "progress", 
                "transcription": progress, 
                "diarization": session["progress"]["diarization"],
                "smooth": True  # Enable smooth animations on frontend
            }, session_id)
            transcription_data = data

        # Speaker diarization
        diarizer = SpeakerDiarization()
        diarization_segments = await diarizer.diarize_audio(audio_file)
        
        # Count unique speakers
        unique_speakers = len(set(seg.get("speaker", "Unknown") for seg in diarization_segments))
        
        await app_state.update_session(session_id, {
            "progress": {"transcription": 100, "diarization": 100},
            "speakers_detected": unique_speakers
        })
        await manager.send_personal_message({
            "type": "progress", 
            "transcription": 100, 
            "diarization": 100,
            "speakers_detected": unique_speakers
        }, session_id)

        # Generate subtitles
        srt_file = await generate_srt(transcription_data, diarization_segments)

        # Send completion message
        await manager.send_personal_message({
            "type": "processing_complete",
            "srt_file": srt_file,
            "transcription_data": transcription_data,
            "diarization_segments": diarization_segments,
            "speakers_detected": unique_speakers,
            "duration": session.get("duration", 0)
        }, session_id)

    except Exception as e:
        logger.error(f"Processing pipeline error for session {session_id}: {e}")
        await manager.send_personal_message({
            "type": "error", 
            "message": f"Processing failed: {str(e)}"
        }, session_id)

# Enhanced HTML Template with Dark Theme and Professional UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TranscrevAI - Professional Audio Transcription</title>
    <style>
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #252525;
            --text-primary: #ffffff;
            --text-secondary: #b3b3b3;
            --accent-primary: #00d4ff;
            --accent-secondary: #0099cc;
            --success: #00ff88;
            --warning: #ffaa00;
            --danger: #ff3366;
            --border-color: #333333;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }
        
        .header {
            background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo h1 {
            font-size: 24px;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border-radius: 20px;
            font-size: 14px;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.1); }
        }
        
        .main-container {
            flex: 1;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            padding: 40px 20px;
        }
        
        .control-panel {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }
        
        .control-row {
            display: flex;
            gap: 20px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }
        
        .control-group {
            flex: 1;
            min-width: 200px;
        }
        
        .control-label {
            display: block;
            margin-bottom: 8px;
            font-size: 14px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        .select-wrapper {
            position: relative;
        }
        
        .select-wrapper::after {
            content: '▼';
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            pointer-events: none;
            color: var(--text-secondary);
            font-size: 12px;
        }
        
        select {
            width: 100%;
            padding: 12px 40px 12px 15px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 15px;
            appearance: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        select:hover {
            border-color: var(--accent-primary);
        }
        
        select:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
        }
        
        .btn {
            padding: 14px 32px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: relative;
            overflow: hidden;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .btn:hover::before {
            left: 100%;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: var(--bg-primary);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 212, 255, 0.3);
        }
        
        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        
        .btn-secondary:hover {
            border-color: var(--accent-primary);
            background: var(--bg-primary);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, var(--danger), #cc0033);
            color: white;
        }
        
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(255, 51, 102, 0.3);
        }
        
        .btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .btn:disabled:hover::before {
            left: -100%;
        }
        
        /* Waveform Visualizer */
        .waveform-container {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }
        
        .waveform-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .waveform-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .recording-timer {
            font-size: 24px;
            font-family: 'Courier New', monospace;
            color: var(--accent-primary);
            font-weight: 600;
        }
        
        #waveformCanvas {
            width: 100%;
            height: 150px;
            background: var(--bg-primary);
            border-radius: 8px;
        }
        
        .speaker-info {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 20px;
            padding: 15px;
            background: var(--bg-tertiary);
            border-radius: 8px;
        }
        
        .speaker-count {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
        }
        
        .speaker-icon {
            width: 24px;
            height: 24px;
            fill: var(--accent-primary);
        }
        
        /* Progress Section */
        .progress-container {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
            display: none;
        }
        
        .progress-container.active {
            display: block;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .progress-item {
            margin-bottom: 25px;
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        
        .progress-label {
            font-size: 14px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        .progress-value {
            font-size: 14px;
            color: var(--accent-primary);
            font-weight: 600;
        }
        
        .progress-bar-bg {
            width: 100%;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 3px;
            width: 0%;
            transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        
        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            width: 4px;
            background: white;
            box-shadow: 0 0 10px rgba(255,255,255,0.5);
            animation: shimmer 1s infinite;
        }
        
        @keyframes shimmer {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
        
        /* Results Section */
        .results-container {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 30px;
            border: 1px solid var(--border-color);
            display: none;
        }
        
        .results-container.active {
            display: block;
            animation: slideIn 0.5s ease;
        }
        
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .results-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .results-stats {
            display: flex;
            gap: 20px;
        }
        
        .stat-item {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: 600;
            color: var(--accent-primary);
        }
        
        .stat-label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }
        
        .transcript-box {
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
            margin-top: 20px;
        }
        
        .transcript-segment {
            margin-bottom: 15px;
            padding: 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            border-left: 3px solid var(--accent-primary);
        }
        
        .speaker-tag {
            display: inline-block;
            padding: 2px 8px;
            background: var(--accent-primary);
            color: var(--bg-primary);
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .transcript-text {
            color: var(--text-primary);
            line-height: 1.5;
        }
        
        /* Error Messages */
        .error-toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--danger);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(255, 51, 102, 0.3);
            z-index: 1000;
            animation: slideInRight 0.3s ease;
        }
        
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 15px;
            }
            
            .control-row {
                flex-direction: column;
            }
            
            .button-group {
                flex-direction: column;
            }
            
            .btn {
                width: 100%;
            }
            
            .results-stats {
                flex-direction: column;
                gap: 10px;
            }
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-tertiary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--accent-primary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-secondary);
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="logo">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2L2 7V12C2 16.5 4.5 20.5 8 22C11.5 20.5 14 16.5 14 12V7L12 2Z" 
                          fill="url(#gradient1)" />
                    <path d="M12 2V22C15.5 20.5 22 16.5 22 12V7L12 2Z" 
                          fill="url(#gradient2)" opacity="0.8" />
                    <defs>
                        <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" style="stop-color:#00d4ff;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#0099cc;stop-opacity:1" />
                        </linearGradient>
                        <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" style="stop-color:#00d4ff;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#0066cc;stop-opacity:1" />
                        </linearGradient>
                    </defs>
                </svg>
                <h1>TranscrevAI</h1>
            </div>
            <div class="status-badge">
                <span class="status-indicator" id="connectionIndicator"></span>
                <span id="connectionText">Connecting...</span>
            </div>
        </div>
    </div>

    <div class="main-container">
        <div class="control-panel">
            <div class="control-row">
                <div class="control-group">
                    <label class="control-label">Language</label>
                    <div class="select-wrapper">
                        <select id="languageSelect">
                            <option value="en">English / Inglês</option>
                            <option value="pt">Portuguese / Português</option>
                            <option value="es">Spanish / Español</option>
                        </select>
                    </div>
                </div>
                <div class="control-group">
                    <label class="control-label">Quality</label>
                    <div class="select-wrapper">
                        <select id="qualitySelect">
                            <option value="high">High Quality</option>
                            <option value="medium">Medium Quality</option>
                            <option value="low">Low Quality</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="button-group">
                <button id="startBtn" class="btn btn-primary">
                    <span>● Start Recording</span>
                </button>
                <button id="pauseBtn" class="btn btn-secondary" disabled>
                    <span>⏸ Pause</span>
                </button>
                <button id="stopBtn" class="btn btn-danger" disabled>
                    <span>■ Stop</span>
                </button>
            </div>
        </div>

        <div class="waveform-container" id="waveformContainer" style="display: none;">
            <div class="waveform-header">
                <div class="waveform-title">Audio Waveform</div>
                <div class="recording-timer" id="recordingTimer">00:00:00</div>
            </div>
            <canvas id="waveformCanvas"></canvas>
            <div class="speaker-info">
                <div class="speaker-count">
                    <svg class="speaker-icon" viewBox="0 0 24 24">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                    </svg>
                    <span>Speakers Detected: <strong id="speakerCount">0</strong></span>
                </div>
            </div>
        </div>

        <div class="progress-container" id="progressContainer">
            <div class="progress-item">
                <div class="progress-header">
                    <span class="progress-label">Transcription Progress</span>
                    <span class="progress-value" id="transcriptionValue">0%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar" id="transcriptionBar"></div>
                </div>
            </div>
            
            <div class="progress-item">
                <div class="progress-header">
                    <span class="progress-label">Speaker Diarization</span>
                    <span class="progress-value" id="diarizationValue">0%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar" id="diarizationBar"></div>
                </div>
            </div>
        </div>

        <div class="results-container" id="resultsContainer">
            <div class="results-header">
                <div class="results-title">Transcription Results</div>
                <div class="results-stats">
                    <div class="stat-item">
                        <div class="stat-value" id="statDuration">0:00</div>
                        <div class="stat-label">Duration</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statSpeakers">0</div>
                        <div class="stat-label">Speakers</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statWords">0</div>
                        <div class="stat-label">Words</div>
                    </div>
                </div>
            </div>
            <div class="transcript-box" id="transcriptBox"></div>
        </div>
    </div>

    <script>
        class TranscrevAI {
            constructor() {
                this.ws = null;
                this.sessionId = this.generateSessionId();
                this.isRecording = false;
                this.isPaused = false;
                this.isConnected = false;
                this.recordingStartTime = null;
                this.recordingTimer = null;
                this.waveformData = [];
                this.animationFrame = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.heartbeatInterval = null;
                
                this.initializeElements();
                this.setupEventListeners();
                this.initializeWaveform();
                this.connect();
            }

            generateSessionId() {
                return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            }

            initializeElements() {
                // Controls
                this.startBtn = document.getElementById('startBtn');
                this.pauseBtn = document.getElementById('pauseBtn');
                this.stopBtn = document.getElementById('stopBtn');
                this.languageSelect = document.getElementById('languageSelect');
                this.qualitySelect = document.getElementById('qualitySelect');
                
                // Status
                this.connectionIndicator = document.getElementById('connectionIndicator');
                this.connectionText = document.getElementById('connectionText');
                
                // Waveform
                this.waveformContainer = document.getElementById('waveformContainer');
                this.waveformCanvas = document.getElementById('waveformCanvas');
                this.waveformCtx = this.waveformCanvas.getContext('2d');
                this.recordingTimer = document.getElementById('recordingTimer');
                this.speakerCount = document.getElementById('speakerCount');
                
                // Progress
                this.progressContainer = document.getElementById('progressContainer');
                this.transcriptionBar = document.getElementById('transcriptionBar');
                this.transcriptionValue = document.getElementById('transcriptionValue');
                this.diarizationBar = document.getElementById('diarizationBar');
                this.diarizationValue = document.getElementById('diarizationValue');
                
                // Results
                this.resultsContainer = document.getElementById('resultsContainer');
                this.transcriptBox = document.getElementById('transcriptBox');
                this.statDuration = document.getElementById('statDuration');
                this.statSpeakers = document.getElementById('statSpeakers');
                this.statWords = document.getElementById('statWords');
            }

            setupEventListeners() {
                this.startBtn.addEventListener('click', () => this.startRecording());
                this.pauseBtn.addEventListener('click', () => this.togglePause());
                this.stopBtn.addEventListener('click', () => this.stopRecording());
                
                // Handle page visibility for reconnection
                document.addEventListener('visibilitychange', () => {
                    if (!document.hidden && !this.isConnected) {
                        this.connect();
                    }
                });
            }

            initializeWaveform() {
                // Set canvas size
                const resizeCanvas = () => {
                    const rect = this.waveformCanvas.getBoundingClientRect();
                    this.waveformCanvas.width = rect.width;
                    this.waveformCanvas.height = rect.height;
                };
                
                resizeCanvas();
                window.addEventListener('resize', resizeCanvas);
                
                // Initialize waveform data array
                this.waveformData = new Array(100).fill(0);
            }

            connect() {
                if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                    this.showError('Failed to connect. Please refresh the page.');
                    return;
                }
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
                
                try {
                    this.ws = new WebSocket(wsUrl);
                    
                    this.ws.onopen = () => {
                        this.isConnected = true;
                        this.reconnectAttempts = 0;
                        this.updateConnectionStatus(true);
                        this.startHeartbeat();
                        console.log('WebSocket connected');
                    };
                    
                    this.ws.onclose = () => {
                        this.isConnected = false;
                        this.updateConnectionStatus(false);
                        this.stopHeartbeat();
                        console.log('WebSocket disconnected');
                        
                        // Attempt reconnection with exponential backoff
                        if (this.reconnectAttempts < this.maxReconnectAttempts) {
                            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
                            setTimeout(() => {
                                this.reconnectAttempts++;
                                this.connect();
                            }, delay);
                        }
                    };
                    
                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                    };
                    
                    this.ws.onmessage = (event) => {
                        this.handleMessage(JSON.parse(event.data));
                    };
                } catch (error) {
                    console.error('Failed to create WebSocket:', error);
                    this.showError('Connection failed. Please try again.');
                }
            }

            startHeartbeat() {
                this.heartbeatInterval = setInterval(() => {
                    if (this.isConnected) {
                        this.sendMessage('ping', {});
                    }
                }, 30000); // Every 30 seconds
            }

            stopHeartbeat() {
                if (this.heartbeatInterval) {
                    clearInterval(this.heartbeatInterval);
                    this.heartbeatInterval = null;
                }
            }

            sendMessage(type, data = {}) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type, data }));
                } else {
                    console.error('WebSocket not connected');
                    this.showError('Connection lost. Attempting to reconnect...');
                }
            }

            handleMessage(message) {
                switch (message.type) {
                    case 'config':
                        console.log('Received configuration:', message.data);
                        break;
                        
                    case 'recording_started':
                        this.handleRecordingStarted();
                        break;
                        
                    case 'recording_stopped':
                        this.handleRecordingStopped(message);
                        break;
                        
                    case 'recording_paused':
                        this.handleRecordingPaused();
                        break;
                        
                    case 'recording_resumed':
                        this.handleRecordingResumed();
                        break;
                        
                    case 'audio_level':
                        this.updateWaveform(message.level);
                        break;
                        
                    case 'progress':
                        this.updateProgress(message);
                        break;
                        
                    case 'processing_complete':
                        this.handleProcessingComplete(message);
                        break;
                        
                    case 'error':
                        this.showError(message.message);
                        break;
                        
                    case 'pong':
                        // Heartbeat response
                        break;
                        
                    default:
                        console.warn('Unknown message type:', message.type);
                }
            }

            startRecording() {
                const language = this.languageSelect.value;
                this.sendMessage('start_recording', { language });
            }

            stopRecording() {
                this.sendMessage('stop_recording');
            }

            togglePause() {
                if (this.isPaused) {
                    this.sendMessage('resume_recording');
                } else {
                    this.sendMessage('pause_recording');
                }
            }

            handleRecordingStarted() {
                this.isRecording = true;
                this.isPaused = false;
                this.recordingStartTime = Date.now();
                
                // Update UI
                this.startBtn.disabled = true;
                this.pauseBtn.disabled = false;
                this.stopBtn.disabled = false;
                this.languageSelect.disabled = true;
                this.qualitySelect.disabled = true;
                
                // Show waveform
                this.waveformContainer.style.display = 'block';
                this.startRecordingTimer();
                this.startWaveformAnimation();
                
                // Hide previous results
                this.resultsContainer.classList.remove('active');
            }

            handleRecordingStopped(message) {
                this.isRecording = false;
                this.isPaused = false;
                
                // Update UI
                this.startBtn.disabled = false;
                this.pauseBtn.disabled = true;
                this.stopBtn.disabled = true;
                this.languageSelect.disabled = false;
                this.qualitySelect.disabled = false;
                
                // Stop animations
                this.stopRecordingTimer();
                this.stopWaveformAnimation();
                
                // Show progress
                this.progressContainer.classList.add('active');
            }

            handleRecordingPaused() {
                this.isPaused = true;
                this.pauseBtn.innerHTML = '<span>▶ Resume</span>';
            }

            handleRecordingResumed() {
                this.isPaused = false;
                this.pauseBtn.innerHTML = '<span>⏸ Pause</span>';
            }

            startRecordingTimer() {
                this.recordingTimer.interval = setInterval(() => {
                    if (!this.isPaused) {
                        const elapsed = Date.now() - this.recordingStartTime;
                        const hours = Math.floor(elapsed / 3600000);
                        const minutes = Math.floor((elapsed % 3600000) / 60000);
                        const seconds = Math.floor((elapsed % 60000) / 1000);
                        
                        this.recordingTimer.textContent = 
                            `${hours.toString().padStart(2, '0')}:` +
                            `${minutes.toString().padStart(2, '0')}:` +
                            `${seconds.toString().padStart(2, '0')}`;
                    }
                }, 100);
            }

            stopRecordingTimer() {
                if (this.recordingTimer.interval) {
                    clearInterval(this.recordingTimer.interval);
                    this.recordingTimer.interval = null;
                }
            }

            updateWaveform(level) {
                // Add new level to data
                this.waveformData.push(level);
                if (this.waveformData.length > 100) {
                    this.waveformData.shift();
                }
            }

            startWaveformAnimation() {
                const animate = () => {
                    this.drawWaveform();
                    this.animationFrame = requestAnimationFrame(animate);
                };
                animate();
            }

            stopWaveformAnimation() {
                if (this.animationFrame) {
                    cancelAnimationFrame(this.animationFrame);
                    this.animationFrame = null;
                }
            }

            drawWaveform() {
                const width = this.waveformCanvas.width;
                const height = this.waveformCanvas.height;
                const ctx = this.waveformCtx;
                
                // Clear canvas
                ctx.fillStyle = '#0a0a0a';
                ctx.fillRect(0, 0, width, height);
                
                // Draw waveform
                const barWidth = width / this.waveformData.length;
                const centerY = height / 2;
                
                ctx.strokeStyle = '#00d4ff';
                ctx.lineWidth = 2;
                ctx.beginPath();
                
                this.waveformData.forEach((level, index) => {
                    const x = index * barWidth;
                    const amplitude = level * (height / 2) * 0.8;
                    
                    if (index === 0) {
                        ctx.moveTo(x, centerY);
                    }
                    
                    // Create smooth waveform
                    const cp1x = x + barWidth / 2;
                    const cp1y = centerY - amplitude;
                    const cp2x = x + barWidth / 2;
                    const cp2y = centerY + amplitude;
                    
                    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x + barWidth, centerY);
                });
                
                ctx.stroke();
                
                // Add glow effect
                ctx.shadowBlur = 10;
                ctx.shadowColor = '#00d4ff';
                ctx.stroke();
            }

            updateProgress(data) {
                // Smooth progress animation
                const updateBar = (bar, valueEl, progress) => {
                    bar.style.width = `${progress}%`;
                    valueEl.textContent = `${progress}%`;
                };
                
                if (data.transcription !== undefined) {
                    updateBar(this.transcriptionBar, this.transcriptionValue, data.transcription);
                }
                
                if (data.diarization !== undefined) {
                    updateBar(this.diarizationBar, this.diarizationValue, data.diarization);
                }
                
                if (data.speakers_detected !== undefined) {
                    this.speakerCount.textContent = data.speakers_detected;
                }
            }

            handleProcessingComplete(data) {
                // Hide progress
                this.progressContainer.classList.remove('active');
                
                // Show results
                this.resultsContainer.classList.add('active');
                
                // Update stats
                const duration = data.duration || 0;
                const minutes = Math.floor(duration / 60);
                const seconds = Math.floor(duration % 60);
                this.statDuration.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                this.statSpeakers.textContent = data.speakers_detected || 0;
                
                // Display transcription
                if (data.transcription_data && data.transcription_data.length > 0) {
                    let wordCount = 0;
                    let transcriptHTML = '';
                    
                    data.transcription_data.forEach((segment, index) => {
                        const words = segment.text ? segment.text.split(' ').length : 0;
                        wordCount += words;
                        
                        const speaker = data.diarization_segments && data.diarization_segments[index] 
                            ? data.diarization_segments[index].speaker 
                            : 'Speaker 1';
                        
                        transcriptHTML += `
                            <div class="transcript-segment">
                                <span class="speaker-tag">${speaker}</span>
                                <div class="transcript-text">${segment.text || ''}</div>
                            </div>
                        `;
                    });
                    
                    this.statWords.textContent = wordCount;
                    this.transcriptBox.innerHTML = transcriptHTML;
                } else {
                    this.transcriptBox.innerHTML = '<p>No transcription data available.</p>';
                }
                
                // Reset waveform
                setTimeout(() => {
                    this.waveformContainer.style.display = 'none';
                    this.waveformData = new Array(100).fill(0);
                }, 1000);
            }

            updateConnectionStatus(connected) {
                this.connectionIndicator.style.background = connected ? '#00ff88' : '#ff3366';
                this.connectionText.textContent = connected ? 'Connected' : 'Disconnected';
                
                // Disable controls if disconnected
                if (!connected && !this.isRecording) {
                    this.startBtn.disabled = true;
                } else if (connected && !this.isRecording) {
                    this.startBtn.disabled = false;
                }
            }

            showError(message) {
                const errorToast = document.createElement('div');
                errorToast.className = 'error-toast';
                errorToast.textContent = message;
                document.body.appendChild(errorToast);
                
                setTimeout(() => {
                    errorToast.style.animation = 'slideInRight 0.3s ease reverse';
                    setTimeout(() => {
                        document.body.removeChild(errorToast);
                    }, 300);
                }, 5000);
            }
        }

        // Initialize application
        document.addEventListener('DOMContentLoaded', () => {
            new TranscrevAI();
        });
    </script>
</body>
</html>
"""

@app.get("/")
async def get_home():
    return HTMLResponse(content=HTML_TEMPLATE)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "active_sessions": len(app_state.sessions)
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    # Ensure all data directories exist
    for subdir in ["recordings", "transcripts", "models", "temp", "processed"]:
        FileManager.ensure_directory_exists(FileManager.get_data_path(subdir))
    logger.info("TranscrevAI application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup all active sessions
    for session_id in list(app_state.sessions.keys()):
        await app_state.cleanup_session(session_id)
    logger.info("TranscrevAI application shutdown complete")

if __name__ == "__main__":
    # Production configuration
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # Disable reload for production
        log_level="info",
        access_log=True
    )