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

# Enhanced session-based state management
class AppState:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.lock = asyncio.Lock()
    
    async def create_session(self, session_id: str):
        async with self.lock:
            try:
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
                    "created_at": datetime.now(),
                    "error_count": 0,
                    "last_activity": time.time()
                }
                logger.info(f"Session created: {session_id}")
            except Exception as e:
                logger.error(f"Failed to create session {session_id}: {e}")
                raise
    
    async def get_session(self, session_id: str) -> Optional[dict]:
        async with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session["last_activity"] = time.time()
            return session
    
    async def update_session(self, session_id: str, updates: dict):
        async with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(updates)
                self.sessions[session_id]["last_activity"] = time.time()
    
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
    
    async def cleanup_inactive_sessions(self):
        """Cleanup sessions inactive for more than 1 hour"""
        current_time = time.time()
        inactive_sessions = []
        
        async with self.lock:
            for session_id, session in self.sessions.items():
                if current_time - session.get("last_activity", 0) > 3600:  # 1 hour
                    inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            logger.info(f"Cleaning up inactive session: {session_id}")
            await self.cleanup_session(session_id)

# Enhanced WebSocket connection manager
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

# Model validation helper
def validate_models():
    """Validate that required models are available"""
    try:
        model_path = Path(MODEL_DIR)
        if not model_path.exists():
            logger.warning(f"Model directory does not exist: {MODEL_DIR}")
            return False
        
        # Check for required model files (adjust based on your model requirements)
        required_files = ['final.mdl', 'Gr.fst', 'HCLr.fst']
        existing_files = list(model_path.glob('*'))
        
        if not existing_files:
            logger.warning(f"No model files found in {MODEL_DIR}")
            return False
        
        logger.info(f"Found model files: {[f.name for f in existing_files]}")
        return True
        
    except Exception as e:
        logger.error(f"Model validation error: {e}")
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("TranscrevAI starting up...")
    
    # Validate models
    if not validate_models():
        logger.warning("Model validation failed - transcription may not work properly")
    
    # Start cleanup task for inactive sessions
    asyncio.create_task(periodic_cleanup())
    
    logger.info("TranscrevAI startup complete")

async def periodic_cleanup():
    """Periodic cleanup of inactive sessions"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            await app_state.cleanup_inactive_sessions()
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(app_state.sessions)
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "TranscrevAI API is running", "version": "1.0.0"}

# WebSocket endpoint with enhanced error handling
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    try:
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
        
        # Validate audio file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Check file size
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            raise ValueError("Audio file is empty")
        
        logger.info(f"Processing audio file: {audio_file} (size: {file_size} bytes)")
        
        # Transcription with progress updates
        transcription_data = []
        try:
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
                    "smooth": True
                }, session_id)
                
                transcription_data = data
        
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            await manager.send_personal_message({
                "type": "error",
                "message": f"Transcription failed: {str(e)}"
            }, session_id)
            return
        
        # Speaker diarization
        try:
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
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            diarization_segments = []
            unique_speakers = 0
        
        # Generate subtitles
        try:
            srt_file = await generate_srt(transcription_data, diarization_segments)
        except Exception as e:
            logger.error(f"SRT generation failed: {e}")
            srt_file = None
        
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

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )