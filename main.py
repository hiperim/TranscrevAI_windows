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

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# HTML Interface Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TranscrevAI - Audio Transcription</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 2rem;
            max-width: 800px;
            width: 90%;
            margin: 2rem;
        }
        
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 0.8rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .btn-primary {
            background: #4CAF50;
            color: white;
        }
        
        .btn-primary:hover:not(:disabled) {
            background: #45a049;
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: #2196F3;
            color: white;
        }
        
        .btn-secondary:hover:not(:disabled) {
            background: #1976D2;
            transform: translateY(-2px);
        }
        
        .btn-danger {
            background: #f44336;
            color: white;
        }
        
        .btn-danger:hover:not(:disabled) {
            background: #da190b;
            transform: translateY(-2px);
        }
        
        .status {
            text-align: center;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 8px;
            font-weight: 500;
        }
        
        .status.connected {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.recording {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .status.processing {
            background: #cce5ff;
            color: #004085;
            border: 1px solid #99d6ff;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .progress-section {
            margin: 2rem 0;
        }
        
        .progress-item {
            margin: 1rem 0;
        }
        
        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        
        .waveform {
            height: 100px;
            background: #f5f5f5;
            border-radius: 8px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px dashed #ddd;
        }
        
        .waveform.active {
            background: #e8f5e8;
            border-color: #4CAF50;
        }
        
        .results {
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid #eee;
        }
        
        .results h3 {
            margin-bottom: 1rem;
            color: #333;
        }
        
        .transcription-result {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #4CAF50;
        }
        
        .speaker-info {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 0.5rem;
        }
        
        .settings {
            margin: 1rem 0;
        }
        
        .settings label {
            display: block;
            margin: 0.5rem 0;
            font-weight: 500;
        }
        
        .settings select {
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1rem;
            width: 200px;
        }
        
        @media (max-width: 600px) {
            .container {
                margin: 1rem;
                padding: 1rem;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 300px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 TranscrevAI</h1>
            <p>Real-time Audio Transcription with Speaker Diarization</p>
        </div>
        
        <div class="settings">
            <label for="language-select">Language:</label>
            <select id="language-select">
                <option value="en">English</option>
                <option value="pt">Portuguese</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
            </select>
        </div>
        
        <div class="status" id="status">Connecting...</div>
        
        <div class="controls">
            <button class="btn btn-primary" id="start-btn" disabled>Start Recording</button>
            <button class="btn btn-secondary" id="pause-btn" disabled>Pause</button>
            <button class="btn btn-danger" id="stop-btn" disabled>Stop Recording</button>
        </div>
        
        <div class="waveform" id="waveform">
            <span id="waveform-text">Audio waveform will appear here</span>
        </div>
        
        <div class="progress-section" id="progress-section" style="display: none;">
            <div class="progress-item">
                <div class="progress-label">
                    <span>Transcription</span>
                    <span id="transcription-percent">0%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="transcription-progress" style="width: 0%;"></div>
                </div>
            </div>
            
            <div class="progress-item">
                <div class="progress-label">
                    <span>Speaker Diarization</span>
                    <span id="diarization-percent">0%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="diarization-progress" style="width: 0%;"></div>
                </div>
            </div>
        </div>
        
        <div class="results" id="results" style="display: none;">
            <h3>Results</h3>
            <div id="results-content"></div>
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
                
                this.initializeElements();
                this.initializeWebSocket();
                this.setupEventListeners();
            }
            
            generateSessionId() {
                return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            }
            
            initializeElements() {
                this.statusEl = document.getElementById('status');
                this.startBtn = document.getElementById('start-btn');
                this.pauseBtn = document.getElementById('pause-btn');
                this.stopBtn = document.getElementById('stop-btn');
                this.languageSelect = document.getElementById('language-select');
                this.waveformEl = document.getElementById('waveform');
                this.waveformText = document.getElementById('waveform-text');
                this.progressSection = document.getElementById('progress-section');
                this.resultsEl = document.getElementById('results');
                this.resultsContent = document.getElementById('results-content');
                this.transcriptionProgress = document.getElementById('transcription-progress');
                this.diarizationProgress = document.getElementById('diarization-progress');
                this.transcriptionPercent = document.getElementById('transcription-percent');
                this.diarizationPercent = document.getElementById('diarization-percent');
            }
            
            initializeWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    this.isConnected = true;
                    this.updateStatus('Connected and ready', 'connected');
                    this.startBtn.disabled = false;
                };
                
                this.ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                };
                
                this.ws.onclose = () => {
                    this.isConnected = false;
                    this.updateStatus('Connection lost. Refreshing...', 'error');
                    setTimeout(() => {
                        window.location.reload();
                    }, 3000);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateStatus('Connection error', 'error');
                };
            }
            
            setupEventListeners() {
                this.startBtn.addEventListener('click', () => this.startRecording());
                this.pauseBtn.addEventListener('click', () => this.togglePause());
                this.stopBtn.addEventListener('click', () => this.stopRecording());
                
                // Heartbeat
                setInterval(() => {
                    if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                        this.sendMessage('ping');
                    }
                }, 30000);
            }
            
            sendMessage(type, data = {}) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type, data }));
                }
            }
            
            handleMessage(message) {
                switch (message.type) {
                    case 'config':
                        console.log('Configuration received:', message.data);
                        break;
                        
                    case 'recording_started':
                        this.isRecording = true;
                        this.updateStatus('Recording in progress...', 'recording');
                        this.startBtn.disabled = true;
                        this.pauseBtn.disabled = false;
                        this.stopBtn.disabled = false;
                        this.waveformEl.classList.add('active');
                        this.waveformText.textContent = 'Recording audio...';
                        break;
                        
                    case 'recording_paused':
                        this.isPaused = true;
                        this.updateStatus('Recording paused', 'recording');
                        this.pauseBtn.textContent = 'Resume';
                        break;
                        
                    case 'recording_resumed':
                        this.isPaused = false;
                        this.updateStatus('Recording in progress...', 'recording');
                        this.pauseBtn.textContent = 'Pause';
                        break;
                        
                    case 'recording_stopped':
                        this.isRecording = false;
                        this.isPaused = false;
                        this.updateStatus('Processing audio...', 'processing');
                        this.startBtn.disabled = false;
                        this.pauseBtn.disabled = true;
                        this.pauseBtn.textContent = 'Pause';
                        this.stopBtn.disabled = true;
                        this.waveformEl.classList.remove('active');
                        this.waveformText.textContent = 'Processing...';
                        this.progressSection.style.display = 'block';
                        break;
                        
                    case 'audio_level':
                        this.updateWaveform(message.level);
                        break;
                        
                    case 'progress':
                        this.updateProgress(message.transcription, message.diarization);
                        break;
                        
                    case 'processing_complete':
                        this.handleProcessingComplete(message);
                        break;
                        
                    case 'error':
                        this.updateStatus(`Error: ${message.message}`, 'error');
                        this.resetControls();
                        break;
                        
                    case 'pong':
                        // Heartbeat response
                        break;
                        
                    default:
                        console.log('Unknown message:', message);
                }
            }
            
            updateStatus(text, className) {
                this.statusEl.textContent = text;
                this.statusEl.className = `status ${className}`;
            }
            
            updateWaveform(level) {
                if (this.isRecording && !this.isPaused) {
                    const intensity = Math.floor(level * 100);
                    this.waveformText.textContent = `Recording... Level: ${intensity}%`;
                }
            }
            
            updateProgress(transcription, diarization) {
                this.transcriptionProgress.style.width = `${transcription}%`;
                this.diarizationProgress.style.width = `${diarization}%`;
                this.transcriptionPercent.textContent = `${transcription}%`;
                this.diarizationPercent.textContent = `${diarization}%`;
            }
            
            handleProcessingComplete(message) {
                this.updateStatus('Processing complete!', 'connected');
                this.progressSection.style.display = 'none';
                this.resultsEl.style.display = 'block';
                
                let resultsHtml = '';
                
                if (message.speakers_detected > 0) {
                    resultsHtml += `<div class="speaker-info">Detected ${message.speakers_detected} speaker(s)</div>`;
                }
                
                if (message.transcription_data && message.transcription_data.length > 0) {
                    message.transcription_data.forEach((item, index) => {
                        resultsHtml += `
                            <div class="transcription-result">
                                <div class="speaker-info">Segment ${index + 1}</div>
                                <div>${item.text || 'No transcription available'}</div>
                            </div>
                        `;
                    });
                } else {
                    resultsHtml += `
                        <div class="transcription-result">
                            <div>No transcription data available</div>
                        </div>
                    `;
                }
                
                if (message.srt_file) {
                    resultsHtml += `
                        <div class="transcription-result">
                            <div class="speaker-info">SRT File: ${message.srt_file}</div>
                        </div>
                    `;
                }
                
                this.resultsContent.innerHTML = resultsHtml;
                this.resetControls();
            }
            
            resetControls() {
                this.isRecording = false;
                this.isPaused = false;
                this.startBtn.disabled = false;
                this.pauseBtn.disabled = true;
                this.pauseBtn.textContent = 'Pause';
                this.stopBtn.disabled = true;
                this.waveformEl.classList.remove('active');
                this.waveformText.textContent = 'Ready for next recording';
            }
            
            startRecording() {
                if (!this.isConnected) return;
                
                const language = this.languageSelect.value;
                this.sendMessage('start_recording', { language });
                this.resultsEl.style.display = 'none';
                this.progressSection.style.display = 'none';
            }
            
            togglePause() {
                if (!this.isRecording) return;
                
                if (this.isPaused) {
                    this.sendMessage('resume_recording');
                } else {
                    this.sendMessage('pause_recording');
                }
            }
            
            stopRecording() {
                if (!this.isRecording) return;
                
                this.sendMessage('stop_recording');
            }
        }
        
        // Initialize the application when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new TranscrevAI();
        });
    </script>
</body>
</html>
"""

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

# Main HTML interface
@app.get("/", response_class=HTMLResponse)
async def get_main_interface():
    """Serve the main HTML interface"""
    return HTMLResponse(content=HTML_TEMPLATE)

# API info endpoint
@app.get("/api")
async def api_info():
    """API information endpoint"""
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
