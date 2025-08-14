# TranscrevAI - Production Ready with Model Management
import asyncio
import logging
import os
import time
import json
import urllib.request
import zipfile
import shutil
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from src.audio_processing import AudioRecorder
from src.transcription import transcribe_audio_with_progress
from src.speaker_diarization import SpeakerDiarization
from src.file_manager import FileManager
from src.subtitle_generator import generate_srt
from config.app_config import MODEL_DIR
from src.logging_setup import setup_app_logging

logger = setup_app_logging()

app = FastAPI(
    title="TranscrevAI",
    description="Real-time Audio Transcription - Production Ready",
    version="1.0.0"
)

# Model URLs for auto-download
MODEL_URLS = {
    "pt": "https://alphacephei.com/vosk/models/vosk-model-pt-0.3.zip",
    "en": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
    "es": "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip",
    "fr": "https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip"
}

class ModelManager:
    """Handles model downloading and validation"""
    
    @staticmethod
    def get_model_path(language: str) -> str:
        """Get the path for a specific language model"""
        return os.path.join(MODEL_DIR, language)
    
    @staticmethod
    def validate_model(language: str) -> bool:
        """Validate if model exists and has required files"""
        model_path = ModelManager.get_model_path(language)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model directory not found: {model_path}")
            return False
        
        # Check for required Vosk model files
        required_files = ['final.mdl', 'Gr.fst', 'HCLr.fst']
        required_dirs = ['am', 'conf', 'graph']
        
        # Check files
        for file_name in required_files:
            file_path = os.path.join(model_path, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"Missing model file: {file_path}")
                return False
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = os.path.join(model_path, dir_name)
            if not os.path.exists(dir_path):
                logger.warning(f"Missing model directory: {dir_path}")
                return False
        
        logger.info(f"Model validation passed for: {language}")
        return True
    
    @staticmethod
    async def download_model(language: str) -> bool:
        """Download and extract model for specified language"""
        if language not in MODEL_URLS:
            logger.error(f"No model URL available for language: {language}")
            return False
        
        model_url = MODEL_URLS[language]
        model_path = ModelManager.get_model_path(language)
        zip_path = f"{model_path}.zip"
        
        try:
            logger.info(f"Downloading model for {language}...")
            
            # Create model directory
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Download with progress (simplified)
            urllib.request.urlretrieve(model_url, zip_path)
            logger.info(f"Downloaded model to: {zip_path}")
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to temporary directory first
                temp_dir = f"{model_path}_temp"
                zip_ref.extractall(temp_dir)
                
                # Find the actual model directory (usually nested)
                extracted_dirs = [d for d in os.listdir(temp_dir) 
                                if os.path.isdir(os.path.join(temp_dir, d))]
                
                if extracted_dirs:
                    # Move the first directory content to final location
                    src_dir = os.path.join(temp_dir, extracted_dirs[0])
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    shutil.move(src_dir, model_path)
                    
                    # Clean up
                    shutil.rmtree(temp_dir)
                    os.remove(zip_path)
                    
                    logger.info(f"Model extracted to: {model_path}")
                    return ModelManager.validate_model(language)
                else:
                    logger.error("No directories found in extracted model")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to download model for {language}: {e}")
            # Clean up on failure
            for path in [zip_path, f"{model_path}_temp"]:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            return False
    
    @staticmethod
    async def ensure_model(language: str) -> bool:
        """Ensure model exists, download if necessary"""
        if ModelManager.validate_model(language):
            return True
        
        logger.info(f"Model not found for {language}, attempting download...")
        return await ModelManager.download_model(language)

# Simple state management
class SimpleState:
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, session_id: str):
        try:
            self.sessions[session_id] = {
                "recorder": AudioRecorder(),
                "recording": False,
                "paused": False,
                "progress": {"transcription": 0, "diarization": 0},
                "websocket": None,
                "start_time": None,
                "task": None,
                "language": "pt"  # Default language
            }
            logger.info(f"Session created: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False
    
    def get_session(self, session_id: str):
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, updates: dict):
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
    
    async def cleanup_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.get("recorder"):
                try:
                    await session["recorder"].cleanup_resources()
                except:
                    pass
            if session.get("task"):
                session["task"].cancel()
            del self.sessions[session_id]
            logger.info(f"Session cleaned up: {session_id}")

# Simple WebSocket manager
class SimpleWebSocketManager:
    def __init__(self):
        self.connections = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")
    
    async def disconnect(self, session_id: str):
        if session_id in self.connections:
            del self.connections[session_id]
        await app_state.cleanup_session(session_id)
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.connections:
            try:
                await self.connections[session_id].send_json(message)
            except Exception as e:
                logger.error(f"Send message failed: {e}")
                await self.disconnect(session_id)

# Global instances
app_state = SimpleState()
websocket_manager = SimpleWebSocketManager()

# Updated HTML with better error display
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TranscrevAI - Simple Transcription</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #333;
        }

        .app {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 2rem;
            max-width: 600px;
            width: 90%;
            text-align: center;
        }

        .logo {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .tagline {
            color: #666;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }

        .status {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .status.connecting { background: #fff3cd; color: #856404; }
        .status.ready { background: #d4edda; color: #155724; }
        .status.recording { background: #f8d7da; color: #721c24; }
        .status.processing { background: #cce5ff; color: #004085; }
        .status.downloading { background: #e2e3e5; color: #383d41; }
        .status.error { background: #f5c6cb; color: #721c24; }

        .controls {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin: 2rem 0;
            flex-wrap: wrap;
        }

        .btn {
            padding: 0.8rem 1.5rem;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 120px;
        }

        .btn:disabled {
            opacity: 0.5;
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

        .waveform {
            height: 80px;
            background: #f8f9fa;
            border-radius: 12px;
            margin: 1.5rem 0;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px dashed #dee2e6;
            transition: all 0.3s ease;
        }

        .waveform.active {
            background: linear-gradient(135deg, #667eea20, #764ba220);
            border-color: #667eea;
            border-style: solid;
        }

        .waveform-bars {
            display: flex;
            gap: 2px;
            align-items: end;
            height: 40px;
        }

        .bar {
            width: 3px;
            background: #667eea;
            border-radius: 2px;
            transition: height 0.1s ease;
        }

        .progress-section {
            margin: 2rem 0;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s ease;
        }

        .progress-section.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .progress-item {
            margin: 1rem 0;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            font-weight: 500;
        }

        .progress-bar {
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
            border-radius: 4px;
        }

        .results {
            margin-top: 2rem;
            text-align: left;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s ease;
        }

        .results.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .result-item {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #667eea;
        }

        .speaker-tag {
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 0.5rem;
        }

        .settings {
            margin-bottom: 2rem;
        }

        .settings select {
            padding: 0.5rem;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 1rem;
            background: white;
        }

        .error-details {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            font-size: 0.9rem;
            text-align: left;
        }

        /* Responsive design */
        @media (max-width: 600px) {
            .app {
                padding: 1rem;
                margin: 1rem;
            }
            
            .logo {
                font-size: 2rem;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 250px;
            }
        }

        /* Animations */
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .recording .btn-danger {
            animation: pulse 2s infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .downloading::before {
            content: "⬇ ";
            animation: spin 1s linear infinite;
        }
    </style>
</head>
<body>
    <div class="app">
        <div class="logo">🎤 TranscrevAI</div>
        <div class="tagline">Simple, real-time audio transcription</div>
        
        <div class="settings">
            <select id="language">
                <option value="pt">Português</option>
                <option value="en">English</option>
                <option value="es">Español</option>
                <option value="fr">Français</option>
            </select>
        </div>

        <div id="status" class="status connecting">Connecting...</div>

        <div class="controls">
            <button id="startBtn" class="btn btn-primary" disabled>Start</button>
            <button id="pauseBtn" class="btn btn-secondary" disabled>Pause</button>
            <button id="stopBtn" class="btn btn-danger" disabled>Stop</button>
        </div>

        <div id="waveform" class="waveform">
            <div id="waveform-content">Ready to record</div>
        </div>

        <div id="progress" class="progress-section">
            <div class="progress-item">
                <div class="progress-label">
                    <span>Transcription</span>
                    <span id="transcription-percent">0%</span>
                </div>
                <div class="progress-bar">
                    <div id="transcription-fill" class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="progress-item">
                <div class="progress-label">
                    <span>Diarization</span>
                    <span id="diarization-percent">0%</span>
                </div>
                <div class="progress-bar">
                    <div id="diarization-fill" class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>

        <div id="error-details" class="error-details" style="display: none;"></div>
        <div id="results" class="results"></div>
    </div>

    <script>
        class TranscrevAI {
            constructor() {
                this.ws = null;
                this.sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                this.isRecording = false;
                this.isPaused = false;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                
                this.initElements();
                this.connect();
                this.setupEventListeners();
                this.createWaveformBars();
            }

            initElements() {
                this.statusEl = document.getElementById('status');
                this.startBtn = document.getElementById('startBtn');
                this.pauseBtn = document.getElementById('pauseBtn');
                this.stopBtn = document.getElementById('stopBtn');
                this.languageEl = document.getElementById('language');
                this.waveformEl = document.getElementById('waveform');
                this.waveformContent = document.getElementById('waveform-content');
                this.progressEl = document.getElementById('progress');
                this.resultsEl = document.getElementById('results');
                this.errorDetailsEl = document.getElementById('error-details');
                this.transcriptionFill = document.getElementById('transcription-fill');
                this.diarizationFill = document.getElementById('diarization-fill');
                this.transcriptionPercent = document.getElementById('transcription-percent');
                this.diarizationPercent = document.getElementById('diarization-percent');
            }

            createWaveformBars() {
                const barsContainer = document.createElement('div');
                barsContainer.className = 'waveform-bars';
                barsContainer.style.display = 'none';
                
                for (let i = 0; i < 20; i++) {
                    const bar = document.createElement('div');
                    bar.className = 'bar';
                    bar.style.height = '5px';
                    barsContainer.appendChild(bar);
                }
                
                this.waveformEl.appendChild(barsContainer);
                this.waveformBars = barsContainer;
            }

            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    this.setStatus('Ready to record', 'ready');
                    this.startBtn.disabled = false;
                    this.reconnectAttempts = 0;
                    this.hideError();
                };
                
                this.ws.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    } catch (e) {
                        console.error('Message parse error:', e);
                    }
                };
                
                this.ws.onclose = () => {
                    this.setStatus('Connection lost', 'error');
                    this.handleReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.setStatus('Connection error', 'error');
                };
            }

            handleReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    this.setStatus(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`, 'connecting');
                    setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
                } else {
                    this.setStatus('Failed to reconnect. Refresh page.', 'error');
                }
            }

            setupEventListeners() {
                this.startBtn.onclick = () => this.startRecording();
                this.pauseBtn.onclick = () => this.togglePause();
                this.stopBtn.onclick = () => this.stopRecording();
                
                // Heartbeat
                setInterval(() => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.send('ping');
                    }
                }, 30000);
            }

            send(type, data = {}) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type, data }));
                }
            }

            handleMessage(message) {
                switch (message.type) {
                    case 'model_downloading':
                        this.setStatus(`Downloading ${message.language} model... Please wait.`, 'downloading');
                        this.startBtn.disabled = true;
                        break;
                        
                    case 'model_ready':
                        this.setStatus('Model ready! You can start recording.', 'ready');
                        this.startBtn.disabled = false;
                        break;
                        
                    case 'recording_started':
                        this.isRecording = true;
                        this.setStatus('Recording...', 'recording');
                        this.startBtn.disabled = true;
                        this.pauseBtn.disabled = false;
                        this.stopBtn.disabled = false;
                        this.showWaveform(true);
                        this.hideError();
                        break;
                        
                    case 'recording_paused':
                        this.isPaused = true;
                        this.setStatus('Paused', 'recording');
                        this.pauseBtn.textContent = 'Resume';
                        break;
                        
                    case 'recording_resumed':
                        this.isPaused = false;
                        this.setStatus('Recording...', 'recording');
                        this.pauseBtn.textContent = 'Pause';
                        break;
                        
                    case 'recording_stopped':
                        this.isRecording = false;
                        this.setStatus('Processing...', 'processing');
                        this.resetControls();
                        this.showWaveform(false);
                        this.showProgress(true);
                        break;
                        
                    case 'audio_level':
                        this.updateWaveform(message.level);
                        break;
                        
                    case 'progress':
                        this.updateProgress(message.transcription, message.diarization);
                        break;
                        
                    case 'processing_complete':
                        this.setStatus('Complete!', 'ready');
                        this.showResults(message);
                        this.showProgress(false);
                        break;
                        
                    case 'error':
                        this.setStatus(`Error: ${message.message}`, 'error');
                        this.showError(message.message, message.details);
                        this.resetControls();
                        this.showWaveform(false);
                        break;
                }
            }

            setStatus(text, type) {
                this.statusEl.textContent = text;
                this.statusEl.className = `status ${type}`;
            }

            showError(message, details) {
                if (details) {
                    this.errorDetailsEl.innerHTML = `<strong>Technical Details:</strong><br>${details}`;
                    this.errorDetailsEl.style.display = 'block';
                } else {
                    this.hideError();
                }
            }

            hideError() {
                this.errorDetailsEl.style.display = 'none';
            }

            showWaveform(show) {
                this.waveformContent.style.display = show ? 'none' : 'block';
                this.waveformBars.style.display = show ? 'flex' : 'none';
                this.waveformEl.classList.toggle('active', show);
            }

            updateWaveform(level) {
                if (this.isRecording && !this.isPaused) {
                    const bars = this.waveformBars.children;
                    for (let i = 0; i < bars.length; i++) {
                        const height = Math.random() * level * 40 + 5;
                        bars[i].style.height = `${height}px`;
                    }
                }
            }

            showProgress(show) {
                this.progressEl.classList.toggle('visible', show);
                if (!show) {
                    this.updateProgress(0, 0);
                }
            }

            updateProgress(transcription, diarization) {
                this.transcriptionFill.style.width = `${transcription}%`;
                this.diarizationFill.style.width = `${diarization}%`;
                this.transcriptionPercent.textContent = `${transcription}%`;
                this.diarizationPercent.textContent = `${diarization}%`;
            }

            showResults(data) {
                let html = '';
                if (data.transcription_data && data.transcription_data.length > 0) {
                    data.transcription_data.forEach((item, i) => {
                        html += `
                            <div class="result-item">
                                <div class="speaker-tag">Segment ${i + 1}</div>
                                <div>${item.text || 'No transcription'}</div>
                            </div>
                        `;
                    });
                } else {
                    html = '<div class="result-item">No transcription available</div>';
                }
                
                if (data.speakers_detected > 0) {
                    html = `<div class="result-item"><strong>${data.speakers_detected} speakers detected</strong></div>` + html;
                }
                
                this.resultsEl.innerHTML = html;
                this.resultsEl.classList.add('visible');
            }

            resetControls() {
                this.startBtn.disabled = false;
                this.pauseBtn.disabled = true;
                this.pauseBtn.textContent = 'Pause';
                this.stopBtn.disabled = true;
                this.isRecording = false;
                this.isPaused = false;
            }

            startRecording() {
                this.resultsEl.classList.remove('visible');
                this.send('start_recording', { language: this.languageEl.value });
            }

            togglePause() {
                if (this.isPaused) {
                    this.send('resume_recording');
                } else {
                    this.send('pause_recording');
                }
            }

            stopRecording() {
                this.send('stop_recording');
            }
        }

        // Start the app when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new TranscrevAI();
        });
    </script>
</body>
</html>
"""

# Health check
@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "sessions": len(app_state.sessions)
    })

# Main interface
@app.get("/", response_class=HTMLResponse)
async def main_interface():
    return HTMLResponse(content=HTML_INTERFACE)

# API endpoint
@app.get("/api")
async def api_status():
    return {"message": "TranscrevAI API is running", "version": "1.0.0"}

# WebSocket handler with model management
@app.websocket("/ws/{session_id}")
async def websocket_handler(websocket: WebSocket, session_id: str):
    await websocket_manager.connect(websocket, session_id)
    
    if not app_state.create_session(session_id):
        await websocket_manager.send_message(session_id, {
            "type": "error", 
            "message": "Failed to create session"
        })
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            await handle_websocket_message(session_id, data)
    except WebSocketDisconnect:
        await websocket_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket_manager.disconnect(session_id)

# Enhanced message handler with model management
async def handle_websocket_message(session_id: str, data: dict):
    message_type = data.get("type")
    message_data = data.get("data", {})
    session = app_state.get_session(session_id)
    
    if not session:
        await websocket_manager.send_message(session_id, {
            "type": "error",
            "message": "Session not found"
        })
        return
    
    recorder = session["recorder"]
    
    if message_type == "start_recording":
        if not session["recording"]:
            try:
                language = message_data.get("language", "pt")
                app_state.update_session(session_id, {"language": language})
                
                # Check and download model if needed
                await websocket_manager.send_message(session_id, {
                    "type": "model_downloading",
                    "language": language,
                    "message": f"Checking {language} model..."
                })
                
                model_ready = await ModelManager.ensure_model(language)
                if not model_ready:
                    await websocket_manager.send_message(session_id, {
                        "type": "error",
                        "message": f"Failed to load {language} model",
                        "details": "Model download or validation failed. Please check your internet connection and try again."
                    })
                    return
                
                await websocket_manager.send_message(session_id, {
                    "type": "model_ready",
                    "language": language
                })
                
                # Start recording
                await recorder.start_recording()
                
                app_state.update_session(session_id, {
                    "recording": True,
                    "start_time": time.time()
                })
                
                await websocket_manager.send_message(session_id, {
                    "type": "recording_started",
                    "message": "Recording started"
                })
                
                # Start audio monitoring and processing
                asyncio.create_task(monitor_audio(session_id))
                task = asyncio.create_task(process_audio(session_id, language))
                app_state.update_session(session_id, {"task": task})
                
            except Exception as e:
                logger.error(f"Start recording error: {e}")
                await websocket_manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Failed to start recording: {str(e)}",
                    "details": f"Technical error: {str(e)}"
                })
    
    elif message_type == "stop_recording":
        if session["recording"]:
            try:
                await recorder.stop_recording()
                duration = time.time() - session.get("start_time", time.time())
                
                app_state.update_session(session_id, {
                    "recording": False,
                    "duration": duration
                })
                
                await websocket_manager.send_message(session_id, {
                    "type": "recording_stopped",
                    "message": "Recording stopped",
                    "duration": duration
                })
                
            except Exception as e:
                logger.error(f"Stop recording error: {e}")
                await websocket_manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Failed to stop recording: {str(e)}"
                })
    
    elif message_type == "pause_recording":
        if session["recording"] and not session["paused"]:
            recorder.pause_recording()
            app_state.update_session(session_id, {"paused": True})
            await websocket_manager.send_message(session_id, {
                "type": "recording_paused"
            })
    
    elif message_type == "resume_recording":
        if session["recording"] and session["paused"]:
            recorder.resume_recording()
            app_state.update_session(session_id, {"paused": False})
            await websocket_manager.send_message(session_id, {
                "type": "recording_resumed"
            })
    
    elif message_type == "ping":
        await websocket_manager.send_message(session_id, {"type": "pong"})

# Simple audio monitoring
async def monitor_audio(session_id: str):
    try:
        session = app_state.get_session(session_id)
        while session and session["recording"]:
            if not session["paused"]:
                # Simulate audio level
                import random
                level = random.uniform(0.1, 1.0) if random.random() > 0.3 else 0.0
                
                await websocket_manager.send_message(session_id, {
                    "type": "audio_level",
                    "level": level
                })
            
            await asyncio.sleep(0.1)
            session = app_state.get_session(session_id)
    except Exception as e:
        logger.error(f"Audio monitoring error: {e}")

# Enhanced processing pipeline with better error handling
async def process_audio(session_id: str, language: str = "pt"):
    try:
        session = app_state.get_session(session_id)
        if not session:
            return
        
        # Wait for recording to complete
        while session and session["recording"]:
            await asyncio.sleep(0.1)
            session = app_state.get_session(session_id)
        
        # Get audio file
        if not session or not session.get("recorder"):
            return
            
        audio_file = session["recorder"].output_file
        
        # Validate file
        if not os.path.exists(audio_file):
            raise Exception(f"Audio file not found: {audio_file}")
            
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            raise Exception("Audio file is empty")
        
        logger.info(f"Processing: {audio_file} (size: {file_size} bytes)")
        
        # Double-check model before transcription
        model_path = ModelManager.get_model_path(language)
        if not ModelManager.validate_model(language):
            raise Exception(f"Model validation failed for {language}")
        
        # Transcription with progress
        transcription_data = []
        try:
            async for progress, data in transcribe_audio_with_progress(
                audio_file, model_path, language
            ):
                session = app_state.get_session(session_id)
                if session:
                    app_state.update_session(session_id, {
                        "progress": {"transcription": progress, "diarization": session["progress"]["diarization"]}
                    })
                    
                    await websocket_manager.send_message(session_id, {
                        "type": "progress",
                        "transcription": progress,
                        "diarization": session["progress"]["diarization"]
                    })
                
                transcription_data = data
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Transcription failed: {error_msg}")
            
            # Provide specific error details
            details = f"Language: {language}\nModel path: {model_path}\nAudio file: {audio_file}\nFile size: {file_size} bytes\nError: {error_msg}"
            
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": "Transcription failed. Please try again or check if the model is properly installed.",
                "details": details
            })
            return
        
        # Diarization (optional, can fail gracefully)
        try:
            diarizer = SpeakerDiarization()
            diarization_segments = await diarizer.diarize_audio(audio_file)
            unique_speakers = len(set(seg.get("speaker", "Unknown") for seg in diarization_segments))
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")
            diarization_segments = []
            unique_speakers = 0
        
        # Update final progress
        app_state.update_session(session_id, {
            "progress": {"transcription": 100, "diarization": 100}
        })
        
        await websocket_manager.send_message(session_id, {
            "type": "progress",
            "transcription": 100,
            "diarization": 100
        })
        
        # Generate SRT (optional, can fail gracefully)
        try:
            srt_file = await generate_srt(transcription_data, diarization_segments)
        except Exception as e:
            logger.warning(f"SRT generation failed: {e}")
            srt_file = None
        
        # Send results
        await websocket_manager.send_message(session_id, {
            "type": "processing_complete",
            "transcription_data": transcription_data,
            "diarization_segments": diarization_segments,
            "speakers_detected": unique_speakers,
            "srt_file": srt_file,
            "duration": session.get("duration", 0)
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Processing error: {error_msg}")
        await websocket_manager.send_message(session_id, {
            "type": "error",
            "message": f"Processing failed: {error_msg}",
            "details": f"Session: {session_id}\nLanguage: {language}\nError: {error_msg}"
        })

# Production startup
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False
    )
