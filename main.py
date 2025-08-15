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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException 
from fastapi.responses import HTMLResponse, JSONResponse 
import uvicorn 

from src.audio_processing import AudioRecorder 
from src.transcription import transcribe_audio_with_progress
from src.speaker_diarization import SpeakerDiarization 
from src.file_manager import FileManager 
from src.subtitle_generator import generate_srt 
from config.app_config import MODEL_DIR, DATA_DIR, LANGUAGE_MODELS 
from src.logging_setup import setup_app_logging

logger = setup_app_logging()
if logger is None:
    import logging
    logger = logging.getLogger("TranscrevAI")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

# FastAPI setup 
app = FastAPI( 
    title="TranscrevAI", 
    description="Real-time Audio Transcription with AI", 
    version="1.0.0" 
)

import os

class ModelManager:
    # Background model management - no UI interference

    @staticmethod
    def get_model_path(language: str) -> str:
        # Use config-based model path
        return os.path.join(MODEL_DIR, language)
    
    @staticmethod
    def validate_model(language: str) -> bool:
        # Validate if model exists and has required files
        model_path = ModelManager.get_model_path(language)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model directory not found: {model_path}")
            return False
        
        # Check for required model files (actual structure)
        required_files = [
            'am/final.mdl',
            'graph/Gr.fst', 
            'graph/HCLr.fst',
            'conf/mfcc.conf'
        ]
        
        # Check for required ivector directory
        ivector_dir = os.path.join(model_path, 'ivector')
        if not os.path.exists(ivector_dir):
            # Debug: List what's actually in the model directory
            try:
                actual_contents = os.listdir(model_path)
                logger.warning(f"Missing ivector directory: {ivector_dir}")
                logger.warning(f"Actual contents of {model_path}: {actual_contents}")
            except Exception as e:
                logger.warning(f"Could not list model directory contents: {e}")
            return False
        
        # Check for required ivector files
        required_ivector_files = [
            'final.dubm',
            'final.ie', 
            'final.mat',
            'global_cmvn.stats'
        ]
        
        # Check main model files
        for file_name in required_files:
            file_path = os.path.join(model_path, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"Missing model file: {file_path}")
                return False
            
            # Check file is not empty
            if os.path.getsize(file_path) == 0:
                logger.warning(f"Empty model file: {file_path}")
                return False
        
        # Check ivector files  
        for file_name in required_ivector_files:
            file_path = os.path.join(ivector_dir, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"Missing ivector file: {file_path}")
                return False
        
        logger.info(f"Model validation passed for: {language}")
        return True
    
    @staticmethod
    async def ensure_model_with_feedback(language: str, websocket_manager=None, session_id=None) -> bool:
        # Ensure model exists with user feedback and retry logic
        if ModelManager.validate_model(language):
            return True
        
        # Try to download silently in background
        try:
            return await ModelManager._download_model_with_retry(language, None, None)
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return False
    
    @staticmethod
    async def ensure_model_silent(language: str) -> bool:
        # Backward compatibility method
        return await ModelManager.ensure_model_with_feedback(language, None, None)
    
    @staticmethod
    async def _download_model_with_retry(language: str, websocket_manager=None, session_id=None, max_retries=3) -> bool:
        # Enhanced download with retry logic and user feedback
        if language not in LANGUAGE_MODELS:
            logger.error(f"No model URL available for language: {language}")
            return False
        
        model_url = LANGUAGE_MODELS[language]
        model_path = ModelManager.get_model_path(language)
        
        for attempt in range(max_retries):
            try:
                # Silent download - no user messages
                
                zip_path = f"{model_path}.zip"
                
                # Create model directory with correct path
                base_models_dir = os.path.dirname(model_path)
                os.makedirs(base_models_dir, exist_ok=True)
                
                # Download model with timeout and retry logic
                try:
                    urllib.request.urlretrieve(model_url, zip_path)
                    logger.info(f"Downloaded model to: {zip_path}")
                except Exception as download_error:
                    raise Exception(f"Download failed: {download_error}")
                
                # Validate ZIP file
                if not zipfile.is_zipfile(zip_path):
                    raise Exception("Downloaded file is not a valid ZIP")
                
                # Silent extraction
                
                # Extract ZIP to temporary directory first
                temp_dir = f"{model_path}_temp"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    logger.info(f"Extracted to temporary directory: {temp_dir}")
                
                # Find the actual model files - look for am/final.mdl structure
                model_source_dir = None
                
                # Look for the directory containing am/final.mdl
                for root, dirs, files in os.walk(temp_dir):
                    # Check if this directory has an 'am' subdirectory with final.mdl
                    am_path = os.path.join(root, 'am', 'final.mdl')
                    if os.path.exists(am_path):
                        model_source_dir = root
                        logger.info(f"Found model files in: {model_source_dir}")
                        break
                
                if not model_source_dir:
                    raise Exception("Could not find model files in downloaded archive")
                
                # Remove existing model directory if it exists
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                
                # Move model files to final location
                shutil.move(model_source_dir, model_path)
                
                # Clean up
                shutil.rmtree(temp_dir)
                os.remove(zip_path)
                
                logger.info(f"Model extracted to: {model_path}")
                
                # Validate the extracted model
                if ModelManager.validate_model(language):
                    return True
                else:
                    raise Exception("Model validation failed after extraction")
                
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed for {language}: {e}")
                
                # Clean up on failure
                cleanup_paths = []
                if 'zip_path' in locals():
                    cleanup_paths.append(zip_path)
                if 'temp_dir' in locals():
                    cleanup_paths.append(f"{model_path}_temp")
                    
                for path in cleanup_paths:
                    if os.path.exists(path):
                        try:
                            if os.path.isdir(path):
                                shutil.rmtree(path)
                            else:
                                os.remove(path)
                        except Exception as cleanup_error:
                            logger.warning(f"Cleanup failed for {path}: {cleanup_error}")
                
                # If this is not the last attempt, wait before retrying
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)  # Progressive backoff
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed - silent failure
                    return False
        
        return False

    @staticmethod
    async def _download_model_silent(language: str) -> bool:
        # Wrapper for backward compatibility
        return await ModelManager._download_model_with_retry(language, None, None, 3)

# Simple state management
class SimpleState:
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, session_id: str):
        try:
            # Create AudioRecorder with data path
            from src.file_manager import FileManager
            recordings_dir = FileManager.get_data_path("recordings")
            output_file = os.path.join(
                recordings_dir, 
                f"recording_{int(time.time())}.wav"
            )
            recorder = AudioRecorder(output_file=output_file)
            
            self.sessions[session_id] = {
                "recorder": recorder,
                "recording": False,
                "paused": False,
                "progress": {"transcription": 0, "diarization": 0},
                "websocket": None,
                "start_time": None,
                "task": None
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

# WebSocket manager
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

# Reponsive HTML interface w/ file path notifications
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
            background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
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
            position: relative;
        }

        .logo {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #333;
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
        .status.downloading { background: #e1ecf4; color: #0c5460; }
        .status.retrying { background: #ffeaa7; color: #d68910; }
        .status.error { background: #f5c6cb; color: #721c24; }

        .file-notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 400px;
            z-index: 1000;
            transform: translateX(450px);
            transition: transform 0.3s ease;
            font-size: 0.9rem;
            line-height: 1.4;
        }

        .file-notification.show {
            transform: translateX(0);
        }

        .file-notification .close-btn {
            float: right;
            background: none;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
            margin-left: 10px;
        }

        .controls {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin: 4rem 0;
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
            margin: 3rem 0;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px dashed #dee2e6;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .waveform.active {
            background: #f8f9fa;
            border-color: #666;
            border-style: solid;
        }

        .waveform.downloading {
            background: linear-gradient(135deg, #e1ecf420, #0c546020);
            border-color: #0c5460;
            border-style: solid;
        }

        .waveform.retrying {
            background: linear-gradient(135deg, #ffeaa720, #d6891020);
            border-color: #d68910;
            border-style: solid;
        }

        .waveform-message {
            text-align: center;
            font-weight: 500;
            color: #666;
            padding: 0 1rem;
            line-height: 1.4;
        }

        .waveform-message.downloading {
            color: #0c5460;
        }

        .waveform-message.retrying {
            color: #d68910;
        }

        .retry-progress {
            position: absolute;
            bottom: 0;
            left: 0;
            height: 3px;
            background: #d68910;
            transition: width 0.3s ease;
            width: 0%;
        }

        .waveform-bars {
            display: flex;
            gap: 2px;
            align-items: end;
            height: 40px;
        }

        .bar {
            width: 3px;
            background: #666;
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
            background: #333;
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
            border-left: 4px solid #333;
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

            .file-notification {
                position: fixed;
                top: 10px;
                left: 10px;
                right: 10px;
                max-width: none;
                transform: translateY(-100px);
            }

            .file-notification.show {
                transform: translateY(0);
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
    </style>
</head>
<body>
    <div class="app">
        <div class="logo">TranscrevAI</div>
        <div class="tagline">Real-time Audio Transcription with Speaker Diarization</div>
        
        <div class="settings">
            <select id="language">
                <option value="en">English</option>
                <option value="pt">Portuguese</option>
                <option value="es">Spanish</option>
            </select>
        </div>

        <div class="controls">
            <button id="startBtn" class="btn btn-primary" disabled>Start</button>
            <button id="pauseBtn" class="btn btn-secondary" disabled>Pause</button>
            <button id="stopBtn" class="btn btn-danger" disabled>Stop</button>
        </div>

        <div id="waveform" class="waveform">
            <div id="waveform-content" class="waveform-message">Ready to record</div>
            <div id="retry-progress" class="retry-progress"></div>
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

        <div id="results" class="results"></div>
    </div>

    <!-- File notification popup -->
    <div id="fileNotification" class="file-notification">
        <button class="close-btn" onclick="this.parentElement.classList.remove('show')">&times;</button>
        <div id="fileContent"></div>
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
                this.transcriptionFill = document.getElementById('transcription-fill');
                this.diarizationFill = document.getElementById('diarization-fill');
                this.transcriptionPercent = document.getElementById('transcription-percent');
                this.diarizationPercent = document.getElementById('diarization-percent');
                this.fileNotification = document.getElementById('fileNotification');
                this.fileContent = document.getElementById('fileContent');
                this.retryProgress = document.getElementById('retry-progress');
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

            showFileNotification(audioPath, srtPath) {
                let content = `<strong>Files saved successfully!</strong><br><br>`;
                content += `<strong>Audio:</strong><br>${audioPath}<br><br>`;
                if (srtPath) {
                    content += `<strong>Subtitles:</strong><br>${srtPath}`;
                }
                
                this.fileContent.innerHTML = content;
                this.fileNotification.classList.add('show');
                
                // Auto-hide after 5 seconds
                setTimeout(() => {
                    this.fileNotification.classList.remove('show');
                }, 5000);
            }

            handleMessage(message) {
                switch (message.type) {
                    case 'model_download':
                        this.handleModelDownload(message.message);
                        break;

                    case 'recording_started':
                        this.isRecording = true;
                        this.setStatus('Recording...', 'recording');
                        this.startBtn.disabled = true;
                        this.pauseBtn.disabled = false;
                        this.stopBtn.disabled = false;
                        this.waveformContent.textContent = 'Recording...';
                        this.showWaveform(true);
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

                        // Show file paths notification
                        this.showFileNotification(message.audio_file, message.srt_file);
                        break;

                    case 'error':
                        this.setStatus(`Error: ${message.message}`, 'error');
                        this.resetControls();
                        this.resetWaveform();
                        break;
                }
            }

            handleModelDownload(message) {
                // Determine the type of download status based on message content
                if (message.includes('Retrying') || message.includes('attempt')) {
                    this.setStatus(message, 'retrying');
                    this.showRetryProgress(message);
                } else if (message.includes('Downloading') || message.includes('Extracting')) {
                    this.setStatus(message, 'downloading');
                    this.showDownloadProgress(message);
                } else {
                    this.setStatus(message, 'processing');
                    this.showWaveformMessage(message, 'downloading');
                }
            }

            showRetryProgress(message) {
                this.showWaveform(false);
                this.waveformEl.classList.remove('downloading');
                this.waveformEl.classList.add('retrying');
                this.waveformContent.className = 'waveform-message retrying';
                this.waveformContent.textContent = message;
                
                // Show retry countdown if available
                const retryMatch = message.match(/retrying in (\\d+) seconds/i);
                if (retryMatch) {
                    const seconds = parseInt(retryMatch[1]);
                    this.startRetryCountdown(seconds);
                }
            }

            showDownloadProgress(message) {
                this.showWaveform(false);
                this.waveformEl.classList.remove('retrying');
                this.waveformEl.classList.add('downloading');
                this.waveformContent.className = 'waveform-message downloading';
                this.waveformContent.textContent = message;
                this.retryProgress.style.width = '0%';
            }

            showWaveformMessage(message, type = '') {
                this.showWaveform(false);
                this.waveformEl.classList.remove('downloading', 'retrying');
                if (type) {
                    this.waveformEl.classList.add(type);
                    this.waveformContent.className = `waveform-message ${type}`;
                } else {
                    this.waveformContent.className = 'waveform-message';
                }
                this.waveformContent.textContent = message;
                this.retryProgress.style.width = '0%';
            }

            startRetryCountdown(seconds) {
                let remaining = seconds;
                this.retryProgress.style.width = '100%';
                
                const countdown = setInterval(() => {
                    remaining--;
                    const progress = (remaining / seconds) * 100;
                    this.retryProgress.style.width = `${Math.max(0, progress)}%`;
                    
                    if (remaining <= 0) {
                        clearInterval(countdown);
                        this.retryProgress.style.width = '0%';
                    }
                }, 1000);
            }

            resetWaveform() {
                this.showWaveform(false);
                this.waveformEl.classList.remove('downloading', 'retrying');
                this.waveformContent.className = 'waveform-message';
                this.waveformContent.textContent = 'Ready to record';
                this.retryProgress.style.width = '0%';
            }

            setStatus(text, type) {
                this.statusEl.textContent = text;
                this.statusEl.className = `status ${type}`;
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
                this.resetWaveform();
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

# WebSocket handler w/ model management
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

# Enhanced message handler w/ model management
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
                language = message_data.get("language", "en")

                # Start recording immediately (no waiting for model)
                await recorder.start_recording()

                app_state.update_session(session_id, {
                    "recording": True,
                    "start_time": time.time(),
                    "language": language
                })

                await websocket_manager.send_message(session_id, {
                    "type": "recording_started",
                    "message": "Recording started"
                })

                # Start model download silently in background (non-blocking)
                model_task = asyncio.create_task(
                    ModelManager.ensure_model_silent(language)
                )
                
                # Start audio monitoring and processing
                asyncio.create_task(monitor_audio(session_id))
                task = asyncio.create_task(process_audio(session_id, language))
                app_state.update_session(session_id, {
                    "task": task,
                    "model_task": model_task
                })

            except Exception as e:
                logger.error(f"Start recording error: {e}")
                await websocket_manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Failed to start recording. Please check your microphone permissions and try again."
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

# Audio monitoring
async def monitor_audio(session_id: str):
    try:
        session = app_state.get_session(session_id)
        while session and session["recording"]:
            if not session["paused"]:
                # Simulate audio level - replace with real audio level detection
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

# Enhanced processing pipeline with proper SRT generation
async def process_audio(session_id: str, language: str = "en"):
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
        
        # Enhanced file validation
        if not os.path.exists(audio_file):
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": "Audio file not found. Recording may have failed."
            })
            return
            
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": "No audio was recorded. Please check your microphone."
            })
            return
        
        # Check for minimum meaningful audio size (at least 1KB)
        if file_size < 1024:
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": "Audio recording too short. Please record for at least 1 second."
            })
            return
        
        logger.info(f"Processing: {audio_file} (size: {file_size} bytes)")
        
        # Wait for background model download to complete silently
        session = app_state.get_session(session_id)
        if session and session.get("model_task"):
            try:
                # Wait for model download to finish (background task)
                await session["model_task"]
            except Exception as e:
                logger.error(f"Background model download failed: {e}")
                # Try to ensure model is available as fallback
                await ModelManager.ensure_model_silent(language)
        
        # Get correct model path
        model_path = ModelManager.get_model_path(language)
        
        # Validate model before transcription
        if not ModelManager.validate_model(language):
            # Try one more time to get the model silently
            await ModelManager.ensure_model_silent(language)
            if not ModelManager.validate_model(language):
                await websocket_manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Model for {language} is not available. Please check your internet connection and try again."
                })
                return
        
        # Transcription with enhanced error handling
        transcription_data = []
        try:
            async for progress, data in transcribe_audio_with_progress(
                audio_file, model_path, language, 16000
            ):
                session = app_state.get_session(session_id)
                if not session:
                    break
                
                await websocket_manager.send_message(session_id, {
                    "type": "progress",
                    "transcription": progress,
                    "diarization": 0
                })
                
                if data:
                    transcription_data.extend(data)
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            await websocket_manager.send_message(session_id, {
                "type": "error", 
                "message": f"Transcription failed: {str(e)}"
            })
            return
        
        # Speaker Diarization with fixed progress updates 
        diarization_segments = []
        unique_speakers = 0
        
        try:
            diarizer = SpeakerDiarization()
            # Use the correct method for diarization; assuming 'diarize' is the correct method
            segments = diarizer.diarize(audio_file)
            diarization_segments = segments if segments else []
            unique_speakers = len(set(seg.get('speaker', 'Unknown') for seg in diarization_segments)) if diarization_segments else 0

            # Optionally, send 100% progress if needed
            await websocket_manager.send_message(session_id, {
                "type": "progress",
                "transcription": 100,
                "diarization": 100
            })

        except Exception as e:
            logger.error(f"Diarization error: {e}")
            unique_speakers = 0
        
        # Generate SRT subtitle file
        srt_file = None
        try:
            srt_file = await generate_srt(transcription_data, diarization_segments)
            if srt_file:
                logger.info(f"SRT generated successfully: {srt_file}")
            else:
                logger.warning("SRT generation returned None")
        except Exception as e:
            logger.error(f"SRT generation failed: {e}")
        
        # Send results with file paths
        await websocket_manager.send_message(session_id, {
            "type": "processing_complete",
            "transcription_data": transcription_data,
            "diarization_segments": diarization_segments,
            "speakers_detected": unique_speakers,
            "srt_file": srt_file,
            "audio_file": audio_file,  # Include audio file path
            "duration": session.get("duration", 0) if session else 0
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Processing error: {error_msg}")
        await websocket_manager.send_message(session_id, {
            "type": "error",
            "message": "Processing failed. Please try again."
        })

# Production startup
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False  # Disable for production
    )