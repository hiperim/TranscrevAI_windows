import asyncio
import logging
import os
import time
import random
import tempfile
import urllib.request
import zipfile
import shutil
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from src.audio_processing import AudioRecorder
from src.transcription import transcribe_audio_with_progress
from src.speaker_diarization import SpeakerDiarization
from src.subtitle_generator import generate_srt
from src.file_manager import FileManager
from config.app_config import WHISPER_MODEL_DIR, WHISPER_MODELS, WHISPER_CONFIG
from src.concurrent_engine import concurrent_processor
from src.logging_setup import setup_app_logging

logger = setup_app_logging()
if logger is None:
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

class WhisperModelManager:
    """Whisper model management with automatic downloads"""
    
    @staticmethod
    def get_model_name(language: str) -> str:
        """Get Whisper model name for language"""
        return WHISPER_MODELS.get(language, "small")
    
    @staticmethod
    async def ensure_whisper_model(language: str, websocket_manager=None, session_id=None) -> bool:
        """Ensure Whisper model is available, download if necessary"""
        try:
            import whisper
            
            model_name = WhisperModelManager.get_model_name(language)
            
            # Send download start notification
            if websocket_manager and session_id:
                await websocket_manager.send_message(session_id, {
                    "type": "model_download_start",
                    "message": f"Downloading {model_name} model for {language}...",
                    "language": language,
                    "model": model_name
                })
            
            # Check if model is already cached
            try:
                # Try to load model to check if it's available
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    whisper.load_model,
                    model_name,
                    "cpu",  # Use CPU for validation
                    str(WHISPER_MODEL_DIR)
                )
                
                logger.info(f"Whisper model '{model_name}' already available for {language}")
                
                # Send download complete notification
                if websocket_manager and session_id:
                    await websocket_manager.send_message(session_id, {
                        "type": "model_download_complete",
                        "message": f"Model {model_name} ready",
                        "language": language,
                        "model": model_name
                    })
                
                return True
                
            except Exception as e:
                logger.info(f"Model '{model_name}' not cached, downloading: {e}")
                
                # Download model
                await loop.run_in_executor(
                    None,
                    whisper.load_model,
                    model_name,
                    "cpu",
                    str(WHISPER_MODEL_DIR)
                )
                
                logger.info(f"Whisper model '{model_name}' downloaded successfully for {language}")
                
                # Send download complete notification
                if websocket_manager and session_id:
                    await websocket_manager.send_message(session_id, {
                        "type": "model_download_complete",
                        "message": f"Model {model_name} downloaded successfully",
                        "language": language,
                        "model": model_name
                    })
                
                return True
                
        except Exception as e:
            logger.error(f"Whisper model setup failed for {language}: {e}")
            
            # Send download error notification
            if websocket_manager and session_id:
                await websocket_manager.send_message(session_id, {
                    "type": "model_download_error",
                    "message": f"Failed to download model for {language}: {str(e)}",
                    "language": language
                })
            
            return False
    
    @staticmethod
    async def ensure_model_silent(language: str) -> bool:
        """Silent model ensuring for backward compatibility"""
        return await WhisperModelManager.ensure_whisper_model(language)

# Simple state management
class SimpleState:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, session_id: str):
        try:
            # Create session without AudioRecorder initially
            # AudioRecorder will be created when recording starts with the correct format
            self.sessions[session_id] = {
                "recorder": None,
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
    
    def create_recorder_for_session(self, session_id: str, format_type: str = "wav"):
        try:
            recordings_dir = FileManager.get_data_path("recordings")
            extension = "wav" if format_type == "wav" else "mp4"
            output_file = os.path.join(
                recordings_dir, 
                f"recording_{int(time.time())}.{extension}"
            )
            recorder = AudioRecorder(output_file=output_file)
            
            if session_id in self.sessions:
                self.sessions[session_id]["recorder"] = recorder
                logger.info(f"AudioRecorder created for session {session_id} with format {format_type}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to create recorder for session: {e}")
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

# Responsive HTML interface w/ file path notifications - FIXED VERSION
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
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5rem;
            font-weight: 700;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1rem;
        }

        .controls {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-bottom: 30px;
        }

        .control-row {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .select {
            padding: 12px 16px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 1rem;
            background: white;
            margin: 0 0.5rem;
        }

        .button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            min-width: 100px;
        }

        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .record-btn {
            background: linear-gradient(45deg, #66bb6a, #43a047);
            color: white;
        }

        .pause-btn {
            background: linear-gradient(45deg, #ffa726, #fb8c00);
            color: white;
        }

        .stop-btn {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
        }

        .button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .status {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: 500;
        }

        .waveform {
            height: 80px;
            background: #f1f3f4;
            border-radius: 8px;
            margin: 20px 0;
            position: relative;
            overflow: hidden;
        }

        .waveform-content {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
        }

        .progress {
            margin: 20px 0;
        }

        .progress-item {
            margin-bottom: 15px;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-weight: 500;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e1e5e9;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(45deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
        }

        .results {
            max-height: 300px;
            overflow-y: auto;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s ease;
        }

        .results.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4caf50;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
            z-index: 1000;
            max-width: 400px;
        }

        .notification.show {
            opacity: 1;
            transform: translateX(0);
        }

        .notification.error {
            background: #f44336;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .container {
                padding: 20px;
                margin: 10px;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .control-row {
                flex-direction: column;
            }
            
            .select, .button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>TranscrevAI</h1>
        <p class="subtitle">Real-time Audio Transcription with Speaker Diarization</p>
        
        <div class="controls">
            <div class="control-row">
                <select id="language" class="select">
                    <option value="" disabled selected>[Select Language]</option>
                    <option value="en">English</option>
                    <option value="pt">Portuguese</option>
                    <option value="es">Spanish</option>
                </select>
                
                <select id="format" class="select">
                    <option value="wav">.WAV</option>
                    <option value="mp4">.MP4</option>
                </select>
            </div>
            
            <div class="control-row">
                <button id="recordBtn" class="button record-btn">Record</button>
                <button id="pauseBtn" class="button pause-btn" disabled>Pause</button>
                <button id="stopBtn" class="button stop-btn" disabled>Stop</button>
            </div>
        </div>
        
        <div id="status" class="status">Ready to record</div>
        
        <div id="waveform" class="waveform">
            <div id="waveform-content" class="waveform-content">Audio visualization will appear here</div>
        </div>
        
        <div id="progress" class="progress">
            <div class="progress-item">
                <div class="progress-label">
                    <span>Transcription</span>
                    <span id="transcription-percent">0%</span>
                </div>
                <div class="progress-bar">
                    <div id="transcription-progress" class="progress-fill"></div>
                </div>
            </div>
            
            <div class="progress-item">
                <div class="progress-label">
                    <span>Diarization</span>
                    <span id="diarization-percent">0%</span>
                </div>
                <div class="progress-bar">
                    <div id="diarization-progress" class="progress-fill"></div>
                </div>
            </div>
        </div>
        
        <div id="results" class="results">
            <h3>Results will appear here</h3>
        </div>
    </div>
    
    <div id="notification" class="notification"></div>

    <script>
        class TranscrevAI {
            constructor() {
                this.ws = null;
                this.sessionId = this.generateSessionId();
                this.isRecording = false;
                this.isPaused = false;
                
                // Get DOM elements
                this.recordBtn = document.getElementById('recordBtn');
                this.pauseBtn = document.getElementById('pauseBtn');
                this.stopBtn = document.getElementById('stopBtn');
                this.languageEl = document.getElementById('language');
                
                // CRITICAL FIX: Safe format element initialization
                this.formatEl = document.getElementById('format');
                if (!this.formatEl) {
                    console.warn('Format selector not found, using default WAV format');
                    // Create a mock element to prevent further errors
                    this.formatEl = { value: 'wav' };
                }
                
                this.waveformEl = document.getElementById('waveform');
                this.waveformContent = document.getElementById('waveform-content');
                this.progressEl = document.getElementById('progress');
                this.transcriptionProgress = document.getElementById('transcription-progress');
                this.diarizationProgress = document.getElementById('diarization-progress');
                this.transcriptionPercent = document.getElementById('transcription-percent');
                this.diarizationPercent = document.getElementById('diarization-percent');
                this.statusEl = document.getElementById('status');
                this.resultsEl = document.getElementById('results');
                this.notificationEl = document.getElementById('notification');
                
                this.setupEventListeners();
                this.connect();
            }
            
            generateSessionId() {
                return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            }
            
            setupEventListeners() {
                this.recordBtn.addEventListener('click', () => this.startRecording());
                this.pauseBtn.addEventListener('click', () => this.togglePause());
                this.stopBtn.addEventListener('click', () => this.stopRecording());
            }
            
            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('Connected to server');
                    this.updateStatus('Connected - Ready to record');
                };
                
                this.ws.onclose = () => {
                    console.log('Disconnected from server');
                    this.updateStatus('Disconnected - Please refresh the page');
                    setTimeout(() => this.connect(), 3000);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.showError('Connection error. Please refresh the page.');
                };
                
                this.ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                };
            }
            
            handleMessage(message) {
                switch (message.type) {
                    case 'recording_started':
                        this.isRecording = true;
                        this.updateButtons();
                        this.updateStatus('Recording...');
                        break;
                        
                    case 'recording_stopped':
                        this.isRecording = false;
                        this.isPaused = false;
                        this.updateButtons();
                        this.updateStatus('Processing audio...');
                        break;
                        
                    case 'recording_paused':
                        this.isPaused = true;
                        this.updateButtons();
                        this.updateStatus('Recording paused');
                        break;
                        
                    case 'recording_resumed':
                        this.isPaused = false;
                        this.updateButtons();
                        this.updateStatus('Recording...');
                        break;
                        
                    case 'audio_level':
                        this.updateWaveform(message.level);
                        break;
                        
                    case 'progress':
                        this.updateProgress(message.transcription || 0, message.diarization || 0);
                        break;
                        
                    case 'model_download_start':
                        this.updateStatus(`Downloading ${message.model} model for ${message.language}...`);
                        break;
                        
                    case 'model_download_complete':
                        this.updateStatus('Model ready - Processing audio...');
                        break;
                        
                    case 'model_download_error':
                        this.showError(`Model download failed: ${message.message}`);
                        this.resetState();
                        break;
                        
                    case 'processing_complete':
                        this.handleResults(message);
                        break;
                        
                    case 'error':
                        this.showError(message.message);
                        this.resetState();
                        break;
                        
                    default:
                        console.log('Unknown message type:', message.type);
                }
            }
            
            updateButtons() {
                this.recordBtn.disabled = this.isRecording;
                this.pauseBtn.disabled = !this.isRecording;
                this.stopBtn.disabled = !this.isRecording;
                
                if (this.isPaused) {
                    this.pauseBtn.textContent = 'Resume';
                } else {
                    this.pauseBtn.textContent = 'Pause';
                }
            }
            
            updateStatus(text) {
                this.statusEl.textContent = text;
            }
            
            updateWaveform(level) {
                if (this.isRecording && !this.isPaused) {
                    const intensity = Math.floor(level * 255);
                    const color = `rgb(${intensity}, ${Math.floor(intensity * 0.7)}, ${Math.floor(intensity * 0.3)})`;
                    this.waveformContent.style.background = `linear-gradient(90deg, ${color} ${level * 100}%, #f1f3f4 ${level * 100}%)`;
                    this.waveformContent.textContent = `Recording... (Level: ${Math.floor(level * 100)}%)`;
                }
            }
            
            updateProgress(transcription, diarization) {
                this.transcriptionProgress.style.width = `${transcription}%`;
                this.diarizationProgress.style.width = `${diarization}%`;
                this.transcriptionPercent.textContent = `${transcription}%`;
                this.diarizationPercent.textContent = `${diarization}%`;
            }
            
            handleResults(data) {
                this.updateStatus('Processing complete!');
                this.resetState();
                
                let resultsHTML = '<h3>Transcription Results</h3>';
                
                if (data.transcription_data && data.transcription_data.length > 0) {
                    resultsHTML += '<div style="margin: 15px 0;"><strong>Transcription:</strong></div>';
                    
                    // ENHANCED: Group transcription text by speaker for better readability
                    const groupedBySpeaker = {};
                    
                    // Group segments by speaker with temporal proximity
                    data.transcription_data.forEach(item => {
                        const speakerName = (item.speaker || 'Speaker_1').replace('_', ' ');
                        const text = item.text || item.content || '';
                        
                        if (text.trim()) {
                            if (!groupedBySpeaker[speakerName]) {
                                groupedBySpeaker[speakerName] = {
                                    texts: [],
                                    firstTime: item.start || 0,
                                    lastTime: item.end || 0
                                };
                            }
                            
                            groupedBySpeaker[speakerName].texts.push(text.trim());
                            
                            // Update timing info
                            if (item.start !== undefined && item.start < groupedBySpeaker[speakerName].firstTime) {
                                groupedBySpeaker[speakerName].firstTime = item.start;
                            }
                            if (item.end !== undefined && item.end > groupedBySpeaker[speakerName].lastTime) {
                                groupedBySpeaker[speakerName].lastTime = item.end;
                            }
                        }
                    });
                    
                    // Display grouped results
                    Object.keys(groupedBySpeaker).sort().forEach(speakerName => {
                        const speakerData = groupedBySpeaker[speakerName];
                        const combinedText = speakerData.texts.join(' ').trim();
                        
                        if (combinedText) {
                            const duration = speakerData.lastTime - speakerData.firstTime;
                            const timeInfo = duration > 0 ? ` (${speakerData.firstTime.toFixed(1)}s - ${speakerData.lastTime.toFixed(1)}s)` : '';
                            
                            resultsHTML += `<div style="margin: 10px 0; padding: 15px; background: white; border-radius: 8px; border-left: 4px solid #007bff;">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <strong style="color: #007bff; font-size: 1.1em;">${speakerName}:</strong>
                                    <span style="font-size: 0.9em; color: #666;">${timeInfo}</span>
                                </div>
                                <div style="margin-top: 8px; line-height: 1.4; font-size: 1em;">
                                    ${combinedText}
                                </div>
                            </div>`;
                        }
                    });
                    
                    // Fallback: if no grouped text found, show original format
                    if (Object.keys(groupedBySpeaker).length === 0) {
                        data.transcription_data.forEach(item => {
                            const text = item.text || item.content || 'No text';
                            if (text.trim() !== 'No text' && text.trim()) {
                                resultsHTML += `<div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                                    <strong>${(item.speaker || 'Speaker_1').replace('_', ' ')}:</strong> ${text}
                                </div>`;
                            }
                        });
                    }
                } else {
                    resultsHTML += '<div>No transcription data available.</div>';
                }
                
                if (data.speakers_detected > 0) {
                    resultsHTML += `<div style="margin: 15px 0;"><strong>Speakers detected:</strong> ${data.speakers_detected}</div>`;
                }
                
                if (data.duration) {
                    resultsHTML += `<div><strong>Duration:</strong> ${Math.round(data.duration)} seconds</div>`;
                }
                
                this.resultsEl.innerHTML = resultsHTML;
                this.resultsEl.classList.add('visible');
                
                // Show file notification if paths are available
                if (data.audio_file || data.srt_file) {
                    this.showFileNotification(data.audio_file, data.srt_file);
                }
            }

            showFileNotification(audioPath, srtPath) {
                let content = `Files saved successfully!\n`;
                
                // Detect format from file extension
                const formatType = audioPath.toLowerCase().endsWith('.mp4') ? 'MP4 Video' : 'WAV Audio';
                content += `${formatType}: ${audioPath}\n`;
                
                if (srtPath) {
                    content += `Subtitles: ${srtPath}`;
                }
                
                this.showNotification(content, 'success');
            }
            
            showNotification(message, type = 'success') {
                this.notificationEl.textContent = message;
                this.notificationEl.className = `notification ${type}`;
                this.notificationEl.classList.add('show');
                
                setTimeout(() => {
                    this.notificationEl.classList.remove('show');
                }, 5000);
            }
            
            showError(message) {
                this.showNotification(message, 'error');
                console.error('Error:', message);
            }
            
            resetState() {
                this.isRecording = false;
                this.isPaused = false;
                this.updateButtons();
                this.waveformContent.style.background = '#f1f3f4';
                this.waveformContent.textContent = 'Audio visualization will appear here';
                this.updateProgress(0, 0);
            }
            
            // CRITICAL FIX: Safe format access with fallback
            startRecording() {
                this.resultsEl.classList.remove('visible');
                
                // Validate language selection
                if (!this.languageEl.value) {
                    this.showError('Please select a language before recording');
                    return;
                }
                
                // Safe format access with fallback
                const formatValue = this.formatEl && this.formatEl.value ? this.formatEl.value : 'wav';
                
                this.send('start_recording', { 
                    language: this.languageEl.value,
                    format: formatValue
                });
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
            
            // CRITICAL FIX: Enhanced error handling for WebSocket send
            send(type, data = {}) {
                try {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({
                            type: type,
                            data: data
                        }));
                    } else {
                        throw new Error('WebSocket is not connected');
                    }
                } catch (error) {
                    console.error('WebSocket send error:', error);
                    this.showError('Connection error. Please refresh the page.');
                    this.resetState();
                }
            }
        }
        
        // Initialize the application
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
    
    if message_type == "start_recording":
        if not session.get("recording", False):
            try:
                language = message_data.get("language", "en")
                format_type = message_data.get("format", "wav")

                # Create recorder with correct format
                if not app_state.create_recorder_for_session(session_id, format_type):
                    await websocket_manager.send_message(session_id, {
                        "type": "error",
                        "message": "Failed to create recorder"
                    })
                    return
                
                # Get the newly created recorder
                session = app_state.get_session(session_id)
                if not session or session.get("recorder") is None:
                    await websocket_manager.send_message(session_id, {
                        "type": "error",
                        "message": "Session or recorder not found"
                    })
                    return
                recorder = session["recorder"]

                # Start recording immediately (no waiting for model)
                await recorder.start_recording()

                app_state.update_session(session_id, {
                    "recording": True,
                    "start_time": time.time(),
                    "language": language,
                    "format": format_type
                })

                await websocket_manager.send_message(session_id, {
                    "type": "recording_started",
                    "message": "Recording started"
                })

                # Start Whisper model download with user feedback
                model_task = asyncio.create_task(
                    WhisperModelManager.ensure_whisper_model(language, websocket_manager, session_id)
                )
                
                # Start audio monitoring and concurrent processing
                asyncio.create_task(monitor_audio(session_id))
                task = asyncio.create_task(process_audio_concurrent(session_id, language, format_type))
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
        if session and session.get("recording") and session.get("recorder"):
            try:
                recorder = session["recorder"]
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
        if session and session.get("recording") and not session.get("paused") and session.get("recorder"):
            recorder = session["recorder"]
            recorder.pause_recording()
            app_state.update_session(session_id, {"paused": True})
            await websocket_manager.send_message(session_id, {
                "type": "recording_paused"
            })
    
    elif message_type == "resume_recording":
        if session and session.get("recording") and session.get("paused") and session.get("recorder"):
            recorder = session["recorder"]
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
                level = random.uniform(0.1, 1.0) if random.random() > 0.3 else 0.0
                
                await websocket_manager.send_message(session_id, {
                    "type": "audio_level",
                    "level": level
                })
            
            await asyncio.sleep(0.1)
            session = app_state.get_session(session_id)
    except Exception as e:
        logger.error(f"Audio monitoring error: {e}")

# Enhanced concurrent processing pipeline
async def process_audio_concurrent(session_id: str, language: str = "en", _format_type: str = "wav"):
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
        
        audio_file = session.get("recorder").output_file
        
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
        
        # Convert MP4 to WAV for transcription if needed
        wav_file_for_processing = audio_file
        if audio_file.endswith('.mp4'):
            try:
                # Create temporary WAV file for processing
                temp_dir = FileManager.get_data_path("temp")
                FileManager.ensure_directory_exists(temp_dir)
                wav_file_for_processing = os.path.join(temp_dir, f"temp_for_transcription_{int(time.time())}.wav")
                
                # Convert MP4 to WAV using FFmpeg
                ffmpeg_args = [
                    "ffmpeg", "-y",
                    "-i", audio_file,
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    wav_file_for_processing
                ]
                
                logger.info(f"Converting MP4 to WAV for transcription: {wav_file_for_processing}")
                
                process = await asyncio.create_subprocess_exec(
                    *ffmpeg_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
                
                if process.returncode != 0:
                    raise Exception(f"FFmpeg conversion failed: {stderr.decode()}")
                
                # Verify conversion was successful
                if not os.path.exists(wav_file_for_processing) or os.path.getsize(wav_file_for_processing) == 0:
                    raise Exception("WAV conversion produced empty file")
                
                logger.info(f"MP4 to WAV conversion successful: {wav_file_for_processing}")
                
            except Exception as e:
                logger.error(f"MP4 to WAV conversion failed: {e}")
                await websocket_manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Failed to convert MP4 for transcription: {str(e)}"
                })
                return
        
        # Wait for background model download to complete
        session = app_state.get_session(session_id)
        if session and session.get("model_task"):
            try:
                model_ready = await session.get("model_task")
                if not model_ready:
                    await websocket_manager.send_message(session_id, {
                        "type": "error",
                        "message": f"Failed to download Whisper model for {language}"
                    })
                    return
            except Exception as e:
                logger.error(f"Model download failed: {e}")
                await websocket_manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Model download failed: {str(e)}"
                })
                return
        
        # Use concurrent processing engine
        try:
            result = await concurrent_processor.process_audio_concurrent(
                session_id, wav_file_for_processing, language, websocket_manager
            )
            
            transcription_data = result.get("transcription_data", [])
            diarization_segments = result.get("diarization_segments", [])
            unique_speakers = result.get("speakers_detected", 0)
            
        except Exception as e:
            logger.error(f"Concurrent processing error: {e}")
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": f"Processing failed: {str(e)}"
            })
            return
        
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
        
        # Clean up temporary WAV file if it was created
        if wav_file_for_processing != audio_file and os.path.exists(wav_file_for_processing):
            try:
                os.remove(wav_file_for_processing)
                logger.info(f"Cleaned up temporary WAV file: {wav_file_for_processing}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary WAV file: {cleanup_error}")
        
        # Send results with file paths
        await websocket_manager.send_message(session_id, {
            "type": "processing_complete",
            "transcription_data": transcription_data,
            "diarization_segments": diarization_segments,
            "speakers_detected": unique_speakers,
            "srt_file": srt_file,
            "audio_file": audio_file,  # Include audio file path (original MP4)
            "duration": session.get("duration", 0) if session else 0
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Processing error: {error_msg}")
        
        # Clean up temporary WAV file if it was created
        try:
            if (
                "wav_file_for_processing" in locals()
                and "audio_file" in locals()
                and wav_file_for_processing is not None
                and audio_file is not None
                and isinstance(wav_file_for_processing, str)
                and isinstance(audio_file, str)
                and wav_file_for_processing != audio_file
                and os.path.exists(wav_file_for_processing)
            ):
                os.remove(wav_file_for_processing)
                logger.info(f"Cleaned up temporary WAV file after error: {wav_file_for_processing}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary WAV file after error: {cleanup_error}")
        
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