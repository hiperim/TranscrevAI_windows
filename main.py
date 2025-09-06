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

# Lazy import functions for heavy dependencies
def get_audio_recorder():
    """Lazy import AudioRecorder"""
    from src.audio_processing import AudioRecorder
    return AudioRecorder

def get_transcription_func():
    """Lazy import transcription function"""
    from src.transcription import transcribe_audio_with_progress
    return transcribe_audio_with_progress

def get_speaker_diarization():
    """Lazy import SpeakerDiarization"""
    from src.speaker_diarization import SpeakerDiarization
    return SpeakerDiarization

def get_model_config():
    """Lazy import model configuration"""
    from config.app_config import WHISPER_MODEL_DIR, WHISPER_MODELS, WHISPER_CONFIG
    return WHISPER_MODEL_DIR, WHISPER_MODELS, WHISPER_CONFIG

def get_concurrent_processor():
    """Lazy import concurrent processor"""
    from src.concurrent_engine import concurrent_processor
    return concurrent_processor

# Keep essential imports
from src.subtitle_generator import generate_srt
from src.file_manager import FileManager
from src.logging_setup import setup_app_logging

logger = setup_app_logging()

# FastAPI setup
app = FastAPI(
    title="TranscrevAI",
    description="Real-time Audio Transcription with AI",
    version="1.0.0"
)

class WhisperModelManager:
    """Whisper model management with automatic downloads and caching"""
    
    _model_cache = {}
    _loading_locks = {}
    _cache_ttl = 86400  # 24 hour cache TTL (24 * 60 * 60)
    _cache_timestamps = {}
    
    @staticmethod
    def _get_whisper():
        """Lazy import of whisper module"""
        global _whisper_module
        if '_whisper_module' not in globals():
            import whisper
            globals()['_whisper_module'] = whisper
        return globals()['_whisper_module']
    
    @staticmethod
    def get_model_name(language: str) -> str:
        """Get Whisper model name for language"""
        _, WHISPER_MODELS, _ = get_model_config()
        return WHISPER_MODELS.get(language, "small")
    
    @classmethod
    async def get_cached_model(cls, language: str):
        """Get cached model if available and not expired"""
        model_name = cls.get_model_name(language)
        
        if model_name in cls._model_cache:
            # Check if cache is still valid
            cache_time = cls._cache_timestamps.get(model_name, 0)
            if time.time() - cache_time < cls._cache_ttl:
                logger.info(f"Using cached Whisper model: {model_name}")
                return cls._model_cache[model_name]
            else:
                # Cache expired, remove it
                del cls._model_cache[model_name]
                del cls._cache_timestamps[model_name]
        
        return None
    
    @classmethod
    async def cache_model(cls, language: str, model):
        """Cache model with timestamp"""
        model_name = cls.get_model_name(language)
        cls._model_cache[model_name] = model
        cls._cache_timestamps[model_name] = time.time()
        logger.info(f"Cached Whisper model: {model_name}")
    
    @classmethod  
    async def ensure_whisper_model(cls, language: str, websocket_manager=None, session_id=None) -> bool:
        """Ensure Whisper model is available, download if necessary"""
        try:
            whisper = cls._get_whisper()
            
            model_name = cls.get_model_name(language)
            
            # Check if model is already cached
            cached_model = await cls.get_cached_model(language)
            if cached_model:
                # Send completion notification for cached model
                if websocket_manager and session_id:
                    await websocket_manager.send_message(session_id, {
                        "type": "model_download_complete",
                        "message": f"Model {model_name} ready (cached)",
                        "language": language,
                        "model": model_name
                    })
                return True
            
            # Use locking to prevent concurrent downloads of same model
            if model_name not in cls._loading_locks:
                cls._loading_locks[model_name] = asyncio.Lock()
            
            async with cls._loading_locks[model_name]:
                # Double-check cache after acquiring lock
                cached_model = await cls.get_cached_model(language)
                if cached_model:
                    return True
            
                # Use running event loop for executor tasks
                loop = asyncio.get_running_loop()
                
                # Send download start notification
                if websocket_manager and session_id:
                    await websocket_manager.send_message(session_id, {
                        "type": "model_download_start",
                        "message": f"Loading {model_name} model for {language}...",
                        "language": language,
                        "model": model_name
                    })
            
                # Try to load and cache model  
                try:
                    WHISPER_MODEL_DIR, _, _ = get_model_config()
                    model = await loop.run_in_executor(
                        None,
                        whisper.load_model,
                        model_name,
                        "cpu",  # Use CPU for validation
                        str(WHISPER_MODEL_DIR)
                    )
                    
                    # Cache the loaded model
                    await cls.cache_model(language, model)
                    
                    logger.info(f"Whisper model '{model_name}' loaded and cached for {language}")
                    
                    # Send completion notification
                    if websocket_manager and session_id:
                        await websocket_manager.send_message(session_id, {
                            "type": "model_download_complete",
                            "message": f"Model {model_name} ready",
                            "language": language,
                            "model": model_name
                        })
                    
                    return True
                    
                except Exception as e:
                    logger.info(f"Model '{model_name}' not available, downloading: {e}")
                    
                    # Download and cache model
                    WHISPER_MODEL_DIR, _, _ = get_model_config()
                    model = await loop.run_in_executor(
                        None,
                        whisper.load_model,
                        model_name,
                        "cpu",
                        str(WHISPER_MODEL_DIR)
                    )
                    
                    # Cache the downloaded model
                    await cls.cache_model(language, model)
                    
                    logger.info(f"Whisper model '{model_name}' downloaded and cached for {language}")
                    
                    # Send completion notification
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
    
    def create_recorder_for_session(self, session_id: str, format_type: str = "wav", 
                                    websocket_manager=None):
        try:
            recordings_dir = FileManager.get_data_path("recordings")
            extension = "wav" if format_type == "wav" else "mp4"
            output_file = os.path.join(
                recordings_dir, 
                f"recording_{int(time.time())}.{extension}"
            )
            AudioRecorderClass = get_audio_recorder()
            recorder = AudioRecorderClass(
                output_file=output_file, 
                websocket_manager=websocket_manager,
                session_id=session_id
            )
            
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
                except Exception as e:
                    logger.warning(f"Failed to cleanup recorder for session {session_id}: {e}")
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
        # Thread-safe retrieval to prevent race condition
        websocket = self.connections.get(session_id)
        if websocket is not None:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Send message failed for session {session_id}: {e}")
                # Only disconnect if the connection still exists in our dict
                if session_id in self.connections:
                    await self.disconnect(session_id)

class OptimizedWebSocketManager(SimpleWebSocketManager):
    """Optimized WebSocket manager with better progress tracking and performance monitoring"""
    
    def __init__(self):
        super().__init__()
        # Enhanced progress tracking
        self.progress_cache = {}  # Cache to avoid duplicate messages
        self.message_count = {}   # Track message frequency per session
        self.last_progress_time = {}  # Throttle progress updates
        
        # Performance tracking
        from src.performance_monitor import get_performance_monitor
        self.performance_monitor = get_performance_monitor()
        
        logger.info("OptimizedWebSocketManager initialized with enhanced progress tracking")
    
    async def send_chunked_progress(self, session_id: str, chunk_progress: dict):
        """Send fine-grained progress for better UX"""
        try:
            current_time = time.time()
            
            # Throttle progress updates (max 10 updates per second)
            last_update = self.last_progress_time.get(session_id, 0)
            if current_time - last_update < 0.1:  # 100ms minimum interval
                return
            
            self.last_progress_time[session_id] = current_time
            
            # Enhanced progress message with more detail
            progress_message = {
                "type": "detailed_progress",
                "timestamp": current_time,
                "transcription_progress": chunk_progress.get("transcription", 0),
                "diarization_progress": chunk_progress.get("diarization", 0), 
                "current_chunk": chunk_progress.get("chunk_number", 0),
                "total_chunks": chunk_progress.get("total_chunks", 1),
                "processing_stage": chunk_progress.get("stage", "processing"),
                "estimated_completion": chunk_progress.get("estimated_completion"),
                "real_time_ratio": chunk_progress.get("real_time_ratio")
            }
            
            # Add chunk-specific details if available
            if "chunk_details" in chunk_progress:
                progress_message["chunk_details"] = chunk_progress["chunk_details"]
            
            await self.send_message(session_id, progress_message)
            
        except Exception as e:
            logger.error(f"Failed to send chunked progress for {session_id}: {e}")
    
    async def send_model_download_progress(self, session_id: str, progress: dict):
        """Send enhanced model download progress with detailed information"""
        try:
            download_message = {
                "type": "model_download_progress",
                "timestamp": time.time(),
                "language": progress.get("language"),
                "model_name": progress.get("model_name"),
                "progress_percent": progress.get("progress", 0),
                "download_speed_mbps": progress.get("download_speed", 0),
                "estimated_time_remaining": progress.get("eta_seconds"),
                "downloaded_mb": progress.get("downloaded_mb", 0),
                "total_size_mb": progress.get("total_mb", 0),
                "stage": progress.get("stage", "downloading")  # downloading, extracting, loading
            }
            
            await self.send_message(session_id, download_message)
            
        except Exception as e:
            logger.error(f"Failed to send model download progress for {session_id}: {e}")
    
    async def send_performance_update(self, session_id: str, performance_data: dict):
        """Send performance metrics to client for monitoring"""
        try:
            # Only send if client supports performance updates
            if not performance_data.get("real_time", True):
                return
            
            performance_message = {
                "type": "performance_update", 
                "timestamp": time.time(),
                "real_time_ratio": performance_data.get("real_time_ratio", 1.0),
                "memory_usage_mb": performance_data.get("memory_mb", 0),
                "processing_speed": performance_data.get("processing_speed", "normal"),
                "quality_level": performance_data.get("quality", "standard"),
                "latency_ms": performance_data.get("latency_ms", 0)
            }
            
            await self.send_message(session_id, performance_message)
            
        except Exception as e:
            logger.debug(f"Performance update failed for {session_id}: {e}")  # Debug level - not critical
    
    async def send_audio_analysis_progress(self, session_id: str, analysis_data: dict):
        """Send audio analysis progress (VAD, chunking, preprocessing)"""
        try:
            analysis_message = {
                "type": "audio_analysis_progress",
                "timestamp": time.time(),
                "stage": analysis_data.get("stage", "analyzing"),  # analyzing, chunking, preprocessing
                "progress": analysis_data.get("progress", 0),
                "audio_duration": analysis_data.get("duration", 0),
                "chunks_created": analysis_data.get("chunks", 0),
                "voice_activity_detected": analysis_data.get("vad_segments", 0),
                "preprocessing_applied": analysis_data.get("preprocessing", [])
            }
            
            await self.send_message(session_id, analysis_message)
            
        except Exception as e:
            logger.error(f"Failed to send audio analysis progress for {session_id}: {e}")
    
    async def send_error_with_context(self, session_id: str, error: str, context: dict = None):
        """Send enhanced error messages with context for better debugging"""
        try:
            error_message = {
                "type": "error",
                "timestamp": time.time(),
                "message": error,
                "severity": context.get("severity", "error") if context else "error",
                "recovery_suggestions": context.get("recovery", []) if context else [],
                "error_code": context.get("code") if context else None,
                "technical_details": context.get("details") if context else None
            }
            
            await self.send_message(session_id, error_message)
            
        except Exception as e:
            logger.error(f"Failed to send enhanced error for {session_id}: {e}")
    
    async def send_completion_summary(self, session_id: str, results: dict):
        """Send completion with comprehensive summary and file information"""
        try:
            # Get session performance data if available
            performance_data = {}
            try:
                # Get performance stats from monitor
                monitor_report = await self.performance_monitor.get_performance_report()
                if monitor_report:
                    current_status = monitor_report.get("current_status", {})
                    performance_data = {
                        "real_time_ratio": current_status.get("average_real_time_ratio", 0),
                        "memory_usage": current_status.get("current_memory_mb", 0),
                        "processing_efficiency": "excellent" if current_status.get("average_real_time_ratio", 1) < 0.8 else "good"
                    }
            except Exception:
                pass  # Performance data is optional
            
            completion_message = {
                "type": "processing_complete",
                "timestamp": time.time(),
                "results": {
                    "transcription_data": results.get("transcription_data", []),
                    "diarization_segments": results.get("diarization_segments", []),
                    "speakers_detected": results.get("speakers_detected", 0),
                    "duration": results.get("duration", 0),
                    "language_detected": results.get("language", "unknown")
                },
                "files": {
                    "audio_file": results.get("audio_file"),
                    "srt_file": results.get("srt_file"),
                    "transcript_file": results.get("transcript_file")
                },
                "quality_metrics": {
                    "confidence_average": results.get("confidence_avg", 0),
                    "segments_processed": len(results.get("transcription_data", [])),
                    "processing_method": results.get("method", "whisper_pyaudioanalysis")
                },
                "performance": performance_data
            }
            
            await self.send_message(session_id, completion_message)
            
        except Exception as e:
            logger.error(f"Failed to send completion summary for {session_id}: {e}")
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Enhanced connection with performance monitoring integration"""
        await super().connect(websocket, session_id)
        
        # Initialize session tracking
        self.progress_cache[session_id] = {}
        self.message_count[session_id] = 0
        self.last_progress_time[session_id] = 0
        
        # Send initial connection confirmation with capabilities
        await self.send_message(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": time.time(),
            "capabilities": {
                "detailed_progress": True,
                "performance_monitoring": True,
                "audio_analysis": True,
                "enhanced_error_reporting": True
            },
            "server_info": {
                "version": "2.0.0",
                "optimizations": "enabled"
            }
        })
    
    async def disconnect(self, session_id: str):
        """Enhanced disconnection with cleanup"""
        # Clean up session data
        self.progress_cache.pop(session_id, None)
        self.message_count.pop(session_id, None) 
        self.last_progress_time.pop(session_id, None)
        
        await super().disconnect(session_id)
    
    async def send_message(self, session_id: str, message: dict):
        """Enhanced message sending with monitoring and throttling"""
        try:
            # Count messages per session
            self.message_count[session_id] = self.message_count.get(session_id, 0) + 1
            
            # Add message ID for tracking
            message["message_id"] = self.message_count[session_id]
            
            # Call parent implementation
            await super().send_message(session_id, message)
            
        except Exception as e:
            logger.error(f"Enhanced message sending failed for {session_id}: {e}")
            raise

# Global instances
app_state = SimpleState()
websocket_manager = OptimizedWebSocketManager()

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
                        
                    case 'system_download_start':
                        this.updateStatus(`${message.message || 'Downloading required components...'}`);
                        break;
                        
                    case 'system_download_progress':
                        if (message.progress) {
                            this.updateStatus(`${message.message || 'Downloading...'} (${message.progress}%)`);
                        } else {
                            this.updateStatus(message.message || 'Downloading...');
                        }
                        break;
                        
                    case 'system_download_complete':
                        this.updateStatus(`${message.message || 'Download completed'}`);
                        setTimeout(() => {
                            if (!this.isRecording) {
                                this.updateStatus('Connected - Ready to record');
                            }
                        }, 2000);
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
                            const timeInfo = duration > 0 ? ` (${speakerData.firstTime.toFixed(1)}s -> ${speakerData.lastTime.toFixed(1)}s)` : '';
                            
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
                if not app_state.create_recorder_for_session(session_id, format_type, websocket_manager):
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
            concurrent_processor_instance = get_concurrent_processor()
            result = await concurrent_processor_instance.process_audio_concurrent(
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
            # Use locals().get to avoid referencing variables that may not have been set
            wav_file_for_processing = locals().get("wav_file_for_processing")
            audio_file = locals().get("audio_file")
            if (
                isinstance(wav_file_for_processing, str)
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
    # Configure port from environment variable with fallback
    port = int(os.getenv("TRANSCREVAI_PORT", "8001"))
    host = os.getenv("TRANSCREVAI_HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False  # Disable for production
    )