import asyncio
import logging
import os
import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Union

import urllib.request
import zipfile
import shutil
try:
    import aiofiles
    import aiofiles.os
    ASYNC_FILES_AVAILABLE = True
except ImportError:
    ASYNC_FILES_AVAILABLE = False

from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from src.audio_processing import AudioRecorder
from src.transcription import transcribe_audio_with_progress
from src.speaker_diarization import SpeakerDiarization
from src.subtitle_generator import generate_srt
from src.streaming_processor import StreamingProcessor
from config.app_config import MODEL_DIR, LANGUAGE_MODELS, FASTAPI_CONFIG
from src.logging_setup import setup_app_logging

logger = setup_app_logging()
if logger is None:
    import logging
    import json
    
    class StructuredFormatter(logging.Formatter):
        """
        Custom formatter for structured JSON logging.
        """
        def format(self, record):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add extra fields if present
            if hasattr(record, 'session_id'):
                log_entry["session_id"] = getattr(record, 'session_id', '')
            if hasattr(record, 'client_ip'):
                log_entry["client_ip"] = getattr(record, 'client_ip', '')
            if hasattr(record, 'user_agent'):
                log_entry["user_agent"] = getattr(record, 'user_agent', '')
            if hasattr(record, 'request_id'):
                log_entry["request_id"] = getattr(record, 'request_id', '')
                
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
                
            return json.dumps(log_entry)
    
    logger = logging.getLogger("TranscrevAI")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

# FastAPI setup
app = FastAPI(
    title=FASTAPI_CONFIG["title"],
    description=FASTAPI_CONFIG["description"],
    version=FASTAPI_CONFIG["version"]
)

class ModelManager:
    """
    Manages AI model lifecycle including download, validation, and caching.
    
    Handles automatic downloading of language models for speech recognition,
    validates model file structure, and provides retry logic for failed downloads.
    Designed for background operation without blocking the UI.
    """

    @staticmethod
    def get_model_path(language: str) -> str:
        """
        Get the file system path for a language model.
        
        Args:
            language (str): Language code (e.g., 'en', 'pt', 'es')
            
        Returns:
            str: Full path to the model directory
        """
        return os.path.join(MODEL_DIR, language)
    
    @staticmethod
    def validate_model(language: str) -> bool:
        """
        Validate that a language model is properly installed and complete.
        
        Checks for required model files including acoustic models, language models,
        and ivector extractors to ensure the model can be used for transcription.
        
        Args:
            language (str): Language code to validate
            
        Returns:
            bool: True if model is valid and complete, False otherwise
        """
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
    async def ensure_model_with_feedback(language: str, _websocket_manager: Optional[Any] = None, _session_id: Optional[str] = None) -> bool:
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
    async def _download_model_with_retry(language: str, _websocket_manager: Optional[Any] = None, _session_id: Optional[str] = None, max_retries: int = 3) -> bool:
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
                for root, _dirs, _files in os.walk(temp_dir):
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
    """
    Manages session state for concurrent audio recording sessions.
    
    Handles creation, tracking, and cleanup of recording sessions, including
    audio recorders, progress tracking, and WebSocket connections. Provides
    session isolation for multiple concurrent users.
    """
    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, session_id: str) -> bool:
        """
        Create a new recording session.
        
        Args:
            session_id (str): Unique identifier for the session
            
        Returns:
            bool: True if session created successfully, False otherwise
        """
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
                "task": None,
                "streaming_processor": None,
                "streaming_mode": False
            }
            logger.info(f"Session created: {session_id}", extra={"session_id": session_id, "action": "session_created"})
            return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False
    
    def create_recorder_for_session(self, session_id: str, format_type: str = "wav") -> bool:
        try:
            from src.file_manager import FileManager
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
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
    
    async def cleanup_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            try:
                session = self.sessions[session_id]
                
                # Clean up recorder resources
                recorder = session.get("recorder")
                if recorder and hasattr(recorder, 'cleanup_resources'):
                    try:
                        await recorder.cleanup_resources()
                    except Exception as e:
                        logger.warning(f"Recorder cleanup failed for session {session_id}: {e}")
                elif recorder:
                    logger.warning(f"Recorder has no cleanup_resources method for session {session_id}")
                
                # Cancel background tasks
                if session.get("task"):
                    try:
                        session["task"].cancel()
                    except Exception as e:
                        logger.warning(f"Task cancellation failed for session {session_id}: {e}")
                
                # Cancel model download task if it exists
                if session.get("model_task"):
                    try:
                        session["model_task"].cancel()
                    except Exception as e:
                        logger.warning(f"Model task cancellation failed for session {session_id}: {e}")
                
                # Clean up streaming processor
                if session.get("streaming_processor"):
                    try:
                        await session["streaming_processor"].stop_processing()
                        session["streaming_processor"].cleanup()
                    except Exception as e:
                        logger.warning(f"Streaming processor cleanup failed for session {session_id}: {e}")
                
                del self.sessions[session_id]
                logger.info(f"Session cleaned up: {session_id}")
                
            except Exception as e:
                logger.error(f"Session cleanup failed for {session_id}: {e}")
                # Force remove from sessions even if cleanup failed
                if session_id in self.sessions:
                    del self.sessions[session_id]

# WebSocket manager
class SimpleWebSocketManager:
    """
    Manages WebSocket connections for real-time communication.
    
    Handles WebSocket connection lifecycle, message broadcasting, and
    graceful disconnection with automatic session cleanup. Provides
    reliable real-time communication between server and clients.
    """
    def __init__(self, max_connections: int = 100) -> None:
        self.connections: Dict[str, WebSocket] = {}
        self.max_connections = max_connections
        self.connection_pool_size = 0
    
    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket (WebSocket): FastAPI WebSocket instance
            session_id (str): Unique session identifier
        """
        # Check connection pool limit
        if self.connection_pool_size >= self.max_connections:
            await websocket.close(code=1013, reason="Server overloaded - too many connections")
            logger.warning(f"Connection rejected - pool full: {session_id}", 
                         extra={"session_id": session_id, "action": "connection_pool_full"})
            return
            
        await websocket.accept()
        self.connections[session_id] = websocket
        self.connection_pool_size += 1
        logger.info(f"WebSocket connected: {session_id} (pool: {self.connection_pool_size}/{self.max_connections})", 
                   extra={"session_id": session_id, "action": "websocket_connected", "pool_size": self.connection_pool_size})
    
    async def disconnect(self, session_id: str) -> None:
        if session_id in self.connections:
            try:
                del self.connections[session_id]
                self.connection_pool_size = max(0, self.connection_pool_size - 1)
                await app_state.cleanup_session(session_id)
                logger.info(f"WebSocket disconnected: {session_id} (pool: {self.connection_pool_size}/{self.max_connections})",
                           extra={"session_id": session_id, "action": "websocket_disconnected", "pool_size": self.connection_pool_size})
            except Exception as e:
                logger.error(f"Error during WebSocket disconnect cleanup: {e}", 
                           extra={"session_id": session_id, "error": str(e)})
    
    async def send_message(self, session_id: str, message: Dict[str, Any]) -> None:
        if session_id in self.connections:
            try:
                await self.connections[session_id].send_json(message)
            except Exception as e:
                logger.error(f"Send message failed: {e}")
                await self.disconnect(session_id)

# Rate limiting
class RateLimiter:
    """
    Implements rate limiting for WebSocket connections and message frequency.
    
    Tracks connection counts per IP and message rates per session to prevent
    abuse and ensure fair resource usage across all clients.
    """
    
    def __init__(self, max_connections_per_ip: int = 10, max_messages_per_minute: int = 60):
        self.max_connections_per_ip = max_connections_per_ip
        self.max_messages_per_minute = max_messages_per_minute
        self.connections_per_ip: Dict[str, int] = defaultdict(int)
        self.message_timestamps: Dict[str, deque] = defaultdict(lambda: deque())
        
    def can_connect(self, client_ip: str) -> bool:
        """Check if a new connection from this IP is allowed."""
        return self.connections_per_ip[client_ip] < self.max_connections_per_ip
    
    def add_connection(self, client_ip: str) -> None:
        """Register a new connection from this IP."""
        self.connections_per_ip[client_ip] += 1
    
    def remove_connection(self, client_ip: str) -> None:
        """Remove a connection from this IP."""
        if self.connections_per_ip[client_ip] > 0:
            self.connections_per_ip[client_ip] -= 1
    
    def can_send_message(self, session_id: str) -> bool:
        """Check if this session can send another message."""
        now = time.time()
        timestamps = self.message_timestamps[session_id]
        
        # Remove timestamps older than 1 minute
        while timestamps and now - timestamps[0] > 60:
            timestamps.popleft()
        
        return len(timestamps) < self.max_messages_per_minute
    
    def record_message(self, session_id: str) -> None:
        """Record a message from this session."""
        self.message_timestamps[session_id].append(time.time())
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up rate limiting data for a session."""
        if session_id in self.message_timestamps:
            del self.message_timestamps[session_id]

# File path sanitization
def get_safe_filename(file_path: str) -> str:
    """
    Extract only the filename from a full path for security.
    
    Args:
        file_path (str): Full file path
        
    Returns:
        str: Safe filename without directory path
    """
    if not file_path:
        return "unknown_file"
    
    filename = os.path.basename(file_path)
    # Sanitize filename to prevent directory traversal
    filename = filename.replace("..", "").replace("/", "_").replace("\\", "_")
    return filename or "unknown_file"

# Input validation
def validate_websocket_message(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate WebSocket message structure and content.
    
    Args:
        data (Dict[str, Any]): Raw message data
        
    Returns:
        Dict[str, str]: Validation result with 'valid' and 'error' keys
    """
    if not isinstance(data, dict):
        return {"valid": "false", "error": "Message must be a JSON object"}
    
    message_type = data.get("type")
    if not message_type or not isinstance(message_type, str):
        return {"valid": "false", "error": "Message must have a 'type' field"}
    
    # Validate allowed message types
    allowed_types = ["start_recording", "stop_recording", "pause_recording", "resume_recording", "ping", "enable_streaming"]
    if message_type not in allowed_types:
        return {"valid": "false", "error": f"Invalid message type: {message_type}"}
    
    # Validate message data
    message_data = data.get("data", {})
    if not isinstance(message_data, dict):
        return {"valid": "false", "error": "Message data must be an object"}
    
    # Validate specific message types
    if message_type == "start_recording":
        language = message_data.get("language", "en")
        format_type = message_data.get("format", "wav")
        
        if not isinstance(language, str) or language not in ["en", "pt", "es"]:
            return {"valid": "false", "error": "Invalid language parameter"}
        
        if not isinstance(format_type, str) or format_type not in ["wav", "mp4"]:
            return {"valid": "false", "error": "Invalid format parameter"}
        
        # Validate advanced_audio parameter if present
        if "advanced_audio" in message_data:
            advanced_audio = message_data.get("advanced_audio")
            if not isinstance(advanced_audio, bool):
                return {"valid": "false", "error": "Invalid advanced_audio parameter - must be boolean"}
    
    elif message_type == "enable_streaming":
        language = message_data.get("language", "en")
        
        if not isinstance(language, str) or language not in ["en", "pt", "es"]:
            return {"valid": "false", "error": "Invalid language parameter for streaming"}
    
    return {"valid": "true", "error": ""}

# Global instances
app_state = SimpleState()
websocket_manager = SimpleWebSocketManager()
rate_limiter = RateLimiter()

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
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
        }

        .pause-btn {
            background: linear-gradient(45deg, #ffa726, #fb8c00);
            color: white;
        }

        .stop-btn {
            background: linear-gradient(45deg, #66bb6a, #43a047);
            color: white;
        }

        .button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .checkbox-container {
            display: flex;
            align-items: center;
            cursor: pointer;
            font-size: 0.9rem;
            margin: 5px;
            user-select: none;
        }

        .checkbox-container input[type="checkbox"] {
            margin-right: 8px;
            transform: scale(1.2);
        }

        .checkmark {
            margin-left: 5px;
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

        .streaming-results, .merged-results {
            max-height: 200px;
            overflow-y: auto;
            background: #f0f8ff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }

        .streaming-segment {
            margin: 5px 0;
            padding: 5px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #667eea;
        }

        .timestamp {
            color: #666;
            font-size: 0.8rem;
            font-weight: bold;
        }

        .confidence {
            color: #888;
            font-size: 0.8rem;
            float: right;
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
                <label class="checkbox-container">
                    <input type="checkbox" id="streamingMode">
                    <span class="checkmark"></span>
                    Real-time Streaming Mode (Phase 2)
                </label>
                
                <label class="checkbox-container">
                    <input type="checkbox" id="advancedAudio" checked>
                    <span class="checkmark"></span>
                    Advanced Audio Processing
                </label>
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
                
                // Phase 2 elements
                this.streamingModeEl = document.getElementById('streamingMode');
                this.advancedAudioEl = document.getElementById('advancedAudio');
                
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
                    try {
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    } catch (error) {
                        console.error('Failed to parse WebSocket message:', error);
                        this.showError('Invalid message received from server');
                    }
                };
            }
            
            handleMessage(message) {
                switch (message.type) {
                    case 'streaming_enabled':
                        this.updateStatus('Streaming mode enabled - Advanced processing active');
                        break;
                        
                    case 'streaming_transcription':
                        this.handleStreamingTranscription(message);
                        break;
                        
                    case 'streaming_diarization':
                        this.handleStreamingDiarization(message);
                        break;
                        
                    case 'merged_results':
                        this.handleMergedResults(message);
                        break;
                        
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
                    data.transcription_data.forEach(item => {
                        resultsHTML += `<div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                            <strong>Speaker ${item.speaker || 'Unknown'}:</strong> ${item.text || item.content || 'No text'}
                        </div>`;
                    });
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
            
            // Phase 2 streaming handlers
            handleStreamingTranscription(message) {
                this.updateProgress(message.transcription || 0, 0);
                
                // Display real-time transcription
                const streamingDiv = document.getElementById('streaming-results') || this.createStreamingResultsDiv();
                streamingDiv.innerHTML += `<div class="streaming-segment">
                    <span class="timestamp">[${message.timestamp.toFixed(1)}s]</span>
                    <span class="text">${message.text}</span>
                    <span class="confidence">(${(message.confidence * 100).toFixed(1)}%)</span>
                </div>`;
                streamingDiv.scrollTop = streamingDiv.scrollHeight;
            }
            
            handleStreamingDiarization(message) {
                // Update speaker information in real-time
                if (message.speakers && message.speakers.length > 0) {
                    const speakerInfo = message.speakers.map(s => `Speaker ${s.speaker}`).join(', ');
                    this.updateStatus(`Recording... (Speakers: ${speakerInfo})`);
                }
            }
            
            handleMergedResults(message) {
                // Update overall progress and merged results
                this.updateProgress(90, 90);
                
                if (message.transcription && message.transcription.length > 0) {
                    const mergedDiv = document.getElementById('merged-results') || this.createMergedResultsDiv();
                    mergedDiv.innerHTML = `<h4>Merged Results (${message.chunk_count} chunks)</h4>
                        <p><strong>Confidence:</strong> ${(message.confidence * 100).toFixed(1)}%</p>
                        <p>${message.merged_text}</p>`;
                }
            }
            
            createStreamingResultsDiv() {
                const streamingDiv = document.createElement('div');
                streamingDiv.id = 'streaming-results';
                streamingDiv.className = 'streaming-results';
                streamingDiv.innerHTML = '<h4>Real-time Transcription</h4>';
                this.resultsEl.appendChild(streamingDiv);
                return streamingDiv;
            }
            
            createMergedResultsDiv() {
                const mergedDiv = document.createElement('div');
                mergedDiv.id = 'merged-results';
                mergedDiv.className = 'merged-results';
                this.resultsEl.appendChild(mergedDiv);
                return mergedDiv;
            }

            // CRITICAL FIX: Safe format access with fallback
            startRecording() {
                this.resultsEl.classList.remove('visible');
                
                // Check if streaming mode should be enabled
                if (this.streamingModeEl && this.streamingModeEl.checked) {
                    this.send('enable_streaming', { 
                        language: this.languageEl.value
                    });
                }
                
                // Safe format access with fallback
                const formatValue = this.formatEl && this.formatEl.value ? this.formatEl.value : 'wav';
                
                this.send('start_recording', { 
                    language: this.languageEl.value,
                    format: formatValue,
                    advanced_audio: this.advancedAudioEl ? this.advancedAudioEl.checked : true
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
    return {"message": "TranscrevAI API is running", "version": FASTAPI_CONFIG["version"]}

# WebSocket handler w/ rate limiting and model management
@app.websocket("/ws/{session_id}")
async def websocket_handler(websocket: WebSocket, session_id: str):
    # Get client IP for rate limiting
    client_ip = getattr(websocket.client, 'host', 'unknown') if websocket.client else 'unknown'
    
    # Check connection rate limit
    if not rate_limiter.can_connect(client_ip):
        await websocket.close(code=1008, reason="Too many connections from this IP")
        logger.warning(f"Rate limit exceeded for IP: {client_ip}", 
                       extra={"client_ip": client_ip, "action": "rate_limit_exceeded"})
        return
    
    rate_limiter.add_connection(client_ip)
    
    try:
        await websocket_manager.connect(websocket, session_id)
        
        if not app_state.create_session(session_id):
            await websocket_manager.send_message(session_id, {
                "type": "error", 
                "message": "Failed to create session"
            })
            return
        
        while True:
            data = await websocket.receive_json()
            
            # Check message rate limit
            if not rate_limiter.can_send_message(session_id):
                await websocket_manager.send_message(session_id, {
                    "type": "error",
                    "message": "Rate limit exceeded. Please slow down."
                })
                continue
            
            rate_limiter.record_message(session_id)
            await handle_websocket_message(session_id, data)
            
    except WebSocketDisconnect:
        await websocket_manager.disconnect(session_id)
        rate_limiter.remove_connection(client_ip)
        rate_limiter.cleanup_session(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket_manager.disconnect(session_id)
        rate_limiter.remove_connection(client_ip)
        rate_limiter.cleanup_session(session_id)

# Enhanced message handler w/ model management
async def handle_websocket_message(session_id: str, data: Dict[str, Any]) -> None:
    # Validate message structure and content
    validation_result = validate_websocket_message(data)
    if validation_result["valid"] == "false":
        await websocket_manager.send_message(session_id, {
            "type": "error",
            "message": f"Invalid message: {validation_result['error']}"
        })
        logger.warning(f"Invalid WebSocket message from {session_id}: {validation_result['error']}", 
                       extra={"session_id": session_id, "action": "invalid_message", "error": validation_result['error']})
        return
    
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
                advanced_audio = message_data.get("advanced_audio", True)

                # Create recorder with correct format
                if not app_state.create_recorder_for_session(session_id, format_type):
                    await websocket_manager.send_message(session_id, {
                        "type": "error",
                        "message": "Failed to create recorder"
                    })
                    return
                
                # Get the newly created recorder with better validation
                session = app_state.get_session(session_id)
                if not session:
                    await websocket_manager.send_message(session_id, {
                        "type": "error",
                        "message": "Session not found after recorder creation"
                    })
                    return
                
                recorder = session.get("recorder")
                if recorder is None:
                    await websocket_manager.send_message(session_id, {
                        "type": "error",
                        "message": "Failed to initialize audio recorder"
                    })
                    return

                # Start recording immediately (no waiting for model)
                await recorder.start_recording()

                app_state.update_session(session_id, {
                    "recording": True,
                    "start_time": time.time(),
                    "language": language,
                    "format": format_type,
                    "advanced_audio": advanced_audio
                })

                await websocket_manager.send_message(session_id, {
                    "type": "recording_started",
                    "message": "Recording started"
                })

                # Check if streaming mode is enabled
                if session.get("streaming_mode") and session.get("streaming_processor"):
                    # Start streaming processing
                    streaming_task = asyncio.create_task(
                        session["streaming_processor"].start_processing(websocket_manager, session_id)
                    )
                    app_state.update_session(session_id, {"task": streaming_task})
                    
                    # Start real-time audio monitoring for streaming
                    asyncio.create_task(monitor_audio_streaming(session_id))
                else:
                    # Start model download silently in background (non-blocking)
                    model_task = asyncio.create_task(
                        ModelManager.ensure_model_silent(language)
                    )
                    
                    # Start audio monitoring and processing (traditional mode)
                    asyncio.create_task(monitor_audio(session_id))
                    task = asyncio.create_task(process_audio(session_id, language, format_type))
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
    
    elif message_type == "enable_streaming":
        try:
            language = message_data.get("language", "en")
            
            # Create and initialize streaming processor
            streaming_processor = StreamingProcessor()
            if await streaming_processor.initialize_services(language):
                app_state.update_session(session_id, {
                    "streaming_processor": streaming_processor,
                    "streaming_mode": True
                })
                
                await websocket_manager.send_message(session_id, {
                    "type": "streaming_enabled",
                    "message": "Real-time streaming mode activated"
                })
            else:
                await websocket_manager.send_message(session_id, {
                    "type": "error",
                    "message": "Failed to enable streaming mode"
                })
        
        except Exception as e:
            logger.error(f"Enable streaming error: {e}")
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": "Failed to enable streaming mode"
            })
    
    elif message_type == "ping":
        await websocket_manager.send_message(session_id, {"type": "pong"})

# Audio monitoring
async def monitor_audio(session_id: str) -> None:
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

# Real-time audio monitoring for streaming mode
async def monitor_audio_streaming(session_id: str) -> None:
    """Monitor audio and feed chunks to streaming processor"""
    try:
        session = app_state.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found for streaming audio monitoring")
            return
            
        streaming_processor = session.get("streaming_processor")
        recorder = session.get("recorder")
        
        if not streaming_processor or not recorder:
            logger.error(f"Missing streaming processor or recorder for session {session_id}")
            return
        
        chunk_duration = 2.0  # 2 second chunks
        sample_rate = 16000
        chunk_samples = int(chunk_duration * sample_rate)
        
        start_time = time.time()
        
        while session and session["recording"]:
            if not session["paused"]:
                try:
                    # Get real audio data from recorder
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Get real audio chunk from recorder
                    audio_chunk = recorder.get_recent_audio_chunk(chunk_duration)
                    
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        # Send to streaming processor with advanced audio setting
                        enable_processing = session.get("advanced_audio", True)
                        await streaming_processor.add_audio_chunk(audio_chunk, elapsed, enable_processing)
                        
                        # Send audio level update based on real audio
                        level = np.abs(audio_chunk).mean()
                        # Normalize level for display (assuming float32 range -1 to 1)
                        level = min(1.0, level * 5.0)  # Scale for better visualization
                        await websocket_manager.send_message(session_id, {
                            "type": "audio_level",
                            "level": level
                        })
                    else:
                        # No audio data available yet, send zero level
                        await websocket_manager.send_message(session_id, {
                            "type": "audio_level",
                            "level": 0.0
                        })
                    
                except Exception as chunk_error:
                    logger.error(f"Streaming chunk processing error: {chunk_error}")
                    # Send zero level on error
                    try:
                        await websocket_manager.send_message(session_id, {
                            "type": "audio_level",
                            "level": 0.0
                        })
                    except:
                        pass
            
            await asyncio.sleep(chunk_duration)
            session = app_state.get_session(session_id)
            
    except Exception as e:
        logger.error(f"Streaming audio monitoring error: {e}")

# Enhanced processing pipeline with proper SRT generation
async def process_audio(session_id: str, language: str = "en", _format_type: str = "wav") -> None:
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
            logger.error(f"No session or recorder found for session {session_id}")
            return
        
        recorder = session.get("recorder")
        if not recorder or not hasattr(recorder, 'output_file'):
            logger.error(f"Invalid recorder or missing output_file for session {session_id}")
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": "Recording failed - no output file found"
            })
            return
        
        audio_file = recorder.output_file
        
        # Enhanced file validation with async operations
        try:
            if ASYNC_FILES_AVAILABLE:
                file_exists = await aiofiles.os.path.exists(audio_file)
                if not file_exists:
                    await websocket_manager.send_message(session_id, {
                        "type": "error",
                        "message": "Audio file not found. Recording may have failed."
                    })
                    return
                
                file_stat = await aiofiles.os.stat(audio_file)
                file_size = file_stat.st_size
            else:
                # Fallback to synchronous operations
                if not os.path.exists(audio_file):
                    await websocket_manager.send_message(session_id, {
                        "type": "error",
                        "message": "Audio file not found. Recording may have failed."
                    })
                    return
                
                file_size = os.path.getsize(audio_file)
        except Exception as e:
            logger.error(f"File validation error: {e}", extra={"session_id": session_id, "action": "file_validation_error"})
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": "File access error during validation."
            })
            return
            
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
        if session:
            model_task = session.get("model_task")
            if model_task and hasattr(model_task, '__await__'):
                try:
                    # Wait for model download to finish (background task)
                    model_result = await model_task
                    if not model_result:
                        logger.warning("Background model download returned False, attempting fallback")
                        await ModelManager.ensure_model_silent(language)
                except Exception as e:
                    logger.error(f"Background model download failed: {e}")
                    # Try to ensure model is available as fallback
                    await ModelManager.ensure_model_silent(language)
            else:
                # No background model task or task not awaitable, ensure model is available
                logger.info("No background model task found, ensuring model availability")
                await ModelManager.ensure_model_silent(language)
        else:
            logger.warning(f"Session {session_id} not found during model check")
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
            # Use the async method for better performance in async context
            segments = await diarizer.diarize_audio(audio_file)
            diarization_segments = segments if segments else []
            unique_speakers = len(set(seg.get('speaker', 'Unknown') for seg in diarization_segments)) if diarization_segments else 0

            # Send 100% progress when diarization is complete
            await websocket_manager.send_message(session_id, {
                "type": "progress",
                "transcription": 100,
                "diarization": 100
            })

        except Exception as e:
            logger.error(f"Diarization error: {e}")
            diarization_segments = []
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
        
        # Send results with sanitized file paths
        await websocket_manager.send_message(session_id, {
            "type": "processing_complete",
            "transcription_data": transcription_data,
            "diarization_segments": diarization_segments,
            "speakers_detected": unique_speakers,
            "srt_file": get_safe_filename(srt_file) if srt_file else None,
            "audio_file": get_safe_filename(audio_file),  # Sanitized filename only
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