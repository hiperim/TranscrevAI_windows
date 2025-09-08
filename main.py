# CRITICAL FIX: Enhanced main application with intelligent model management and user choices
import asyncio
import logging
import os
import time
import datetime
import random
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# Lazy import functions for heavy dependencies
def get_audio_recorder():
    """Lazy import AudioRecorder"""
    from src.audio_processing import AudioRecorder
    return AudioRecorder

def get_transcription_func():
    """Lazy import transcription functions"""
    from src.transcription import get_transcription_functions
    return get_transcription_functions()

def get_speaker_diarization():
    """Lazy import SpeakerDiarization"""
    from src.speaker_diarization import SpeakerDiarization
    return SpeakerDiarization

def get_model_config():
    """Lazy import model configuration"""
    from config.app_config import (
        WHISPER_MODEL_DIR, WHISPER_MODELS, WHISPER_CONFIG, 
        ADAPTIVE_PROMPTS, PROCESSING_PROFILES
    )
    return WHISPER_MODEL_DIR, WHISPER_MODELS, WHISPER_CONFIG, ADAPTIVE_PROMPTS, PROCESSING_PROFILES

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
    title="TranscrevAI Enhanced",
    description="Advanced Real-time Audio Transcription with AI - Multi-language, Multi-method Diarization",
    version="2.0.0"
)

# Jinja2 templates setup
templates = Jinja2Templates(directory="templates")

# Global whisper module initialization
_whisper_module = None

class EnhancedWhisperModelManager:
    """CRITICAL FIX: Enhanced Whisper model management with complexity-based selection and intelligent caching"""
    
    _model_cache = {}
    _loading_locks = {}
    _cache_ttl = 86400  # 24 hour cache TTL
    _cache_timestamps = {}
    
    @staticmethod
    def _get_whisper():
        """Lazy import of whisper module"""
        global _whisper_module
        try:
            # Initialize if not exists
            if '_whisper_module' not in globals() or _whisper_module is None:
                import whisper
                _whisper_module = whisper
            return _whisper_module
        except Exception as e:
            logger.error(f"Failed to load whisper module: {e}")
            raise ImportError(f"Cannot load whisper: {e}")
    
    @staticmethod
    def get_model_name(language: str, complexity: str = "medium") -> str:
        """CRITICAL FIX: Get Whisper model name based on language and complexity"""
        _, WHISPER_MODELS, _, _, _ = get_model_config()
        
        # Use medium model for all supported languages
        return WHISPER_MODELS.get(language, "medium")
    
    @classmethod
    async def get_cached_model(cls, language: str, complexity: str = "medium"):
        """Get cached model if available and not expired"""
        model_name = cls.get_model_name(language, complexity)
        
        # Thread-safe cache access
        if model_name in cls._model_cache:
            cached_model = cls._model_cache.get(model_name)
            if cached_model is not None:
                # Check if cache is still valid
                cache_time = cls._cache_timestamps.get(model_name, 0)
                if time.time() - cache_time < cls._cache_ttl:
                    logger.info(f"Using cached Whisper model: {model_name}")
                    return cached_model
                else:
                    # Cache expired, remove it safely
                    cls._model_cache.pop(model_name, None)
                    cls._cache_timestamps.pop(model_name, None)
        
        return None
    
    @classmethod
    async def cache_model(cls, language: str, model, complexity: str = "medium"):
        """Cache model with timestamp"""
        model_name = cls.get_model_name(language, complexity)
        cls._model_cache[model_name] = model
        cls._cache_timestamps[model_name] = time.time()
        logger.info(f"Cached Whisper model: {model_name}")
    
    @classmethod  
    async def ensure_whisper_model(cls, language: str, complexity: str = "medium", websocket_manager=None, session_id=None) -> bool:
        """CRITICAL FIX: Enhanced model management with complexity-based selection"""
        try:
            whisper = cls._get_whisper()
            
            model_name = cls.get_model_name(language, complexity)
            logger.info(f"Ensuring Whisper model: {model_name} for {language} ({complexity} complexity)")
            
            # Check if model is already cached
            cached_model = await cls.get_cached_model(language, complexity)
            if cached_model:
                # Send completion notification for cached model
                if websocket_manager and session_id:
                    await websocket_manager.send_message(session_id, {
                        "type": "model_download_complete",
                        "message": f"Model {model_name} ready (cached)",
                        "language": language,
                        "model": model_name,
                        "complexity": complexity
                    })
                return True
            
            # Use locking to prevent concurrent downloads of same model
            if model_name not in cls._loading_locks:
                cls._loading_locks[model_name] = asyncio.Lock()
            
            async with cls._loading_locks[model_name]:
                # Double-check cache after acquiring lock
                cached_model = await cls.get_cached_model(language, complexity)
                if cached_model:
                    return True
            
                # Use running event loop for executor tasks
                loop = asyncio.get_running_loop()
                
                # Send download start notification
                if websocket_manager and session_id:
                    await websocket_manager.send_message(session_id, {
                        "type": "model_download_start",
                        "message": f"Loading {model_name} model for {language} ({complexity} complexity)...",
                        "language": language,
                        "model": model_name,
                        "complexity": complexity
                    })
            
                # Try to load and cache model  
                try:
                    WHISPER_MODEL_DIR, _, _, _, _ = get_model_config()
                    model = await loop.run_in_executor(
                        None,
                        whisper.load_model,
                        model_name,
                        "cpu",  # Use CPU for validation
                        str(WHISPER_MODEL_DIR)
                    )
                    
                    # Cache the loaded model
                    await cls.cache_model(language, model, complexity)
                    
                    logger.info(f"Whisper model '{model_name}' loaded and cached for {language} ({complexity})")
                    
                    # Send completion notification
                    if websocket_manager and session_id:
                        await websocket_manager.send_message(session_id, {
                            "type": "model_download_complete",
                            "message": f"Model {model_name} ready",
                            "language": language,
                            "model": model_name,
                            "complexity": complexity
                        })
                    
                    return True
                    
                except Exception as e:
                    logger.info(f"Model '{model_name}' not available, downloading: {e}")
                    
                    # Download and cache model
                    WHISPER_MODEL_DIR, _, _, _, _ = get_model_config()
                    model = await loop.run_in_executor(
                        None,
                        whisper.load_model,
                        model_name,
                        "cpu",
                        str(WHISPER_MODEL_DIR)
                    )
                    
                    # Cache the downloaded model
                    await cls.cache_model(language, model, complexity)
                    
                    logger.info(f"Whisper model '{model_name}' downloaded and cached for {language} ({complexity})")
                    
                    # Send completion notification
                    if websocket_manager and session_id:
                        await websocket_manager.send_message(session_id, {
                            "type": "model_download_complete", 
                            "message": f"Model {model_name} downloaded successfully",
                            "language": language,
                            "model": model_name,
                            "complexity": complexity
                        })
                    
                    return True
                
        except Exception as e:
            logger.error(f"Enhanced Whisper model setup failed for {language} ({complexity}): {e}")
            
            # Send download error notification
            if websocket_manager and session_id:
                await websocket_manager.send_message(session_id, {
                    "type": "model_download_error",
                    "message": f"Failed to download model for {language}: {str(e)}",
                    "language": language,
                    "complexity": complexity
                })
            
            return False

# Enhanced state management with user choices
class EnhancedState:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, session_id: str):
        try:
            # Create session with enhanced metadata for user choices
            self.sessions[session_id] = {
                "recorder": None,
                "recording": False,
                "paused": False,
                "progress": {
                    "complexity_analysis": 0,
                    "transcription": 0, 
                    "diarization": 0
                },
                "websocket": None,
                "start_time": None,
                "task": None,
                "model_task": None,
                # CRITICAL FIX: User choice tracking
                "user_choices": {
                    "language": "pt",  # Default
                    "audio_input_type": "neutral",  # Default
                    "processing_profile": "balanced"  # Default
                },
                "complexity": "medium",
                "quality_metrics": {}
            }
            logger.info(f"Enhanced session created: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create enhanced session: {e}")
            return False
    
    def update_user_choices(self, session_id: str, language: Optional[str] = None, audio_input_type: Optional[str] = None, processing_profile: Optional[str] = None):
        """Update user choices for session"""
        if session_id in self.sessions:
            if language:
                self.sessions[session_id]["user_choices"]["language"] = language
            if audio_input_type:
                self.sessions[session_id]["user_choices"]["audio_input_type"] = audio_input_type
            if processing_profile:
                self.sessions[session_id]["user_choices"]["processing_profile"] = processing_profile
            logger.info(f"Updated choices for {session_id}: {self.sessions[session_id]['user_choices']}")
    
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
            if session.get("model_task"):
                session["model_task"].cancel()
            del self.sessions[session_id]
            logger.info(f"Enhanced session cleaned up: {session_id}")

# Enhanced WebSocket manager
class EnhancedWebSocketManager:
    def __init__(self):
        self.connections = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.connections[session_id] = websocket
        logger.info(f"Enhanced WebSocket connected: {session_id}")
    
    async def disconnect(self, session_id: str):
        websocket = self.connections.pop(session_id, None)
        if websocket:
            try:
                await websocket.close()
            except Exception as e:
                logger.warning(f"Error closing websocket for {session_id}: {e}")
        await app_state.cleanup_session(session_id)
        logger.info(f"Enhanced WebSocket disconnected: {session_id}")
    
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

# Global instances
app_state = EnhancedState()
websocket_manager = EnhancedWebSocketManager()

# Health check with enhanced info
@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.datetime.now().isoformat(),
        "sessions": len(app_state.sessions),
        "features": [
            "adaptive_transcription",
            "multi_method_diarization", 
            "quality_metrics",
            "user_choice_support",
            "complexity_analysis"
        ]
    })

# Main interface
@app.get("/")
async def main_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint with enhanced info
@app.get("/api")
async def api_status():
    _, WHISPER_MODELS, _, ADAPTIVE_PROMPTS, PROCESSING_PROFILES = get_model_config()
    return {
        "message": "TranscrevAI Enhanced API is running", 
        "version": "2.0.0",
        "supported_languages": list(WHISPER_MODELS.keys()),
        "audio_input_types": list(ADAPTIVE_PROMPTS.get("pt", {}).keys()),
        "processing_profiles": list(PROCESSING_PROFILES.keys())
    }

# Enhanced WebSocket handler with user choices support
@app.websocket("/ws/{session_id}")
async def enhanced_websocket_handler(websocket: WebSocket, session_id: str):
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
            await handle_enhanced_websocket_message(session_id, data)
    except WebSocketDisconnect:
        await websocket_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"Enhanced WebSocket error: {e}")
        await websocket_manager.disconnect(session_id)

# CRITICAL FIX: Enhanced message handler with user choices and complexity analysis
async def handle_enhanced_websocket_message(session_id: str, data: dict):
    message_type = data.get("type")
    message_data = data.get("data", {})
    session = app_state.get_session(session_id)
    
    if not session:
        await websocket_manager.send_message(session_id, {
            "type": "error",
            "message": "Session not found"
        })
        return
    
    # CRITICAL FIX: Handle user choices before recording
    if message_type == "set_user_choices":
        language = message_data.get("language", "pt")
        audio_input_type = message_data.get("audio_input_type", "neutral") 
        processing_profile = message_data.get("processing_profile", "balanced")
        
        app_state.update_user_choices(session_id, language, audio_input_type, processing_profile)
        
        await websocket_manager.send_message(session_id, {
            "type": "choices_updated",
            "choices": {
                "language": language,
                "audio_input_type": audio_input_type,
                "processing_profile": processing_profile
            },
            "message": f"Settings updated: {language} language, {audio_input_type} input type, {processing_profile} profile"
        })
        return
    
    if message_type == "start_recording":
        if not session.get("recording", False):
            try:
                # Get user choices
                user_choices = session.get("user_choices", {})
                language = message_data.get("language") or user_choices.get("language", "pt")
                audio_input_type = message_data.get("audio_input_type") or user_choices.get("audio_input_type", "neutral")
                processing_profile = message_data.get("processing_profile") or user_choices.get("processing_profile", "balanced")
                format_type = message_data.get("format", "wav")

                # Update user choices with any new values
                app_state.update_user_choices(session_id, language, audio_input_type, processing_profile)

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

                # Start recording immediately
                await recorder.start_recording()

                app_state.update_session(session_id, {
                    "recording": True,
                    "start_time": time.time(),
                    "language": language,
                    "audio_input_type": audio_input_type,
                    "processing_profile": processing_profile,
                    "format": format_type
                })

                await websocket_manager.send_message(session_id, {
                    "type": "recording_started",
                    "message": f"Recording started with {language} language, {audio_input_type} input type, {processing_profile} profile",
                    "settings": {
                        "language": language,
                        "audio_input_type": audio_input_type,
                        "processing_profile": processing_profile
                    }
                })

                # Start audio monitoring and enhanced concurrent processing
                asyncio.create_task(monitor_audio(session_id))
                task = asyncio.create_task(
                    enhanced_process_audio_concurrent(session_id, language, audio_input_type, processing_profile, format_type)
                )
                app_state.update_session(session_id, {"task": task})

            except Exception as e:
                logger.error(f"Enhanced start recording error: {e}")
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

# Audio monitoring (unchanged but enhanced logging)
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
        logger.error(f"Enhanced audio monitoring error: {e}")

# CRITICAL FIX: Enhanced concurrent processing pipeline with user choices
async def enhanced_process_audio_concurrent(session_id: str, language: str, audio_input_type: str, processing_profile: str, format_type: str = "wav"):
    try:
        session = app_state.get_session(session_id)
        if not session:
            return
        
        logger.info(f"Starting enhanced processing for {session_id}: {language}, {audio_input_type}, {processing_profile}")
        
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
        
        logger.info(f"Processing enhanced: {audio_file} (size: {file_size} bytes)")
        
        # Convert MP4 to WAV for transcription if needed
        wav_file_for_processing = audio_file
        if audio_file.endswith('.mp4'):
            try:
                # Create temporary WAV file for processing
                temp_dir = FileManager.get_data_path("temp")
                FileManager.ensure_directory_exists(temp_dir)
                wav_file_for_processing = os.path.join(temp_dir, f"enhanced_temp_for_transcription_{int(time.time())}.wav")
                
                # Convert MP4 to WAV using FFmpeg
                ffmpeg_args = [
                    "ffmpeg", "-y",
                    "-i", audio_file,
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    wav_file_for_processing
                ]
                
                logger.info(f"Converting MP4 to WAV for enhanced transcription: {wav_file_for_processing}")
                
                process = await asyncio.create_subprocess_exec(
                    *ffmpeg_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
                
                if process.returncode != 0:
                    raise Exception(f"Enhanced FFmpeg conversion failed: {stderr.decode()}")
                
                # Verify conversion was successful
                if not os.path.exists(wav_file_for_processing) or os.path.getsize(wav_file_for_processing) == 0:
                    raise Exception("Enhanced WAV conversion produced empty file")
                
                logger.info(f"Enhanced MP4 to WAV conversion successful: {wav_file_for_processing}")
                
            except Exception as e:
                logger.error(f"Enhanced MP4 to WAV conversion failed: {e}")
                await websocket_manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Failed to convert MP4 for enhanced transcription: {str(e)}"
                })
                return
        
        # Use enhanced concurrent processing engine
        try:
            concurrent_processor_instance = get_concurrent_processor()
            result = await concurrent_processor_instance.process_audio_concurrent(
                session_id, wav_file_for_processing, language, websocket_manager, 
                audio_input_type, processing_profile
            )
            
            transcription_data = result.get("transcription_data", [])
            diarization_segments = result.get("diarization_segments", [])
            unique_speakers = result.get("speakers_detected", 0)
            quality_metrics = result.get("quality_metrics", {})
            complexity = result.get("complexity", "medium")
            
        except Exception as e:
            logger.error(f"Enhanced concurrent processing error: {e}")
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": f"Enhanced processing failed: {str(e)}"
            })
            return
        
        # Generate SRT subtitle file only if we have valid transcription data
        srt_file = None
        try:
            if transcription_data and len(transcription_data) > 0:
                # Validate that transcription data contains actual text content
                has_valid_content = any(
                    segment.get('text', '').strip() 
                    for segment in transcription_data 
                    if isinstance(segment, dict) and segment.get('text')
                )
                
                if has_valid_content:
                    srt_file = await generate_srt(transcription_data, diarization_segments)
                    if srt_file:
                        logger.info(f"Enhanced SRT generated successfully: {srt_file}")
                    else:
                        logger.warning("Enhanced SRT generation returned None")
                else:
                    logger.warning("Enhanced SRT generation skipped: No valid text content in transcription data")
            else:
                logger.warning("Enhanced SRT generation skipped: No transcription data available")
        except Exception as e:
            logger.error(f"Enhanced SRT generation failed: {e}")
        
        # Clean up temporary WAV file if it was created
        if wav_file_for_processing != audio_file and os.path.exists(wav_file_for_processing):
            try:
                os.remove(wav_file_for_processing)
                logger.info(f"Cleaned up temporary WAV file: {wav_file_for_processing}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary WAV file: {cleanup_error}")
        
        # CRITICAL FIX: Enhanced results with comprehensive data
        await websocket_manager.send_message(session_id, {
            "type": "processing_complete",
            "transcription_data": transcription_data,
            "diarization_segments": diarization_segments,
            "speakers_detected": unique_speakers,
            "complexity": complexity,
            "quality_metrics": quality_metrics,
            "user_choices": {
                "language": language,
                "audio_input_type": audio_input_type,
                "processing_profile": processing_profile
            },
            "srt_file": srt_file,
            "audio_file": audio_file,
            "duration": session.get("duration", 0) if session else 0
        })
        
        # Update session with quality metrics
        app_state.update_session(session_id, {
            "quality_metrics": quality_metrics,
            "complexity": complexity
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Enhanced processing error: {error_msg}")
        
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
            "message": "Enhanced processing failed. Please try again."
        })

# Production startup
if __name__ == "__main__":
    # Configure port from environment variable with fallback
    port = int(os.getenv("TRANSCREVAI_PORT", "8001"))
    host = os.getenv("TRANSCREVAI_HOST", "0.0.0.0")
    
    logger.info("Starting TranscrevAI Enhanced v2.0.0")
    logger.info("Features: Adaptive Transcription, Multi-method Diarization, Quality Metrics, User Choices")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False  # Disable for production
    )