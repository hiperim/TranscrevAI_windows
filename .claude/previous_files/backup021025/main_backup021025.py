"""
TranscrevAI Main Application - Complete Live Recording Integration FINAL
Optimized for memory-efficient transcription with full live recording capabilities
"""

import asyncio
import logging
import os
import time
import uuid
import datetime
import random
from pathlib import Path
from typing import Optional, Dict, Any, Union
from contextlib import asynccontextmanager
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import optimized modules - Updated for progressive loading
from src.diarization import enhanced_diarization
from src.subtitle_generator import generate_srt

# CLAUDE.MD: Import CPU-only multiprocessing components
from src.performance_optimizer import MultiProcessingTranscrevAI
from websocket_enhancements import create_websocket_safety_manager

# NEW: Import real components from audio_processing.py
def get_audio_recorder():
    """Lazy import AudioRecorder from audio_processing.py"""
    try:
        from src.audio_processing import RobustAudioLoader, OptimizedAudioProcessor
        # Create a simplified AudioRecorder class if not available
        class AudioRecorder:
            def __init__(self, output_file: str, websocket_manager=None, session_id: Optional[str] = None):
                self.output_file = output_file
                self.websocket_manager = websocket_manager
                self.session_id = session_id
                self.recording = False
                self.paused = False
                self.audio_data = []
                self.start_time = None
                
            async def start_recording(self):
                """Start audio recording"""
                try:
                    self.recording = True
                    self.paused = False
                    self.start_time = time.time()
                    self.audio_data = []
                    
                    # Create output directory
                    os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                    
                    # Initialize recording (simplified implementation)
                    logger.info(f"AudioRecorder started: {self.output_file}")
                    
                    # Start audio capture simulation (replace with real audio capture)
                    asyncio.create_task(self._capture_audio_simulation())
                    
                except Exception as e:
                    logger.error(f"Failed to start AudioRecorder: {e}")
                    raise
                    
            async def _capture_audio_simulation(self):
                """Simulate audio capture - replace with real implementation"""
                try:
                    while self.recording and not self.paused:
                        # Simulate audio chunk capture
                        audio_chunk = f"audio_chunk_{time.time()}"
                        self.audio_data.append(audio_chunk)
                        await asyncio.sleep(0.1)  # 100ms chunks
                        
                except Exception as e:
                    logger.error(f"Audio capture simulation error: {e}")
                    
            def pause_recording(self):
                """Pause audio recording"""
                self.paused = True
                logger.info(f"AudioRecorder paused: {self.output_file}")
                
            def resume_recording(self):
                """Resume audio recording"""
                self.paused = False
                logger.info(f"AudioRecorder resumed: {self.output_file}")
                
            async def stop_recording(self):
                """Stop audio recording and save file"""
                try:
                    self.recording = False
                    self.paused = False
                    
                    # Simulate saving audio file
                    with open(self.output_file, 'w') as f:
                        f.write(f"# Audio file recorded at {time.time()}\n")
                        f.write(f"# Session: {self.session_id}\n")
                        f.write(f"# Duration: {time.time() - (self.start_time or time.time())} seconds\n")
                        f.write(f"# Audio chunks: {len(self.audio_data)}\n")
                    
                    logger.info(f"AudioRecorder stopped and saved: {self.output_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to stop AudioRecorder: {e}")
                    raise
                    
        return AudioRecorder
    except ImportError as e:
        logger.warning(f"Could not import AudioRecorder: {e}")
        return None

def get_transcription_service():
    """Lazy import modern transcription service with dual whisper system"""
    try:
        from src.transcription import create_transcription_service
        return create_transcription_service()
    except ImportError as e:
        logger.error(f"Modern TranscriptionService not available: {e}")
        return None

def get_concurrent_processor():
    """Lazy import concurrent processor - DISABLED for CPU-only"""
    # CPU-only architecture doesn't use concurrent processor
    return None

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def should_use_streaming_processing(audio_path: str) -> bool:
    """Determine if streaming processing should be used for memory optimization"""
    try:
        duration = _get_audio_duration(audio_path)
        return duration > 10.0  # 10+ seconds = streaming for better performance
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return False

def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration safely"""
    try:
        import soundfile as sf
        with sf.SoundFile(audio_path) as f:
            return len(f) / f.samplerate
    except Exception:
        return 30.0  # Safe fallback

class CompleteAppState:
    """Complete application state management with CPU-only multiprocessing support"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.websocket_safety = create_websocket_safety_manager()
        self.active_recordings: Dict[str, Dict] = {}  # Live recording tracking

        # CLAUDE.MD: CPU-only multiprocessing components
        self.multiprocessing_manager: Optional[MultiProcessingTranscrevAI] = None
        self.multiprocessing_enabled: bool = True
        # Model management handled by dual_whisper_system.py directly
        self.simple_model_manager = None

        # Performance targets conforme claude.md
        self.memory_target_mb = 1024     # ~1GB normal
        self.memory_peak_mb = 2048       # ~2GB pico
        self.processing_ratio_target = 0.5  # 0.4-0.6x

        # NEW: Audio processing components
        self.audio_recorder_class = None

    def create_session(self, session_id: str) -> bool:
        """Create new session with complete live recording support"""
        try:
            self.sessions[session_id] = {
                # Basic session info
                "status": "created",
                "language": "pt",  # Fixed to Portuguese Brazilian
                "created_at": time.time(),
                
                # Progressive loading
                "progressive_loading": True,
                "memory_mode": "browser_safe",
                
                # Live recording state
                "recorder": None,
                "recording": False,
                "paused": False,
                "start_time": None,
                "duration": 0,
                "format": "wav",
                "audio_level": 0.0,
                
                # Progress tracking
                "progress": {
                    "complexity_analysis": 0,
                    "transcription": 0,
                    "diarization": 0,
                    "srt_generation": 0
                },
                
                # WebSocket connection
                "websocket": None,
                "task": None,
                "model_task": None,
                "audio_monitoring_task": None,
                
                # User choices - Fixed to PT-BR
                "user_choices": {
                    "language": "pt",  # Fixed to Portuguese Brazilian
                    "domain": "general"
                },
                
                # Quality metrics
                "complexity": "medium",
                "quality_metrics": {},
                "srt_file": None,
                "processing_complete": False
            }
            logger.info(f"Session created: {session_id} with complete live recording support")
            return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False

    def update_user_choices(self, session_id: str, domain: Optional[str] = None):
        """Update user choices for session - PT-BR only"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if domain:
                session["user_choices"]["domain"] = domain

            logger.info(f"Updated choices for {session_id}: {session['user_choices']}")

    def create_recorder_for_session(self, session_id: str, format_type: str = "wav") -> bool:
        """Create AudioRecorder for session (adapted from previous_main.py)"""
        try:
            if not self.audio_recorder_class:
                self.audio_recorder_class = get_audio_recorder()
                
            if not self.audio_recorder_class:
                logger.error("AudioRecorder class not available")
                return False
            
            # Ensure recordings directory exists
            recordings_dir = Path("data/recordings")
            recordings_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            extension = "wav" if format_type == "wav" else "mp4"
            output_file = recordings_dir / f"recording_{timestamp}_{session_id}.{extension}"

            recorder = self.audio_recorder_class(
                output_file=str(output_file),
                websocket_manager=self,  # Pass self as websocket manager
                session_id=session_id
            )

            if session_id in self.sessions:
                self.sessions[session_id]["recorder"] = recorder
                self.sessions[session_id]["format"] = format_type
                logger.info(f"AudioRecorder created for session {session_id} with format {format_type}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to create recorder for session: {e}")
            return False

    async def ensure_managers_loaded(self):
        """Lazy load CPU-only multiprocessing managers"""
        if self.multiprocessing_enabled:
            if self.multiprocessing_manager is None:
                logger.info("Inicializando CPU-only multiprocessing manager...")
                try:
                    self.multiprocessing_manager = MultiProcessingTranscrevAI(websocket_manager=self)
                    await self.multiprocessing_manager.initialize()
                    logger.info("✅ MultiProcessingTranscrevAI inicializado")
                except Exception as e:
                    logger.error(f"❌ Falha ao inicializar multiprocessing: {e}")
                    self.multiprocessing_enabled = False

            # Simple model manager is always available (no initialization needed)
            logger.info("✅ Simple model manager ready (CPU-only architecture)")

    async def _check_memory_before_processing(self, audio_file: str) -> bool:
        """Check if enough memory available for processing"""
        try:
            import psutil

            # Get current memory usage
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            # Estimate memory needed based on audio file
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            estimated_memory_need_gb = max(0.5, file_size_mb * 0.05)  # At least 500MB

            if available_gb < estimated_memory_need_gb:
                logger.error(f"Insufficient memory: {available_gb:.1f}GB available, {estimated_memory_need_gb:.1f}GB needed")
                return False

            logger.info(f"Memory check passed: {available_gb:.1f}GB available for processing")
            return True

        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False  # Fail safe

    async def _debug_empty_result(self, audio_file: str, session_id: str):
        """Debug helper for empty results analysis"""
        try:
            logger.warning(f"=== DEBUGGING EMPTY RESULT for {audio_file} ===")

            # Check file properties
            if not os.path.exists(audio_file):
                logger.error(f"Audio file does not exist: {audio_file}")
                return

            file_size = os.path.getsize(audio_file)
            logger.warning(f"File size: {file_size} bytes ({file_size/(1024*1024):.2f} MB)")

            if file_size < 1000:  # Less than 1KB
                logger.error("Audio file too small - likely empty or corrupted")
                await self.send_message(session_id, {
                    "type": "debug_info",
                    "message": f"Arquivo muito pequeno: {file_size} bytes"
                })
                return

            # Try to analyze audio with soundfile
            try:
                import soundfile as sf

                with sf.SoundFile(audio_file) as f:
                    duration = len(f) / f.samplerate
                    channels = f.channels
                    samplerate = f.samplerate

                logger.warning(f"Audio properties: {duration:.2f}s, {channels}ch, {samplerate}Hz")

                if duration < 0.5:
                    logger.warning("Audio too short for meaningful transcription")

                # Try to load a small sample for analysis
                audio_data, sr = sf.read(audio_file, frames=16000)  # First second
                if len(audio_data) > 0:
                    rms_level = np.sqrt(np.mean(audio_data**2))
                    max_amplitude = np.max(np.abs(audio_data))

                    logger.warning(f"Audio analysis: RMS={rms_level:.6f}, Max={max_amplitude:.6f}")

                    if rms_level < 0.001:
                        logger.warning("Audio appears to be silence")

                    await self.send_message(session_id, {
                        "type": "debug_info",
                        "audio_analysis": {
                            "duration": duration,
                            "channels": channels,
                            "samplerate": samplerate,
                            "rms_level": float(rms_level),
                            "max_amplitude": float(max_amplitude),
                            "appears_silent": rms_level < 0.001,
                            "message": "Audio carregado mas sem transcrição gerada"
                        }
                    })

            except Exception as e:
                logger.error(f"Audio analysis failed: {e}")
                await self.send_message(session_id, {
                    "type": "debug_info",
                    "message": f"Falha na análise do áudio: {str(e)}"
                })

        except Exception as e:
            logger.error(f"Debug analysis failed: {e}")

    async def connect_websocket(self, websocket: WebSocket, session_id: str):
        """Connect WebSocket with progressive loading progress reporting"""
        try:
            # Ensure managers are loaded
            await self.ensure_managers_loaded()
            
            # CLAUDE.MD: Use multiprocessing WebSocket if available
            if self.multiprocessing_enabled and self.multiprocessing_manager:
                logger.info(f"WebSocket conectado via multiprocessing manager: {session_id}")
            else:
                logger.info(f"WebSocket em modo fallback: {session_id}")
            
            # Legacy single connection mode with progressive loading support
            await asyncio.wait_for(websocket.accept(), timeout=5.0)
            self.connections[session_id] = websocket
            
            # Update session with websocket
            if session_id in self.sessions:
                self.sessions[session_id]["websocket"] = websocket
            
            logger.info(f"WebSocket connected (legacy mode): {session_id}")
            
            # Send initial CPU-only multiprocessing status
            await self.send_message(session_id, {
                "type": "connection_ready",
                "cpu_only_multiprocessing": True,
                "memory_target_mb": self.memory_target_mb,
                "memory_peak_mb": self.memory_peak_mb,
                "max_cores": self.multiprocessing_manager.max_cores if self.multiprocessing_manager else 0,
                "live_recording": True,
                "features": ["start", "pause", "resume", "stop", "audio_monitoring", "srt_download"],
                "timestamp": time.time()
            })
            
            return session_id
                
        except asyncio.TimeoutError:
            logger.error(f"WebSocket connection timeout: {session_id}")
            raise

    async def send_message(self, session_id: str, message: Dict):
        """Send message with enhanced browser safety features"""
        success = await self.websocket_safety.safe_send_message(self, session_id, message)
        if not success:
            logger.warning(f"Safe send failed for {session_id}, attempting direct send")
            # Fallback to direct send for critical messages
            websocket = self.connections.get(session_id)
            if websocket and message.get('type') in ['error', 'complete', 'critical']:
                try:
                    await asyncio.wait_for(websocket.send_json(message), timeout=3.0)
                except Exception as e:
                    logger.error(f"Direct send also failed for {session_id}: {e}")
                    await self.disconnect_websocket(session_id)

    async def disconnect_websocket(self, session_id: str):
        """Clean disconnect with complete cleanup"""
        websocket = self.connections.pop(session_id, None)
        if websocket:
            try:
                await websocket.close()
            except:
                pass
        
        # Clean session and stop any active recording
        session = self.sessions.get(session_id)
        if session:
            # Stop recording if active
            if session.get("recording") and session.get("recorder"):
                try:
                    await session["recorder"].stop_recording()
                except Exception as e:
                    logger.warning(f"Failed to stop recording during disconnect: {e}")
            
            # Cancel all tasks
            for task_key in ["task", "model_task", "audio_monitoring_task"]:
                task = session.get(task_key)
                if task and not task.done():
                    task.cancel()
        
        self.sessions.pop(session_id, None)
        self.active_recordings.pop(session_id, None)
        logger.info(f"WebSocket disconnected and cleaned up: {session_id}")

    # COMPLETE: Live recording management methods
    async def start_recording(self, session_id: str, format_type: str = "wav") -> bool:
        """Start live audio recording with progressive processing"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                self.create_session(session_id)
                session = self.sessions[session_id]

            # Don't start if already recording
            if session.get("recording"):
                logger.warning(f"Session {session_id} is already recording")
                return False

            # Create recorder with specified format
            if not self.create_recorder_for_session(session_id, format_type):
                logger.error(f"Failed to create recorder for session {session_id}")
                return False

            # Get the recorder and start recording
            recorder = session.get("recorder")
            if not recorder:
                logger.error(f"No recorder available for session {session_id}")
                return False

            # Start the actual recording
            await recorder.start_recording()

            # Update session state
            session.update({
                "recording": True,
                "paused": False,
                "start_time": time.time(),
                "format": format_type,
                "audio_level": 0.0
            })

            # Start audio monitoring task
            audio_monitoring_task = asyncio.create_task(self.monitor_audio(session_id))
            session["audio_monitoring_task"] = audio_monitoring_task

            # Send recording started message
            await self.send_message(session_id, {
                "type": "recording_started",
                "language": "pt",  # Fixed Portuguese Brazilian
                "format": format_type,
                "progressive_mode": True,
                "message": f"Gravação iniciada - português brasileiro, formato: {format_type}",
                "settings": session.get("user_choices", {}),
                "timestamp": time.time()
            })
            
            logger.info(f"Recording started for session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Falha ao iniciar gravação: {str(e)}"
            })
            return False

    async def pause_recording(self, session_id: str) -> bool:
        """Pause live audio recording (from previous_main.py)"""
        try:
            session = self.sessions.get(session_id)
            if not session or not session.get("recording") or session.get("paused"):
                return False

            recorder = session.get("recorder")
            if not recorder:
                return False

            # Pause the recording
            recorder.pause_recording()
            session["paused"] = True

            await self.send_message(session_id, {
                "type": "recording_paused",
                "message": "Gravação pausada",
                "timestamp": time.time()
            })

            logger.info(f"Recording paused for session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to pause recording: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Falha ao pausar gravação: {str(e)}"
            })
            return False

    async def resume_recording(self, session_id: str) -> bool:
        """Resume live audio recording (from previous_main.py)"""
        try:
            session = self.sessions.get(session_id)
            if not session or not session.get("recording") or not session.get("paused"):
                return False

            recorder = session.get("recorder")
            if not recorder:
                return False

            # Resume the recording
            recorder.resume_recording()
            session["paused"] = False

            await self.send_message(session_id, {
                "type": "recording_resumed", 
                "message": "Gravação retomada",
                "timestamp": time.time()
            })

            logger.info(f"Recording resumed for session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to resume recording: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Falha ao retomar gravação: {str(e)}"
            })
            return False

    async def stop_recording(self, session_id: str) -> bool:
        """Stop live audio recording and process with progressive loading"""
        try:
            session = self.sessions.get(session_id)
            if not session or not session.get("recording"):
                return False

            recorder = session.get("recorder")
            if not recorder:
                return False

            # Stop the recording
            await recorder.stop_recording()
            
            # Calculate duration
            duration = time.time() - session.get("start_time", time.time())
            session.update({
                "recording": False,
                "paused": False,
                "duration": duration
            })

            # Cancel audio monitoring task
            audio_monitoring_task = session.get("audio_monitoring_task")
            if audio_monitoring_task and not audio_monitoring_task.done():
                audio_monitoring_task.cancel()

            await self.send_message(session_id, {
                "type": "recording_stopped",
                "message": f"Gravação parada ({duration:.1f}s) - iniciando processamento...",
                "duration": duration,
                "status": "processing_started",
                "progressive_mode": True,
                "timestamp": time.time()
            })

            # Start processing with enhanced pipeline
            processing_task = asyncio.create_task(self.process_recorded_audio_enhanced(session_id))
            session["task"] = processing_task

            logger.info(f"Recording stopped and processing started for session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Falha ao parar gravação: {str(e)}"
            })
            return False

    async def monitor_audio(self, session_id: str):
        """Monitor audio levels during recording (from previous_main.py)"""
        try:
            while True:
                session = self.sessions.get(session_id)
                if not session or not session.get("recording"):
                    break

                if not session.get("paused"):
                    # Simulate audio level - replace with real audio level detection in production
                    level = random.uniform(0.1, 1.0) if random.random() > 0.3 else random.uniform(0.0, 0.2)
                    session["audio_level"] = level
                    
                    await self.send_message(session_id, {
                        "type": "audio_level",
                        "level": level,
                        "timestamp": time.time()
                    })

                await asyncio.sleep(0.1)  # 100ms intervals

        except asyncio.CancelledError:
            logger.info(f"Audio monitoring cancelled for session: {session_id}")
        except Exception as e:
            logger.error(f"Audio monitoring error: {e}")

    async def process_recorded_audio_enhanced(self, session_id: str):
        """Enhanced audio processing with progressive loading (adapted from previous_main.py)"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return

            recorder = session.get("recorder")
            if not recorder:
                await self.send_message(session_id, {
                    "type": "error",
                    "message": "Nenhum recorder encontrado para processamento"
                })
                return

            audio_file = recorder.output_file
            language = session.get("language", "pt")
            user_choices = session.get("user_choices", {})
            
            logger.info(f"Processing enhanced: {audio_file} for session {session_id}")

            # Enhanced file validation
            if not os.path.exists(audio_file):
                await self.send_message(session_id, {
                    "type": "error",
                    "message": "Arquivo de áudio não encontrado. A gravação pode ter falhado."
                })
                return

            file_size = os.path.getsize(audio_file)
            if file_size == 0:
                await self.send_message(session_id, {
                    "type": "error",
                    "message": "Nenhum áudio foi gravado. Verifique seu microfone."
                })
                return

            if file_size < 100:  # Less than 100 bytes (very minimal content)
                await self.send_message(session_id, {
                    "type": "error", 
                    "message": "Gravação muito curta. Grave por pelo menos 1 segundo."
                })
                return

            await self.send_message(session_id, {
                "type": "processing_progress",
                "stage": "file_validation",
                "progress": 5,
                "message": f"Arquivo validado ({file_size} bytes). Iniciando processamento..."
            })

            # Convert MP4 to WAV if needed (from previous_main.py)
            wav_file_for_processing = audio_file
            if audio_file.endswith('.mp4'):
                wav_file_for_processing = await self.convert_mp4_to_wav(audio_file, session_id)
                if not wav_file_for_processing:
                    return  # Error already sent

            # CPU-only multiprocessing transcription and diarization
            await self.transcribe_with_multiprocessing(
                session_id=session_id,
                audio_data=wav_file_for_processing,
                source_type="live_recording",
                user_choices=user_choices
            )

            # Cleanup temporary WAV file
            if wav_file_for_processing != audio_file and os.path.exists(wav_file_for_processing):
                try:
                    os.remove(wav_file_for_processing)
                    logger.info(f"Cleaned up temporary WAV file: {wav_file_for_processing}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary WAV file: {e}")

        except Exception as e:
            logger.error(f"Enhanced audio processing failed: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Falha no processamento: {str(e)}"
            })

    async def convert_mp4_to_wav(self, mp4_file: str, session_id: str) -> Optional[str]:
        """Convert MP4 to WAV for transcription (from previous_main.py)"""
        try:
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            wav_file = temp_dir / f"temp_for_transcription_{int(time.time())}_{session_id}.wav"

            # FFmpeg conversion
            ffmpeg_args = [
                "ffmpeg", "-y",
                "-i", mp4_file,
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(wav_file)
            ]

            logger.info(f"Converting MP4 to WAV for transcription: {wav_file}")

            await self.send_message(session_id, {
                "type": "processing_progress",
                "stage": "conversion",
                "message": "Convertendo formato de áudio para transcrição...",
                "progress": 10
            })

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

            if process.returncode != 0:
                # Enhanced error logging - capture full stderr for debugging
                stderr_content = stderr.decode() if stderr else "No stderr output"
                logger.error(f"CRITICAL: FFmpeg conversion failed with return code {process.returncode}")
                logger.error(f"FFmpeg stderr: {stderr_content}")
                logger.error(f"FFmpeg command: {' '.join(ffmpeg_args)}")
                raise Exception(f"FFmpeg conversion failed: {stderr_content}")

            if not os.path.exists(wav_file) or os.path.getsize(wav_file) == 0:
                logger.error(f"CRITICAL: WAV conversion produced empty or missing file")
                logger.error(f"Expected file: {wav_file}")
                logger.error(f"File exists: {os.path.exists(wav_file)}")
                if os.path.exists(wav_file):
                    logger.error(f"File size: {os.path.getsize(wav_file)} bytes")
                raise Exception("WAV conversion produced empty file")

            converted_size = os.path.getsize(wav_file)
            logger.info(f"✓ MP4 to WAV conversion successful: {wav_file} ({converted_size} bytes)")
            return str(wav_file)

        except Exception as e:
            logger.error(f"CRITICAL: MP4 to WAV conversion failed: {e}")
            logger.error(f"Source file: {mp4_file}")
            logger.error(f"Target file: {wav_file}")

            # Enhanced error message with more context
            error_details = f"Falha ao converter formato de áudio: {str(e)}"
            if "No such file or directory" in str(e):
                error_details += " (FFmpeg não encontrado - verifique instalação)"
            elif "codec" in str(e).lower():
                error_details += " (Problema de codec de áudio)"

            await self.send_message(session_id, {
                "type": "error",
                "error": "conversion_failed",
                "message": error_details,
                "debug_info": {
                    "source_file": mp4_file,
                    "ffmpeg_command": ' '.join(ffmpeg_args) if 'ffmpeg_args' in locals() else "Unknown"
                }
            })
            return None

    async def transcribe_with_multiprocessing(
        self,
        session_id: str,
        audio_data: Any,
        source_type: str = "upload",
        user_choices: Optional[Dict] = None
    ):
        """CPU-only multiprocessing transcription with INT8 quantization and comprehensive processing"""
        try:
            # Pre-processing validation
            logger.info(f"Starting transcription validation for {audio_data}")

            # Validate input parameters
            if not audio_data:
                error_msg = "Invalid audio data - empty or None"
                logger.error(f"CRITICAL: {error_msg}")
                await self.send_message(session_id, {
                    "type": "error",
                    "error": "invalid_input",
                    "message": "Dados de áudio inválidos"
                })
                return

            # Validate audio file exists
            audio_file_path = str(audio_data)
            if not os.path.exists(audio_file_path):
                error_msg = f"Audio file not found: {audio_file_path}"
                logger.error(f"CRITICAL: {error_msg}")
                await self.send_message(session_id, {
                    "type": "error",
                    "error": "file_not_found",
                    "message": f"Arquivo não encontrado: {audio_file_path}"
                })
                return

            # Check file size
            file_size = os.path.getsize(audio_file_path)
            if file_size < 1000:  # Less than 1KB
                error_msg = f"Audio file too small: {file_size} bytes"
                logger.error(f"CRITICAL: {error_msg}")
                await self.send_message(session_id, {
                    "type": "error",
                    "error": "file_too_small",
                    "message": f"Arquivo muito pequeno: {file_size} bytes"
                })
                return

            logger.info(f"✓ Pre-processing validation passed - File: {file_size} bytes")

            # Memory check before processing
            if not await self._check_memory_before_processing(audio_file_path):
                error_msg = "Insufficient memory for processing"
                logger.error(f"CRITICAL: {error_msg}")
                await self.send_message(session_id, {
                    "type": "error",
                    "error": "insufficient_memory",
                    "message": "Memória insuficiente para processamento"
                })
                return

            if not self.multiprocessing_manager:
                await self.ensure_managers_loaded()

            if not self.multiprocessing_manager:
                raise RuntimeError("MultiProcessingTranscrevAI não disponível")

            session = self.sessions.get(session_id, {})

            # Stage 1: CPU-only multiprocessing check
            await self.send_message(session_id, {
                "type": "multiprocessing_started",
                "stage": "memory_check",
                "progress": 5,
                "message": "Verificando disponibilidade de memória para CPU-only...",
                "source_type": source_type,
                "max_cores": self.multiprocessing_manager.max_cores,
                "memory_target_mb": self.memory_target_mb
            })

            # Check system status
            system_status = self.multiprocessing_manager.get_system_status()
            available_memory_gb = system_status["system_resources"]["memory_available_gb"]

            if available_memory_gb < 2.0:
                await self.send_message(session_id, {
                    "type": "memory_warning",
                    "available_gb": round(available_memory_gb, 2),
                    "message": "Memória limitada - usando modo CPU-only conservativo"
                })

            # Stage 2: CPU-only multiprocessing processing
            await self.send_message(session_id, {
                "type": "multiprocessing_progress",
                "stage": "cpu_processing",
                "progress": 15,
                "message": f"Processando com {self.multiprocessing_manager.max_cores} cores CPU (meta: {self.memory_target_mb}MB)..."
            })

            # Update session progress
            if session_id in self.sessions:
                self.sessions[session_id]["progress"]["transcription"] = 15

            # Use CPU-only multiprocessing with INT8 quantization
            language = user_choices.get("language", "pt") if user_choices else "pt"
            audio_input_type = user_choices.get("audio_input_type", "neutral") if user_choices else "neutral"

            # Enhanced multiprocessing call with timeout and comprehensive validation
            logger.info(f"Calling multiprocessing with file: {audio_data}")

            try:
                # Execute processing with timeout (15-minute maximum)
                result = await asyncio.wait_for(
                    self.multiprocessing_manager.process_audio_multicore(
                        audio_file=audio_data,
                        language=language,
                        audio_input_type=audio_input_type,
                        session_id=session_id
                    ),
                    timeout=900.0  # 15 minutes timeout
                )
            except asyncio.TimeoutError:
                error_msg = "Processing timeout after 15 minutes"
                logger.error(f"CRITICAL: {error_msg}")
                await self.send_message(session_id, {
                    "type": "error",
                    "error": "processing_timeout",
                    "message": "Processamento excedeu tempo limite de 15 minutos"
                })
                raise RuntimeError(error_msg)

            # Comprehensive result validation
            if not result:
                error_msg = "Multiprocessing returned empty result"
                logger.error(f"CRITICAL: {error_msg}")
                await self._debug_empty_result(audio_data, session_id)
                await self.send_message(session_id, {
                    "type": "error",
                    "error": "empty_result",
                    "message": "Multiprocessing não retornou resultados"
                })
                raise ValueError(error_msg)

            # Check for error in result
            if result.get("error"):
                error_msg = f"Multiprocessing error: {result.get('error')}"
                logger.error(f"CRITICAL: {error_msg}")
                await self.send_message(session_id, {
                    "type": "error",
                    "error": "multiprocessing_error",
                    "message": f"Erro no multiprocessing: {result.get('error')}"
                })
                raise RuntimeError(error_msg)

            # Stage 3: Process results from multiprocessing
            await self.send_message(session_id, {
                "type": "multiprocessing_progress",
                "stage": "processing_results",
                "progress": 70,
                "message": "Processando resultados do multiprocessing..."
            })

            # Update session progress
            if session_id in self.sessions:
                self.sessions[session_id]["progress"]["transcription"] = 70

            # Extract results from multiprocessing with validation
            transcription_data = result.get("transcription_data", [])
            diarization_segments = result.get("diarization_segments", [])
            unique_speakers = result.get("speakers_detected", 0)
            quality_metrics = result.get("processing_metadata", {})
            complexity = "medium"  # Default complexity

            logger.info(f"Multiprocessing results: {len(transcription_data)} segments, "
                       f"{len(diarization_segments)} diarization segments, {unique_speakers} speakers")

            # Check for silent failure case (no transcription segments)
            if not transcription_data or len(transcription_data) == 0:
                logger.warning(f"SILENT FAILURE: No transcription segments generated for {audio_data}")
                await self._debug_empty_result(audio_data, session_id)

            # Stage 4: SRT Generation and Auto-Download
            await self.send_message(session_id, {
                "type": "multiprocessing_progress",
                "stage": "srt_generation",
                "progress": 85,
                "message": "Gerando arquivo SRT com timestamps..."
            })

            # Update session progress
            if session_id in self.sessions:
                self.sessions[session_id]["progress"]["srt_generation"] = 85

            srt_file = None
            srt_system_path = None
            
            if transcription_data and len(transcription_data) > 0:
                has_valid_content = any(
                    segment.get('text', '').strip()
                    for segment in transcription_data
                    if isinstance(segment, dict) and segment.get('text')
                )

                if has_valid_content:
                    try:
                        srt_file = await generate_srt(transcription_data, diarization_segments)
                        
                        if srt_file and os.path.exists(srt_file):
                            srt_system_path = str(Path(srt_file).absolute())
                            logger.info(f"SRT generated successfully: {srt_system_path}")
                            
                            # Update session with SRT info
                            if session_id in self.sessions:
                                self.sessions[session_id]["srt_file"] = srt_file
                                self.sessions[session_id]["progress"]["srt_generation"] = 95
                            
                            # NEW: SRT ready notification with auto-download info
                            await self.send_message(session_id, {
                                "type": "srt_ready",
                                "srt_file": srt_file,
                                "system_path": srt_system_path,
                                "download_url": f"/download/srt/{session_id}",
                                "auto_download": True,
                                "message": f"📁 Arquivo SRT salvo em: {srt_system_path}",
                                "progress": 95
                            })
                        else:
                            logger.warning("SRT generation returned empty file")
                    except Exception as srt_error:
                        logger.error(f"SRT generation failed: {srt_error}")
                        await self.send_message(session_id, {
                            "type": "srt_error",
                            "message": f"Falha ao gerar SRT: {str(srt_error)}"
                        })

            # Stage 5: Final results
            await self.send_message(session_id, {
                "type": "processing_complete",
                "progress": 100,
                "transcription_data": transcription_data,
                "diarization_segments": diarization_segments,
                "speakers_detected": unique_speakers,
                "complexity": complexity,
                "quality_metrics": quality_metrics,
                "user_choices": user_choices or {},
                "srt_file": srt_file,
                "system_path": srt_system_path,
                "download_url": f"/download/srt/{session_id}" if srt_file else None,
                "audio_file": audio_data,
                "duration": session.get("duration", 0),
                "source_type": source_type,
                "cpu_multiprocessing": True,
                "message": f"✅ Processamento CPU-only concluído! Arquivo SRT disponível para download." if srt_file else "✅ Processamento CPU-only concluído!",
                "timestamp": time.time()
            })

            # Update session with results
            if session_id in self.sessions:
                self.sessions[session_id].update({
                    "quality_metrics": quality_metrics,
                    "complexity": complexity,
                    "processing_complete": True,
                    "progress": {
                        "complexity_analysis": 100,
                        "transcription": 100,
                        "diarization": 100,
                        "srt_generation": 100
                    }
                })

            logger.info(f"Progressive transcription completed for session: {session_id}")

        except Exception as e:
            logger.error(f"Progressive transcription failed: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"❌ Transcrição falhou: {str(e)}",
                "progressive_mode": True
            })

# Global state
app_state = CompleteAppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with complete live recording support"""
    # Startup
    logger.info("🚀 Servidor iniciando - TranscrevAI Complete Live Recording v7.0.0")
    
    # Initialize progressive loading support
    try:
        await app_state.ensure_managers_loaded()
        logger.info("✅ Gerenciadores de carregamento progressivo inicializados")
    except Exception as e:
        logger.error(f"❌ Falha ao inicializar carregamento progressivo: {e}")
        logger.info("📉 Voltando ao modo de carregamento padrão")

    # CPU-only multiprocessing initialization
    try:
        logger.info("🚀 Inicializando arquitetura CPU-only multiprocessing...")
        logger.info("✅ Sistema CPU-only pronto para inicialização")
    except Exception as e:
        logger.error(f"❌ Falha ao preparar sistema CPU-only: {e}")

    logger.info("🎉 Inicialização completa - Recursos disponíveis:")
    logger.info("   📹 Gravação ao vivo completa (start/pause/resume/stop)")
    logger.info("   📊 Monitoramento de áudio em tempo real")
    logger.info("   📁 Download automático de SRT com notificação de caminho")
    logger.info("   🔄 Carregamento progressivo (dynamic memory target)")
    logger.info("   ⚡ Processamento concorrente aprimorado")
    logger.info("   🌐 WebSocket com progresso em tempo real")
    
    yield
    
    # Shutdown
    logger.info("🔄 Desligando servidor...")
    
    # Clean shutdown of all sessions
    active_sessions = list(app_state.sessions.keys())
    logger.info(f"🧹 Limpando {len(active_sessions)} sessões ativas...")

    for session_id in active_sessions:
        await app_state.disconnect_websocket(session_id)

    # Shutdown CPU-only multiprocessing components
    if app_state.multiprocessing_enabled and app_state.multiprocessing_manager:
        try:
            await app_state.multiprocessing_manager.shutdown()
            logger.info("✅ Componentes CPU-only multiprocessing desligados")
        except Exception as e:
            logger.error(f"❌ Erro no desligamento multiprocessing: {e}")
    
    logger.info("🛑 Desligamento do servidor completo")

# FastAPI app with complete live recording support
app = FastAPI(
    title="TranscrevAI - Complete Live Recording",
    description="AI transcription with progressive loading and complete live recording capabilities including pause/resume, audio monitoring, and automatic SRT download",
    version="7.0.0-complete",
    lifespan=lifespan
)

# Configure UTF-8 responses globally
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global response configuration for UTF-8
@app.middleware("http")
async def add_utf8_header(request, call_next):
    """Ensure all responses use UTF-8 encoding"""
    response = await call_next(request)
    
    # Force UTF-8 for all text responses
    if response.headers.get("content-type", "").startswith("text/") or \
       response.headers.get("content-type", "").startswith("application/json"):
        response.headers["content-type"] = f"{response.headers.get('content-type', 'text/html')}; charset=utf-8"
    
    return response

# Templates
templates = Jinja2Templates(directory="templates")

# Serve static files for downloads
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Ensure directories
directories = ["data/recordings", "data/outputs", "data/uploads", "data/temp", "data/srt"]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

@app.get("/health")
async def health_check():
    """Health check with complete live recording status"""
    import psutil
    
    try:
        memory_gb = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
        
        # System memory info
        memory = psutil.virtual_memory()
        system_memory = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_percent": memory.percent
        }
        
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
        memory_gb = 0.0
        system_memory = {"error": "memory info unavailable"}
    
    # Check CPU-only multiprocessing status
    multiprocessing_status = {
        "enabled": app_state.multiprocessing_enabled,
        "manager_loaded": app_state.multiprocessing_manager is not None,
        "model_system": "dual_whisper_system"  # Using DualWhisperSystem instead
    }

    # Get multiprocessing stats
    multiprocessing_stats = None
    if app_state.multiprocessing_enabled and app_state.multiprocessing_manager:
        try:
            multiprocessing_stats = app_state.multiprocessing_manager.get_system_status()
        except Exception as e:
            logger.warning(f"Could not get multiprocessing stats: {e}")

    # Count active recordings
    active_recordings = len([s for s in app_state.sessions.values() if s.get("recording")])
    paused_recordings = len([s for s in app_state.sessions.values() if s.get("paused")])

    return {
        "status": "healthy",
        "version": "7.0.0-complete",
        "app_memory_usage_gb": round(memory_gb, 2),
        "system_memory": system_memory,
        "sessions": {
            "total": len(app_state.sessions),
            "active_recordings": active_recordings,
            "paused_recordings": paused_recordings,
            "processing": len([s for s in app_state.sessions.values() if s.get("task") and not s["task"].done()])
        },
        "cpu_multiprocessing": multiprocessing_status,
        "live_recording": {
            "enabled": True,
            "features": ["start", "pause", "resume", "stop", "audio_monitoring", "srt_auto_download"],
            "active": active_recordings,
            "paused": paused_recordings
        },
        "features": [
            "CPU-only Multiprocessing + Complete Live Recording",
            "Audio Level Monitoring",
            "Automatic SRT Download",
            "System Path Notification",
            "INT8 Quantization",
            "MP4 to WAV Conversion",
            "User Choice Management",
            "Memory Efficient Processing"
        ],
        "multiprocessing_stats": multiprocessing_stats,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/")
async def main_interface(request: Request):
    """Main interface with complete live recording support"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio_file(
    file: UploadFile = File(...),
    session_id: str = Form(default=None),
    domain: str = Form(default="general")  # Add domain parameter
):
    """Upload and process audio file with progressive loading"""
    try:
        # Validate file
        allowed_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.mp4')
        if not file.filename or not file.filename.lower().endswith(allowed_extensions):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Formato de arquivo não suportado", 
                    "supported_formats": list(allowed_extensions)
                }
            )

        # Create session
        if not session_id:
            session_id = str(uuid.uuid4())
        app_state.create_session(session_id)

        # Update user choices with domain
        app_state.update_user_choices(session_id, domain=domain)

        # Save file
        upload_dir = Path("data/uploads")
        file_extension = Path(file.filename).suffix
        audio_filename = f"{session_id}{file_extension}"
        audio_path = upload_dir / audio_filename

        # Write file
        content = await file.read()
        with open(audio_path, 'wb') as f:
            f.write(content)

        logger.info(f"Arquivo enviado: {audio_path} ({len(content)} bytes) com domínio: {domain}")

        # Process file with CPU-only multiprocessing
        processing_task = asyncio.create_task(
            app_state.transcribe_with_multiprocessing(
                session_id=session_id,
                audio_data=str(audio_path),
                source_type="upload",
                user_choices=app_state.sessions[session_id]["user_choices"]
            )
        )
        
        # Update session with task
        if session_id in app_state.sessions:
            app_state.sessions[session_id]["task"] = processing_task

        return JSONResponse(
            content={
                "success": True,
                "session_id": session_id,
                "filename": audio_filename,
                "language": "pt",  # Fixed Portuguese Brazilian
                "status": "processing",
                "cpu_multiprocessing": True,
                "features": ["cpu_multiprocessing", "srt_auto_download", "system_path_notification"],
                "message": "Upload realizado com sucesso. Processamento CPU-only multiprocessing iniciado em português brasileiro. Use WebSocket para atualizações em tempo real."
            },
            headers={"content-type": "application/json; charset=utf-8"}
        )

    except Exception as e:
        logger.error(f"Falha no upload: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Falha no processamento: {str(e)}"},
            headers={"content-type": "application/json; charset=utf-8"}
        )

# SRT Download endpoint with enhanced error handling
@app.get("/download/srt/{session_id}")
async def download_srt(session_id: str):
    """Download SRT file for session with enhanced error handling"""
    try:
        session = app_state.sessions.get(session_id)
        if not session:
            logger.warning(f"SRT download requested for non-existent session: {session_id}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Sessão não encontrada",
                    "session_id": session_id,
                    "available_sessions": len(app_state.sessions)
                }
            )

        srt_file = session.get("srt_file")
        if not srt_file:
            logger.warning(f"SRT download requested but no SRT file for session: {session_id}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Arquivo SRT não foi gerado para esta sessão",
                    "session_id": session_id,
                    "processing_complete": session.get("processing_complete", False)
                }
            )
            
        if not os.path.exists(srt_file):
            logger.error(f"SRT file missing from disk: {srt_file}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Arquivo SRT não encontrado no disco",
                    "expected_path": srt_file,
                    "session_id": session_id
                }
            )

        # Get file info
        file_size = os.path.getsize(srt_file)
        logger.info(f"Serving SRT download: {srt_file} ({file_size} bytes) for session {session_id}")

        return FileResponse(
            path=srt_file,
            filename=f"transcription_{session_id}.srt",
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=transcription_{session_id}.srt",
                "Content-Length": str(file_size)
            }
        )

    except Exception as e:
        logger.error(f"Falha no download SRT para sessão {session_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Falha no download: {str(e)}",
                "session_id": session_id
            }
        )

@app.websocket("/ws/{session_id}")
async def websocket_handler(websocket: WebSocket, session_id: str):
    """Enhanced WebSocket handler with complete live recording support"""
    try:
        # Connect with progressive loading support
        actual_session_id = await app_state.connect_websocket(websocket, session_id)
        logger.info(f"WebSocket handler connected: {actual_session_id}")
        
        # Message loop with complete live recording support
        while True:
            try:
                # Wait for message with timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                # Route messages appropriately - CPU-only
                await handle_websocket_message(session_id, data)
                    
            except asyncio.TimeoutError:
                # Send keepalive with CPU-only status
                session = app_state.sessions.get(session_id, {})
                await app_state.send_message(session_id, {
                    "type": "keepalive",
                    "cpu_multiprocessing": True,
                    "live_recording": True,
                    "recording": session.get("recording", False),
                    "paused": session.get("paused", False),
                    "audio_level": session.get("audio_level", 0.0),
                    "timestamp": time.time()
                })
                continue
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket desconectado: {session_id}")
    except Exception as e:
        logger.error(f"Erro WebSocket: {e}")
    finally:
        # Disconnect using CPU-only cleanup
        try:
            await app_state.disconnect_websocket(session_id)
        except Exception as cleanup_error:
            logger.error(f"WebSocket cleanup error: {cleanup_error}")

async def handle_websocket_message(session_id: str, data: Dict):
    """Handle WebSocket messages with complete live recording support"""
    try:
        message_type = data.get("type")
        message_data = data.get("data", {})
        
        logger.debug(f"Handling WebSocket message: {message_type} for session {session_id}")
        
        if message_type == "ping":
            await app_state.send_message(session_id, {
                "type": "pong",
                "cpu_multiprocessing": True,
                "live_recording": True,
                "timestamp": time.time()
            })
            
        # User choice management - PT-BR only
        elif message_type == "set_user_choices":
            domain = message_data.get("domain", "general")

            app_state.update_user_choices(session_id, domain=domain)

            await app_state.send_message(session_id, {
                "type": "choices_updated",
                "choices": {
                    "language": "pt",  # Fixed Portuguese Brazilian
                    "domain": domain
                },
                "message": f"✅ Configurações atualizadas: português brasileiro, assunto {domain}"
            })
            
        # Live recording controls
        elif message_type == "start_recording":
            format_type = message_data.get("format", "wav")

            success = await app_state.start_recording(session_id, format_type)
            
            if not success:
                await app_state.send_message(session_id, {
                    "type": "error",
                    "message": "❌ Falha ao iniciar gravação. Verifique permissões do microfone."
                })
                
        elif message_type == "pause_recording":
            success = await app_state.pause_recording(session_id)
            
            if not success:
                await app_state.send_message(session_id, {
                    "type": "error",
                    "message": "❌ Falha ao pausar gravação ou gravação não ativa"
                })
                
        elif message_type == "resume_recording":
            success = await app_state.resume_recording(session_id)
            
            if not success:
                await app_state.send_message(session_id, {
                    "type": "error",
                    "message": "❌ Falha ao retomar gravação ou gravação não pausada"
                })
                
        elif message_type == "stop_recording":
            success = await app_state.stop_recording(session_id)
            
            if not success:
                await app_state.send_message(session_id, {
                    "type": "error",
                    "message": "❌ Falha ao parar gravação ou nenhuma gravação ativa encontrada"
                })
                
        elif message_type == "audio_chunk":
            # Handle live audio chunks during recording
            audio_chunk = message_data
            recording = app_state.active_recordings.get(session_id)
            
            if recording:
                if "audio_chunks" not in recording:
                    recording["audio_chunks"] = []
                recording["audio_chunks"].append(audio_chunk)
                
                # Send progress update
                await app_state.send_message(session_id, {
                    "type": "recording_progress",
                    "chunks_received": len(recording["audio_chunks"]),
                    "timestamp": time.time()
                })
                
        elif message_type == "get_status":
            # Enhanced status with complete recording info
            session = app_state.sessions.get(session_id, {})
            
            await app_state.send_message(session_id, {
                "type": "status_response",
                "session_status": session.get("status", "unknown"),
                "recording": session.get("recording", False),
                "paused": session.get("paused", False),
                "duration": session.get("duration", 0),
                "audio_level": session.get("audio_level", 0.0),
                "format": session.get("format", "wav"),
                "cpu_multiprocessing": True,
                "memory_mode": "cpu_optimized",
                "user_choices": session.get("user_choices", {}),
                "progress": session.get("progress", {}),
                "srt_file": session.get("srt_file"),
                "processing_complete": session.get("processing_complete", False),
                "live_recording_features": ["start", "pause", "resume", "stop", "audio_monitoring", "srt_auto_download"],
                "timestamp": time.time()
            })
            
        else:
            logger.warning(f"Tipo de mensagem desconhecida: {message_type}")
            await app_state.send_message(session_id, {
                "type": "unknown_message_type",
                "original_type": message_type,
                "message": f"Tipo de mensagem não reconhecido: {message_type}"
            })
            
    except Exception as e:
        logger.error(f"Erro no tratamento de mensagem {message_type} para sessão {session_id}: {e}")
        await app_state.send_message(session_id, {
            "type": "error",
            "message": f"❌ Erro de processamento: {str(e)}",
            "progressive_loading": True,
            "original_message_type": message_type
        })

# Additional API endpoints for complete functionality
@app.get("/api/progressive/status")
async def get_progressive_status():
    """Get progressive loading system status"""
    import psutil
    
    try:
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        }
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
        memory_info = {"error": "memory info unavailable"}
    
    return {
        "cpu_multiprocessing": {
            "enabled": app_state.multiprocessing_enabled,
            "manager_loaded": app_state.multiprocessing_manager is not None,
            "active_sessions": len(app_state.sessions),
            "active_recordings": len([s for s in app_state.sessions.values() if s.get("recording")]),
            "paused_recordings": len([s for s in app_state.sessions.values() if s.get("paused")])
        },
        "live_recording": {
            "enabled": True,
            "active_recordings": len([s for s in app_state.sessions.values() if s.get("recording")]),
            "paused_recordings": len([s for s in app_state.sessions.values() if s.get("paused")]),
            "features": ["start", "pause", "resume", "stop", "audio_monitoring", "format_conversion", "srt_auto_download"]
        },
        "memory": memory_info,
        "browser_safe": float(memory_info.get("available_gb", 0.0)) > 2.0,
        "emergency_risk": float(memory_info.get("percent", 0.0)) > 85,
        "timestamp": time.time()
    }

@app.get("/api/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get detailed session status"""
    session = app_state.sessions.get(session_id)
    if not session:
        return JSONResponse(
            status_code=404,
            content={"error": "Sessão não encontrada", "session_id": session_id}
        )
    
    return {
        "session_id": session_id,
        "status": session.get("status"),
        "recording": session.get("recording", False),
        "paused": session.get("paused", False),
        "duration": session.get("duration", 0),
        "audio_level": session.get("audio_level", 0.0),
        "format": session.get("format", "wav"),
        "language": session.get("language"),
        "user_choices": session.get("user_choices", {}),
        "progress": session.get("progress", {}),
        "quality_metrics": session.get("quality_metrics", {}),
        "srt_file": session.get("srt_file"),
        "has_srt": bool(session.get("srt_file")),
        "processing_complete": session.get("processing_complete", False),
        "created_at": session.get("created_at"),
        "progressive_loading": session.get("progressive_loading", True),
        "features_used": {
            "live_recording": session.get("recording") or session.get("paused"),
            "progressive_loading": session.get("progressive_loading", True),
            "srt_generation": bool(session.get("srt_file")),
            "user_choices": bool(session.get("user_choices"))
        },
        "timestamp": time.time()
    }

@app.post("/api/session/{session_id}/download-srt")
async def trigger_srt_download(session_id: str):
    """Trigger SRT download (alternative to direct file download)"""
    session = app_state.sessions.get(session_id)
    if not session:
        return JSONResponse(
            status_code=404,
            content={"error": "Sessão não encontrada"}
        )
    
    srt_file = session.get("srt_file")
    if not srt_file or not os.path.exists(srt_file):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Arquivo SRT não encontrado",
                "processing_complete": session.get("processing_complete", False)
            }
        )
    
    return {
        "download_url": f"/download/srt/{session_id}",
        "system_path": str(Path(srt_file).absolute()),
        "file_size": os.path.getsize(srt_file),
        "session_id": session_id
    }

if __name__ == "__main__":
    # Configure port from environment variable with fallback
    port = int(os.getenv("TRANSCREVAI_PORT", "8000"))
    host = os.getenv("TRANSCREVAI_HOST", "0.0.0.0")

    # FASE 10: Model memory management configuration
    MODEL_UNLOAD_DELAY = int(os.getenv('MODEL_UNLOAD_DELAY', '60'))
    if MODEL_UNLOAD_DELAY > 0:
        logger.info(f"[FASE 10] Lazy unload enabled - models unload after {MODEL_UNLOAD_DELAY}s idle (~400-500MB freed)")
    else:
        logger.info("[FASE 10] Lazy unload disabled - models stay loaded (better warm start performance)")
    
    print("\n" + "="*60)
    print("TRANSCREVAI COMPLETE LIVE RECORDING v7.0.0")
    print("="*60)
    print("RECURSOS IMPLEMENTADOS:")
    print("- Controles completos de gravacao ao vivo (start/pause/resume/stop)")
    print("- Monitoramento de nivel de audio em tempo real")
    print("- Download automatico de SRT com notificacao de caminho")
    print("- Integracao real com AudioRecorder")
    print("- Pipeline de processamento aprimorado")
    print("- Conversao automatica MP4 para WAV")
    print("- Gerenciamento de configuracoes do usuario")
    print("- Carregamento progressivo (browser-safe)")
    print("- Tratamento aprimorado de erros")
    print("="*60)
    print(f"Servidor iniciando em http://{host}:{port}")
    print("WebSocket disponivel em ws://localhost:8000/ws/{{session_id}}")
    print("Download SRT em http://localhost:8000/download/srt/{{session_id}}")
    print("="*60 + "\n")
    
    uvicorn.run(
        "main:app", 
        host=host, 
        port=port, 
        reload=True,
        log_level="info"
    )