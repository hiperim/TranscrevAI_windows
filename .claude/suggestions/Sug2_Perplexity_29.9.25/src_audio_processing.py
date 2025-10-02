"""
TranscrevAI Optimized - Audio Processing Module
Sistema de processamento de áudio browser-safe com integração ao resource manager
"""

import asyncio
import gc
import os
import shutil
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import time
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List
import subprocess
import psutil

# Import our optimized modules
from logging_setup import get_logger, log_performance, log_resource_usage
from resource_manager import get_resource_manager, ResourceStatus
from config import CONFIG

logger = get_logger("transcrevai.audio_processing")

# Lazy imports for heavy dependencies
_static_ffmpeg = None
_librosa = None

def get_static_ffmpeg():
    """Lazy import static_ffmpeg"""
    global _static_ffmpeg
    if _static_ffmpeg is None:
        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
            _static_ffmpeg = static_ffmpeg
            logger.info("static_ffmpeg configured successfully")
        except ImportError:
            logger.info("Using system FFmpeg")
            _static_ffmpeg = True
    return _static_ffmpeg

def get_librosa():
    """Lazy import librosa"""
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
            logger.info("librosa loaded successfully")
        except ImportError as e:
            logger.warning(f"librosa not available: {e}")
            _librosa = None
    return _librosa


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    def __init__(self, message: str, error_type: str = "unknown"):
        self.error_type = error_type
        super().__init__(f"[{error_type}] {message}")


class BrowserSafeAudioRecorder:
    """
    Browser-safe audio recorder with resource management integration
    Prevents browser freezing through progressive processing and memory management
    """
    
    def __init__(self, session_id: str, output_format: str = "wav"):
        self.session_id = session_id
        self.output_format = output_format
        self.sample_rate = CONFIG["audio"]["sample_rate"]
        self.channels = CONFIG["audio"]["channels"]
        self.blocksize = 1024  # Small blocks for low latency
        
        # Recording state
        self.is_recording = False
        self.is_paused = False
        self.recording_data: List[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None
        
        # File management
        self.temp_file = None
        self.output_file = None
        self.recording_start_time = None
        
        # Resource management
        self.resource_manager = get_resource_manager()
        self.memory_reserved = False
        
        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.level_callback: Optional[Callable] = None
        
        logger.info(f"AudioRecorder initialized for session {session_id}")
    
    def set_callbacks(self, progress_callback: Optional[Callable] = None, 
                     level_callback: Optional[Callable] = None):
        """Set callback functions for progress and audio level updates"""
        self.progress_callback = progress_callback
        self.level_callback = level_callback
    
    async def start_recording(self) -> bool:
        """
        Start audio recording with resource management
        
        Returns:
            bool: True if recording started successfully
        """
        if self.is_recording:
            logger.warning(f"Recording already active for session {self.session_id}")
            return False
        
        try:
            # Check memory pressure before starting
            if self.resource_manager.is_memory_pressure_high():
                logger.warning("High memory pressure detected, cannot start recording")
                return False
            
            # Reserve memory for recording
            estimated_memory_mb = self._estimate_recording_memory()
            if not self.resource_manager.reserve_memory(
                f"recording_{self.session_id}", 
                estimated_memory_mb, 
                "audio_recording"
            ):
                logger.error("Failed to reserve memory for recording")
                return False
            
            self.memory_reserved = True
            
            # Prepare output file
            self._prepare_output_file()
            
            # Initialize audio stream
            await self._start_audio_stream()
            
            # Set recording state
            self.is_recording = True
            self.recording_start_time = time.time()
            self.recording_data = []
            
            logger.info(f"Recording started for session {self.session_id}")
            
            # Start monitoring task
            asyncio.create_task(self._monitoring_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            await self._cleanup_resources()
            return False
    
    async def stop_recording(self) -> Optional[str]:
        """
        Stop recording and return the audio file path
        
        Returns:
            str: Path to the recorded audio file, or None if failed
        """
        if not self.is_recording:
            logger.warning(f"No active recording for session {self.session_id}")
            return None
        
        try:
            # Stop recording state
            self.is_recording = False
            recording_duration = time.time() - self.recording_start_time if self.recording_start_time else 0
            
            # Stop audio stream
            await self._stop_audio_stream()
            
            # Process recorded data
            audio_file = await self._process_recording_data()
            
            # Log performance metrics
            log_performance(
                f"Recording completed for session {self.session_id}",
                duration=recording_duration,
                samples=len(self.recording_data),
                format=self.output_format,
                file_size_mb=os.path.getsize(audio_file) / (1024*1024) if audio_file and os.path.exists(audio_file) else 0
            )
            
            logger.info(f"Recording stopped for session {self.session_id}: {audio_file}")
            return audio_file
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return None
        finally:
            await self._cleanup_resources()
    
    async def pause_recording(self) -> bool:
        """Pause recording"""
        if not self.is_recording or self.is_paused:
            return False
        
        self.is_paused = True
        logger.info(f"Recording paused for session {self.session_id}")
        return True
    
    async def resume_recording(self) -> bool:
        """Resume recording"""
        if not self.is_recording or not self.is_paused:
            return False
        
        self.is_paused = False
        logger.info(f"Recording resumed for session {self.session_id}")
        return True
    
    def _estimate_recording_memory(self) -> float:
        """Estimate memory usage for recording"""
        # Conservative estimate: 5 minutes of audio at 16kHz
        max_duration_seconds = 5 * 60
        bytes_per_sample = 4  # float32
        samples_per_second = self.sample_rate * self.channels
        
        estimated_mb = (max_duration_seconds * samples_per_second * bytes_per_sample) / (1024 * 1024)
        return estimated_mb * 1.5  # Add 50% buffer
    
    def _prepare_output_file(self):
        """Prepare output file paths"""
        recordings_dir = Path(CONFIG["paths"]["recordings_dir"])
        recordings_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"recording_{self.session_id}_{timestamp}.{self.output_format}"
        self.output_file = str(recordings_dir / filename)
        
        # Create temporary file for recording
        temp_dir = Path(CONFIG["paths"]["temp_dir"])
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_file = str(temp_dir / f"temp_{filename}")
    
    async def _start_audio_stream(self):
        """Start the audio input stream"""
        def audio_callback(indata, frames, time_info, status):
            """Callback function for audio stream"""
            if status:
                logger.warning(f"Audio stream status: {status}")
            
            if self.is_recording and not self.is_paused:
                if indata is not None and len(indata) > 0:
                    # Copy audio data (prevent memory issues)
                    audio_chunk = indata.copy()
                    
                    # Check for valid audio levels
                    audio_level = np.sqrt(np.mean(audio_chunk**2))
                    
                    # Add to recording data
                    self.recording_data.append(audio_chunk)
                    
                    # Callback for audio level (for UI visualization)
                    if self.level_callback:
                        try:
                            asyncio.create_task(self.level_callback(audio_level))
                        except:
                            pass  # Non-critical, continue recording
        
        try:
            # Get default input device
            default_device = sd.default.device[0] if sd.default.device else None
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                dtype=np.float32,
                device=default_device,
                blocksize=self.blocksize,
                latency='low'
            )
            
            self.stream.start()
            logger.info(f"Audio stream started with device {default_device}")
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to start audio stream: {e}", "stream_error")
    
    async def _stop_audio_stream(self):
        """Stop the audio input stream"""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                logger.debug("Audio stream stopped successfully")
            except Exception as e:
                logger.warning(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None
    
    async def _process_recording_data(self) -> Optional[str]:
        """Process recorded audio data and save to file"""
        if not self.recording_data:
            logger.warning("No audio data recorded")
            return None
        
        try:
            # Concatenate all audio chunks
            audio_array = np.concatenate(self.recording_data, axis=0)
            
            # Ensure proper audio format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Apply audio processing (normalize, filter, etc.)
            audio_array = await self._enhance_audio(audio_array)
            
            # Save to temporary file first
            sf.write(self.temp_file, audio_array, self.sample_rate)
            
            # Convert to final format if needed
            if self.output_format == "mp4":
                await self._convert_to_mp4()
            else:
                # Move temp file to final location
                shutil.move(self.temp_file, self.output_file)
            
            # Validate output file
            if not os.path.exists(self.output_file) or os.path.getsize(self.output_file) == 0:
                raise AudioProcessingError("Output file is invalid", "file_error")
            
            return self.output_file
            
        except Exception as e:
            logger.error(f"Failed to process recording data: {e}")
            return None
    
    async def _enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply audio enhancements for better transcription"""
        try:
            # Normalize audio levels
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                # Normalize to 80% to avoid clipping
                audio_data = audio_data * (0.8 / max_val)
            
            # Apply gentle high-pass filter to reduce noise
            if len(audio_data) > self.sample_rate:  # Only for longer recordings
                try:
                    from scipy import signal
                    # High-pass filter at 85 Hz
                    sos = signal.butter(3, 85, btype='highpass', fs=self.sample_rate, output='sos')
                    audio_data = signal.sosfilt(sos, audio_data)
                except ImportError:
                    logger.debug("scipy not available, skipping filtering")
            
            # Ensure float32 format
            return audio_data.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed, using original: {e}")
            return audio_data
    
    async def _convert_to_mp4(self):
        """Convert WAV to MP4 using FFmpeg"""
        get_static_ffmpeg()  # Ensure FFmpeg is available
        
        ffmpeg_cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-i", self.temp_file,
            "-f", "mp4",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-movflags", "frag_keyframe+empty_moov",
            self.output_file
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
            
            if process.returncode != 0:
                raise AudioProcessingError(f"FFmpeg conversion failed: {stderr.decode()}", "conversion_error")
            
            logger.info("Audio converted to MP4 successfully")
            
        except asyncio.TimeoutError:
            process.kill()
            raise AudioProcessingError("MP4 conversion timed out", "timeout_error")
        except Exception as e:
            raise AudioProcessingError(f"MP4 conversion failed: {e}", "conversion_error")
        finally:
            # Clean up temp file
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
    
    async def _monitoring_loop(self):
        """Monitor recording and provide progress updates"""
        try:
            while self.is_recording:
                # Check memory pressure
                if self.resource_manager.is_memory_pressure_high():
                    logger.warning("High memory pressure during recording")
                    if self.progress_callback:
                        await self.progress_callback("memory_pressure", {
                            "message": "Pressão de memória alta detectada"
                        })
                
                # Progress update
                if self.progress_callback and self.recording_start_time:
                    duration = time.time() - self.recording_start_time
                    await self.progress_callback("recording_progress", {
                        "duration": duration,
                        "samples": len(self.recording_data),
                        "is_paused": self.is_paused
                    })
                
                await asyncio.sleep(1.0)  # Update every second
                
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
    
    async def _cleanup_resources(self):
        """Clean up resources and memory"""
        try:
            # Release memory reservation
            if self.memory_reserved:
                self.resource_manager.release_memory_reservation(f"recording_{self.session_id}")
                self.memory_reserved = False
            
            # Clear recording data
            self.recording_data = []
            
            # Remove temporary files
            if self.temp_file and os.path.exists(self.temp_file):
                os.remove(self.temp_file)
            
            # Force garbage collection
            gc.collect()
            
            logger.debug(f"Resources cleaned up for session {self.session_id}")
            
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")


class AudioFileProcessor:
    """
    Process uploaded audio files with format validation and optimization
    """
    
    def __init__(self):
        self.resource_manager = get_resource_manager()
        self.supported_formats = CONFIG["audio"]["supported_formats"]
        self.max_file_size = CONFIG["audio"]["max_file_size_mb"] * 1024 * 1024
        
    async def process_uploaded_file(self, file_path: str, session_id: str) -> Dict[str, Any]:
        """
        Process uploaded audio file
        
        Args:
            file_path: Path to uploaded file
            session_id: Session identifier
            
        Returns:
            Dict with processing results
        """
        try:
            # Validate file
            validation_result = await self._validate_audio_file(file_path)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["error"]}
            
            # Get file info
            file_info = await self._get_audio_info(file_path)
            
            # Check if conversion is needed
            output_path = file_path
            if self._needs_conversion(file_path):
                output_path = await self._convert_to_optimized_format(file_path, session_id)
                if not output_path:
                    return {"success": False, "error": "Failed to convert audio file"}
            
            return {
                "success": True,
                "output_path": output_path,
                "info": file_info,
                "converted": output_path != file_path
            }
            
        except Exception as e:
            logger.error(f"Failed to process uploaded file: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Validate audio file format and size"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {"valid": False, "error": "File not found"}
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return {"valid": False, "error": f"File too large (max {self.max_file_size // (1024*1024)}MB)"}
            
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return {"valid": False, "error": f"Unsupported format: {file_ext}"}
            
            # Try to read audio file header
            try:
                info = sf.info(file_path)
                if info.frames == 0:
                    return {"valid": False, "error": "Empty audio file"}
            except Exception:
                # Try with librosa as fallback
                librosa = get_librosa()
                if librosa:
                    try:
                        _, sr = librosa.load(file_path, sr=None, duration=1.0)
                        if sr is None:
                            return {"valid": False, "error": "Invalid audio format"}
                    except Exception as e:
                        return {"valid": False, "error": f"Cannot read audio file: {e}"}
                else:
                    return {"valid": False, "error": "Cannot validate audio file"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {e}"}
    
    async def _get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """Get audio file information"""
        try:
            info = sf.info(file_path)
            duration = info.frames / info.samplerate
            
            return {
                "duration": duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype,
                "file_size": os.path.getsize(file_path)
            }
            
        except Exception:
            # Fallback with librosa
            librosa = get_librosa()
            if librosa:
                try:
                    duration = librosa.get_duration(filename=file_path)
                    y, sr = librosa.load(file_path, sr=None, duration=1.0)
                    
                    return {
                        "duration": duration,
                        "sample_rate": sr,
                        "channels": 1 if y.ndim == 1 else y.shape[0],
                        "frames": int(duration * sr),
                        "format": "unknown",
                        "subtype": "unknown",
                        "file_size": os.path.getsize(file_path)
                    }
                except Exception as e:
                    logger.warning(f"Failed to get audio info with librosa: {e}")
            
            return {
                "duration": 0,
                "sample_rate": 16000,
                "channels": 1,
                "frames": 0,
                "format": "unknown",
                "subtype": "unknown",
                "file_size": os.path.getsize(file_path)
            }
    
    def _needs_conversion(self, file_path: str) -> bool:
        """Check if file needs conversion for optimal processing"""
        try:
            info = sf.info(file_path)
            
            # Convert if not optimal sample rate or channels
            if info.samplerate != 16000 or info.channels != 1:
                return True
            
            # Convert if not WAV format (for processing efficiency)
            if not file_path.lower().endswith('.wav'):
                return True
            
            return False
            
        except Exception:
            # If we can't read the file info, assume conversion is needed
            return True
    
    async def _convert_to_optimized_format(self, input_path: str, session_id: str) -> Optional[str]:
        """Convert audio file to optimized format for processing"""
        try:
            # Reserve memory for conversion
            file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            if not self.resource_manager.reserve_memory(
                f"conversion_{session_id}",
                file_size_mb * 2,  # Conservative estimate
                "audio_conversion"
            ):
                logger.warning("Could not reserve memory for conversion")
            
            # Create output path
            temp_dir = Path(CONFIG["paths"]["temp_dir"])
            output_path = temp_dir / f"converted_{session_id}_{int(time.time())}.wav"
            
            get_static_ffmpeg()  # Ensure FFmpeg is available
            
            # FFmpeg conversion command
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",      # Mono
                "-acodec", "pcm_s16le",
                "-f", "wav",
                str(output_path)
            ]
            
            # Run conversion
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
            
            if process.returncode != 0:
                logger.error(f"FFmpeg conversion failed: {stderr.decode()}")
                return None
            
            # Validate converted file
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error("Conversion produced invalid file")
                return None
            
            logger.info(f"Audio file converted successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None
        finally:
            # Release memory reservation
            self.resource_manager.release_memory_reservation(f"conversion_{session_id}")


class AudioProcessor:
    """
    Main audio processor class that coordinates recording and file processing
    Browser-safe implementation with resource management integration
    """
    
    def __init__(self):
        self.active_recordings: Dict[str, BrowserSafeAudioRecorder] = {}
        self.file_processor = AudioFileProcessor()
        self.resource_manager = get_resource_manager()
        
        logger.info("AudioProcessor initialized")
    
    async def start_recording(self, session_id: str, output_format: str = "wav",
                            progress_callback: Optional[Callable] = None,
                            level_callback: Optional[Callable] = None) -> bool:
        """
        Start recording for a session
        
        Args:
            session_id: Session identifier
            output_format: Output format ('wav' or 'mp4')
            progress_callback: Callback for progress updates
            level_callback: Callback for audio level updates
            
        Returns:
            bool: True if recording started successfully
        """
        if session_id in self.active_recordings:
            logger.warning(f"Recording already active for session {session_id}")
            return False
        
        try:
            # Create recorder
            recorder = BrowserSafeAudioRecorder(session_id, output_format)
            recorder.set_callbacks(progress_callback, level_callback)
            
            # Start recording
            success = await recorder.start_recording()
            
            if success:
                self.active_recordings[session_id] = recorder
                
                # Log resource usage
                log_resource_usage(
                    f"recording_start_{session_id}",
                    self.resource_manager.get_current_metrics().memory_used_mb,
                    self.resource_manager.get_current_metrics().cpu_percent * 100,
                    active_recordings=len(self.active_recordings)
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start recording for session {session_id}: {e}")
            return False
    
    async def stop_recording(self, session_id: str) -> Optional[str]:
        """
        Stop recording for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            str: Path to recorded file, or None if failed
        """
        recorder = self.active_recordings.get(session_id)
        if not recorder:
            logger.warning(f"No active recording for session {session_id}")
            return None
        
        try:
            # Stop recording
            audio_file = await recorder.stop_recording()
            
            # Remove from active recordings
            del self.active_recordings[session_id]
            
            # Log resource usage
            log_resource_usage(
                f"recording_stop_{session_id}",
                self.resource_manager.get_current_metrics().memory_used_mb,
                self.resource_manager.get_current_metrics().cpu_percent * 100,
                active_recordings=len(self.active_recordings),
                output_file=audio_file
            )
            
            return audio_file
            
        except Exception as e:
            logger.error(f"Failed to stop recording for session {session_id}: {e}")
            # Cleanup on error
            if session_id in self.active_recordings:
                del self.active_recordings[session_id]
            return None
    
    async def pause_recording(self, session_id: str) -> bool:
        """Pause recording for a session"""
        recorder = self.active_recordings.get(session_id)
        if recorder:
            return await recorder.pause_recording()
        return False
    
    async def resume_recording(self, session_id: str) -> bool:
        """Resume recording for a session"""
        recorder = self.active_recordings.get(session_id)
        if recorder:
            return await recorder.resume_recording()
        return False
    
    async def process_uploaded_file(self, file_path: str, session_id: str) -> Dict[str, Any]:
        """
        Process an uploaded audio file
        
        Args:
            file_path: Path to uploaded file
            session_id: Session identifier
            
        Returns:
            Dict with processing results
        """
        return await self.file_processor.process_uploaded_file(file_path, session_id)
    
    async def get_audio_duration(self, file_path: str) -> float:
        """Get audio file duration in seconds"""
        try:
            info = await self.file_processor._get_audio_info(file_path)
            return info.get("duration", 0.0)
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
            return 0.0
    
    def get_active_recordings(self) -> List[str]:
        """Get list of active recording session IDs"""
        return list(self.active_recordings.keys())
    
    async def cleanup_session(self, session_id: str):
        """Clean up resources for a session"""
        recorder = self.active_recordings.get(session_id)
        if recorder:
            try:
                if recorder.is_recording:
                    await recorder.stop_recording()
                await recorder._cleanup_resources()
                del self.active_recordings[session_id]
                logger.info(f"Cleaned up audio resources for session {session_id}")
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
    
    async def cleanup_all_sessions(self):
        """Clean up all active sessions"""
        session_ids = list(self.active_recordings.keys())
        for session_id in session_ids:
            await self.cleanup_session(session_id)


# Utility functions for external use
def get_supported_formats() -> List[str]:
    """Get list of supported audio formats"""
    return CONFIG["audio"]["supported_formats"]


def estimate_processing_time(audio_duration: float, is_warm_start: bool = False) -> float:
    """
    Estimate processing time based on audio duration
    
    Args:
        audio_duration: Duration of audio in seconds
        is_warm_start: Whether model is already cached
        
    Returns:
        float: Estimated processing time in seconds
    """
    if is_warm_start:
        ratio = CONFIG["performance"]["targets"]["processing_ratio_warm"]
    else:
        ratio = CONFIG["performance"]["targets"]["processing_ratio_cold"]
    
    return audio_duration * ratio