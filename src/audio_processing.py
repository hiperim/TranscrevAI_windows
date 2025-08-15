import asyncio
import subprocess
import logging
import tempfile
import sys
import os
from pathlib import Path
import time
from enum import Enum
import shutil
import psutil
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import Optional

# Import Windows-specific modules only if available
try:
    import win32file
    import win32con
    import pywintypes
    from ctypes import windll
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

from src.logging_setup import setup_app_logging
from src.file_manager import FileManager
from config.app_config import APP_PACKAGE_NAME

# Use proper logging setup
logger = setup_app_logging(logger_name="transcrevai.audio_processing")

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    
    class ErrorType(Enum):
        FILE_ACCESS = "file_access"
        FILE_OPERATION = "file_operation"
        RECORDING_FAILED = "recording_failed"
        CONVERSION_FAILED = "conversion_failed"
        SYSTEM_ERROR = "system_error"
        INVALID_FORMAT = "invalid_format"
        TIMEOUT = "timeout"
    
    def __init__(self, message: str, error_type: ErrorType):
        self.error_type = error_type
        super().__init__(f"{error_type.value}: {message}")

class CrossPlatformFileHandler:
    """Cross-platform file handling with fallback methods"""
    
    @staticmethod
    async def safe_atomic_move(temp_path: str, final_path: str) -> bool:
        """Safely move file with cross-platform atomic operations"""
        try:
            temp_path_obj = Path(temp_path)
            final_path_obj = Path(final_path)
            
            # Ensure destination directory exists
            final_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Try Windows-specific atomic move if available
            if WINDOWS_AVAILABLE and sys.platform == "win32":
                return await CrossPlatformFileHandler._windows_atomic_move(temp_path_obj, final_path_obj)
            else:
                return await CrossPlatformFileHandler._posix_atomic_move(temp_path_obj, final_path_obj)
                
        except Exception as e:
            logger.error(f"Atomic move failed: {e}")
            return False
    
    @staticmethod
    async def _windows_atomic_move(temp_path: Path, final_path: Path) -> bool:
        """Windows-specific atomic file move"""
        if not WINDOWS_AVAILABLE:
            return await CrossPlatformFileHandler._posix_atomic_move(temp_path, final_path)
        
        try:
            # Kill any processes that might be locking the files
            await CrossPlatformFileHandler._kill_file_locks(temp_path, final_path)
            
            # Use Windows API for atomic move with correct flags
            MOVEFILE_REPLACE_EXISTING = 0x1
            MOVEFILE_WRITE_THROUGH = 0x8
            
            success = windll.kernel32.MoveFileExW(
                str(temp_path),
                str(final_path),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
            )
            
            if success:
                # Force filesystem sync
                await CrossPlatformFileHandler._force_file_sync(final_path)
                return True
            else:
                logger.warning(f"Windows MoveFileEx failed, falling back to standard move")
                return await CrossPlatformFileHandler._posix_atomic_move(temp_path, final_path)
                
        except Exception as e:
            logger.warning(f"Windows atomic move failed: {e}, falling back")
            return await CrossPlatformFileHandler._posix_atomic_move(temp_path, final_path)
    
    @staticmethod
    async def _posix_atomic_move(temp_path: Path, final_path: Path) -> bool:
        """POSIX-compatible atomic file move"""
        try:
            # Use asyncio.to_thread for blocking operations
            await asyncio.to_thread(shutil.move, str(temp_path), str(final_path))
            
            # Verify the move was successful
            if final_path.exists() and final_path.stat().st_size > 0:
                return True
            else:
                logger.error("Move succeeded but file validation failed")
                return False
                
        except Exception as e:
            logger.error(f"POSIX atomic move failed: {e}")
            return False
    
    @staticmethod
    async def _kill_file_locks(temp_path: Path, final_path: Path):
        """Kill processes that might be locking files"""
        try:
            # Kill any FFmpeg processes that might be locking our files
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if any(x in proc.name().lower() for x in ['ffmpeg', 'ffprobe']):
                        proc.kill()
                        logger.debug(f"Killed process {proc.pid}: {proc.name()}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.debug(f"Error killing file locks: {e}")
    
    @staticmethod
    async def _force_file_sync(path: Path):
        """Force file synchronization to disk"""
        if WINDOWS_AVAILABLE and sys.platform == "win32":
            try:
                handle = win32file.CreateFile(
                    str(path),
                    win32con.GENERIC_READ,
                    0,
                    None,
                    win32con.OPEN_EXISTING,
                    0,
                    None
                )
                windll.kernel32.FlushFileBuffers(handle.handle)
                win32file.CloseHandle(handle.handle)
            except Exception as e:
                logger.debug(f"Windows file sync failed: {e}")
        else:
            try:
                # POSIX: open the file and fsync its file descriptor
                with open(path, "rb") as f:
                    await asyncio.to_thread(os.fsync, f.fileno())
            except Exception as e:
                logger.debug(f"POSIX file sync failed: {e}")
    
    @staticmethod
    async def safe_delete(file_path: str) -> bool:
        """Safely delete with retry logic"""
        path = Path(file_path)
        if not path.exists():
            return True
        
        for attempt in range(5):
            try:
                await asyncio.to_thread(path.unlink)
                return True
            except Exception as e:
                logger.warning(f"Delete attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
        
        logger.error(f"Failed to delete file after 5 attempts: {file_path}")
        return False

class AtomicAudioFile:
    """Improved atomic file context manager with cross-platform support"""
    
    def __init__(self, extension: str = ".wav"):
        self.final_path: Optional[str] = None
        self.temp_path = self._create_temp(extension)
        self._committed = False
    
    def _create_temp(self, extension: str) -> str:
        """Create a temporary file path"""
        temp_dir = FileManager.get_data_path("temp")
        FileManager.ensure_directory_exists(temp_dir)
        return os.path.join(
            temp_dir,
            f"atomic_temp_{os.getpid()}_{int(time.time()*1000)}{extension}"
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async cleanup - no new event loops needed"""
        try:
            if self._committed and self.final_path:
                await self._async_commit()
            else:
                await CrossPlatformFileHandler.safe_delete(self.temp_path)
        except Exception as e:
            logger.error(f"Error in AtomicAudioFile cleanup: {e}")
            # Ensure temp file is cleaned up even if commit fails
            if os.path.exists(self.temp_path):
                try:
                    await CrossPlatformFileHandler.safe_delete(self.temp_path)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup temp file: {cleanup_error}")
    
    def commit(self, final_path: str):
        """Mark file for preservation with atomic replacement"""
        self.final_path = final_path
        self._committed = True
    
    async def _async_commit(self):
        """Perform the actual atomic commit operation"""
        if self.final_path and self.temp_path:
            success = await CrossPlatformFileHandler.safe_atomic_move(
                self.temp_path, self.final_path
            )
            
            if not success:
                raise AudioProcessingError(
                    f"Failed to commit atomic file operation: {self.final_path}",
                    AudioProcessingError.ErrorType.FILE_OPERATION
                )

class AudioRecorder:
    """Enhanced audio recorder with proper async support and resource management"""
    
    def __init__(self, output_file: Optional[str] = None, sample_rate: int = 16000):
        self.output_file = output_file or os.path.join(
            FileManager.get_data_path("recordings"),
            f"recording_{int(time.time())}.wav"
        )
        
        self.temp_wav = self.get_temp_path()
        self.wav_file = self.temp_wav
        self.sample_rate = sample_rate
        self.is_recording = False
        self._stream: Optional[sd.InputStream] = None
        self._frames = []
        self._recording_start_time: Optional[float] = None
        self.is_paused = False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Verify audio device availability (non-blocking)
        self._verify_audio_device()
        
        # Verify FFmpeg availability (non-blocking)
        self._verify_ffmpeg()
    
    def _verify_audio_device(self):
        """Verify audio input device availability"""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
            if input_devices:
                logger.info(f"Found {len(input_devices)} audio input devices")
            else:
                logger.warning("No audio input devices found")
        except Exception as e:
            logger.warning(f"Audio device verification failed: {e}")
    
    def _verify_ffmpeg(self):
        """Verify FFmpeg availability"""
        try:
            if FFMPEG_AVAILABLE:
                subprocess.run(
                    ["ffmpeg", "-version"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10
                )
                logger.info("FFmpeg verified via static_ffmpeg")
            else:
                subprocess.run(
                    ["ffmpeg", "-version"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10
                )
                logger.info("System FFmpeg verified")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"FFmpeg verification failed: {e}")
    
    def get_temp_path(self, extension: str = ".wav") -> str:
        """Generate a temporary file path"""
        temp_dir = FileManager.get_data_path("temp")
        FileManager.ensure_directory_exists(temp_dir)
        return os.path.normpath(os.path.join(
            temp_dir,
            f"temp_recording_{int(time.time()*1000)}{extension}"
        ))
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_recording()
        await self.cleanup_resources()
    
    def _check_system_resources(self):
        """Check system resources before recording"""
        try:
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage > 95:
                logger.warning(f"High CPU usage detected: {cpu_usage}%")
            
            # Check disk space
            free_space = shutil.disk_usage(os.path.dirname(self.output_file)).free / (1024 ** 3)
            if free_space < 0.1:  # Less than 100MB
                raise IOError(f"Insufficient disk space: {free_space:.2f}GB available")
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
    
    async def start_recording(self):
        """Start audio recording"""
        try:
            self._check_system_resources()
            
            self.is_recording = True
            self._frames = []
            self._recording_start_time = time.time()
            
            await self._start_audio_stream()
            
            logger.info("Audio recording started")
            
        except Exception as e:
            self.is_recording = False
            raise AudioProcessingError(
                f"Failed to start recording: {str(e)}",
                AudioProcessingError.ErrorType.RECORDING_FAILED
            )
    
    async def _start_audio_stream(self):
        """Initialize and start the audio input stream"""
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio stream status: {status}")
            
            if self.is_recording and not self.is_paused:
                # Ensure we have valid audio data
                if indata is not None and len(indata) > 0:
                    # Check for valid audio levels (not all zeros)
                    if np.any(np.abs(indata) > 0.001):  # Threshold for silence
                        self._frames.append(indata.copy())
                    else:
                        # Still append to maintain timing, but log silence
                        self._frames.append(indata.copy())
        
        try:
            # Get default input device
            try:
                default_device = sd.default.device[0] if sd.default.device else None
            except:
                default_device = None
            
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                dtype=np.float32,
                device=default_device,
                blocksize=1024,  # Smaller block size for better responsiveness
                latency='low'
            )
            
            self._stream.start()
            logger.info(f"Audio stream started with device {default_device}")
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to initialize audio stream: {str(e)}",
                AudioProcessingError.ErrorType.SYSTEM_ERROR
            )
    
    def get_recent_audio_chunk(self, duration_seconds: float = 2.0) -> Optional[np.ndarray]:
        """
        Get recent audio chunk for real-time processing.
        
        Args:
            duration_seconds: Duration of audio chunk to return
            
        Returns:
            Recent audio chunk as numpy array or None if insufficient data
        """
        if not self.is_recording or not self._frames:
            return None
        
        # Calculate required frames for duration
        frames_needed = int(duration_seconds * self.sample_rate)
        
        # Get recent frames
        with self._record_lock:
            if len(self._frames) == 0:
                return None
            
            # Convert list of frame chunks to single array
            all_frames = np.concatenate(self._frames, axis=0) if self._frames else np.array([])
            
            if len(all_frames) < frames_needed:
                # Return all available frames if we don't have enough
                return all_frames.flatten() if len(all_frames) > 0 else None
            
            # Return the most recent frames
            recent_frames = all_frames[-frames_needed:]
            return recent_frames.flatten()

    def pause_recording(self):
        """Pause the recording"""
        if not self.is_recording:
            logger.warning("Cannot pause: recording is not active")
            return
        
        self.is_paused = True
        logger.info("Recording paused")
    
    def resume_recording(self):
        """Resume the recording"""
        if not self.is_recording:
            logger.warning("Cannot resume: recording is not active")
            return
        
        self.is_paused = False
        logger.info("Recording resumed")
    
    async def stop_recording(self) -> bool:
        """Stop recording and save audio file"""
        if not self.is_recording:
            logger.warning("Recording is not active")
            return False
        
        try:
            self.is_recording = False
            
            # Stop audio stream with proper error handling
            if self._stream:
                try:
                    self._stream.stop()
                except Exception as stop_error:
                    logger.warning(f"Error stopping audio stream: {stop_error}")
                
                try:
                    self._stream.close()
                except Exception as close_error:
                    logger.warning(f"Error closing audio stream: {close_error}")
                finally:
                    self._stream = None
            
            # Process recorded audio
            await self._save_recorded_audio()
            
            # Convert to final format if needed
            if self.output_file.endswith('.mp4'):
                await self._convert_to_mp4()
            
            # Validate output
            await self._validate_output()
            
            logger.info(f"Recording stopped and saved: {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping recording: {str(e)}")
            raise AudioProcessingError(
                f"Failed to stop recording: {str(e)}",
                AudioProcessingError.ErrorType.RECORDING_FAILED
            )
        finally:
            await self.cleanup_resources()
    
    async def _save_recorded_audio(self):
        """Save recorded audio frames to file"""
        async with AtomicAudioFile() as temp_ctx:
            try:
                if self._frames and len(self._frames) > 0:
                    # Concatenate all audio frames
                    audio_data = np.concatenate(self._frames)
                    
                    # Check if we have meaningful audio data
                    if len(audio_data) > 0:
                        # Normalize audio levels
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 0:
                            audio_data = audio_data / max_val * 0.8  # Normalize to 80% to avoid clipping
                        
                        # Save to temporary file
                        sf.write(temp_ctx.temp_path, audio_data, self.sample_rate)
                        sf.write(self.temp_wav, audio_data, self.sample_rate)
                        
                        logger.info(f"Saved {len(audio_data)} audio samples ({len(audio_data)/self.sample_rate:.2f} seconds)")
                    else:
                        raise AudioProcessingError(
                            "No audio data recorded",
                            AudioProcessingError.ErrorType.RECORDING_FAILED
                        )
                else:
                    # Create minimal audio file instead of empty
                    silence_duration = 0.5  # 500ms of silence
                    empty_audio = np.zeros((int(self.sample_rate * silence_duration),), dtype=np.float32)
                    
                    sf.write(temp_ctx.temp_path, empty_audio, self.sample_rate)
                    sf.write(self.temp_wav, empty_audio, self.sample_rate)
                    
                    logger.warning("No audio frames recorded, created minimal audio file")
                
                # Validate temporary file
                if not os.path.exists(temp_ctx.temp_path) or os.path.getsize(temp_ctx.temp_path) == 0:
                    raise AudioProcessingError(
                        "Temporary audio file is missing or empty",
                        AudioProcessingError.ErrorType.FILE_OPERATION
                    )
                
                # Commit to final location
                temp_ctx.commit(self.output_file)
                self.wav_file = self.output_file
                
            except Exception as e:
                raise AudioProcessingError(
                    f"Failed to save audio: {str(e)}",
                    AudioProcessingError.ErrorType.FILE_OPERATION
                )
    
    async def _convert_to_mp4(self):
        """Convert WAV to MP4 using FFmpeg with timeout and cleanup"""
        ffmpeg_args = [
            "ffmpeg", "-y",
            "-i", self.temp_wav,
            "-f", "mp4",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "frag_keyframe+empty_moov",
            self.output_file
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                await asyncio.wait_for(process.communicate(), timeout=120)
            except asyncio.TimeoutError:
                process.kill()
                raise AudioProcessingError(
                    "MP4 conversion timed out",
                    AudioProcessingError.ErrorType.TIMEOUT
                )
            
            if process.returncode != 0:
                raise AudioProcessingError(
                    f"FFmpeg failed with code {process.returncode}",
                    AudioProcessingError.ErrorType.CONVERSION_FAILED
                )
            
            logger.info("Audio conversion to MP4 completed successfully")
            
        except AudioProcessingError:
            raise
        except Exception as e:
            raise AudioProcessingError(
                f"Conversion error: {str(e)}",
                AudioProcessingError.ErrorType.CONVERSION_FAILED
            )
    
    async def _validate_output(self):
        """Validate the output audio file"""
        if not os.path.exists(self.output_file):
            raise AudioProcessingError(
                f"Output file not found: {self.output_file}",
                AudioProcessingError.ErrorType.FILE_OPERATION
            )
        
        if os.path.getsize(self.output_file) == 0:
            raise AudioProcessingError(
                "Output file is empty",
                AudioProcessingError.ErrorType.INVALID_FORMAT
            )
        
        # Validate audio content
        try:
            if self.output_file.endswith('.wav'):
                with sf.SoundFile(self.output_file) as f:
                    if f.frames == 0:
                        raise AudioProcessingError(
                            "Audio file contains no audio data",
                            AudioProcessingError.ErrorType.INVALID_FORMAT
                        )
        except Exception as e:
            logger.warning(f"Audio validation warning: {e}")
    
    async def cleanup_resources(self):
        """Public cleanup method"""
        await self._cleanup_resources()
    
    async def _cleanup_resources(self):
        """Cleanup temporary files and resources with proper async handling"""
        try:
            # Clean up temporary files
            temp_files = [self.temp_wav] if hasattr(self, 'temp_wav') and self.temp_wav else []
            
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    await CrossPlatformFileHandler.safe_delete(temp_file)
            
            # Clear audio data safely
            if hasattr(self, '_frames'):
                self._frames = []
            
            # Reset stream with proper error handling
            if hasattr(self, '_stream') and self._stream:
                try:
                    if hasattr(self._stream, 'stop'):
                        self._stream.stop()
                except Exception as stop_error:
                    logger.debug(f"Error stopping stream during cleanup: {stop_error}")
                
                try:
                    if hasattr(self._stream, 'close'):
                        self._stream.close()
                except Exception as close_error:
                    logger.debug(f"Error closing stream during cleanup: {close_error}")
                finally:
                    self._stream = None
                    
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
    
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Get audio file duration"""
        try:
            if file_path.endswith('.wav'):
                with sf.SoundFile(file_path) as f:
                    return float(f.frames) / f.samplerate
            else:
                # Use FFprobe for other formats
                cmd = [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of",
                    "default=noprint_wrappers=1:nokey=1", file_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
                return float(result.stdout.strip())
                
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            raise AudioProcessingError(
                f"Cannot determine audio duration: {str(e)}",
                AudioProcessingError.ErrorType.INVALID_FORMAT
            )