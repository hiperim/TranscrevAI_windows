import asyncio
import subprocess
import logging
import tempfile
import sys
import os
from pathlib import Path
import static_ffmpeg
import wave
import time
from enum import Enum
from ctypes import windll
from src.logging_setup import setup_app_logging
from src.file_manager import FileManager
from config.app_config import APP_PACKAGE_NAME
import sounddevice as sd
import soundfile as sf
import numpy as np
import shutil
import psutil
import win32file
import win32con
import pywintypes

logger = setup_app_logging()

class AudioProcessingError(Exception):
    class ErrorType(Enum):
        FILE_ACCESS = "file_access"
        FILE_OPERATION = "file_operation"
        RECORDING_FAILED = "recording_failed"
        CONVERSION_FAILED = "conversion_failed"
        SYSTEM_ERROR = "system_error"
        INVALID_FORMAT = "invalid_format"
    
    def __init__(self, message, error_type):
        self.error_type = (error_type if isinstance(error_type, self.ErrorType) else self.ErrorType(error_type))
        super().__init__(f"{self.error_type.value}: {message}")


class AtomicAudioFile:
    def __init__(self, extension=".wav"):
        self.final_path = None
        self.temp_path = self._create_temp(extension)
        self._committed = False  # Track commit state
    
    def _create_temp(self, extension):
        """Create a temporary file path using the project's temp directory"""
        temp_dir = FileManager.get_data_path("temp")
        FileManager.ensure_directory_exists(temp_dir)
        return os.path.join(temp_dir, f"atomic_temp_{os.getpid()}_{int(time.time()*1000)}{extension}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._committed and self.final_path:
                # Final atomic move
                self._atomic_replace()
        finally:
            if not self._committed:
                self._safe_delete(self.temp_path)

    def commit(self, final_path):
        # Mark file for preservation with atomic replacement
        self.final_path = final_path
        self._committed = True

    def _atomic_replace(self):
        output_dir = os.path.dirname(self.final_path)
        if not os.path.exists(output_dir):
            FileManager.ensure_directory_exists(output_dir)
        # Windows-safe atomic replacement
        for attempt in range(5):
            try:
                # Use low-level API for handle management
                self._windows_atomic_replace()
                if os.path.exists(self.final_path):
                    break    
            except (PermissionError, FileNotFoundError, OSError) as e:
                logger.warning(f"Replace attempt {attempt} failed: {str(e)}")
                time.sleep(0.2 * (attempt + 1))

    def _windows_atomic_replace(self):
        temp_path = Path(self.temp_path)
        final_path = Path(self.final_path)
        MOVEFILE_WRITE_THROUGH = 0x00000008
        for attempt in range(5):
            try:
                # First process termination
                self._kill_all_handles(temp_path, final_path)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                # Atomic replacement with error
                success = windll.kernel32.MoveFileExW(str(temp_path), str(final_path), win32con.MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) # 0x1 | 0x8 = 0x9
                if not success:
                    error_code = windll.kernel32.GetLastError()
                    logger.warning(f"MoveFile failed with error code {error_code} on attempt {attempt+1}")
                    raise OSError(f"MoveFile error {error_code}")
                # Force immediate filesystem commit
                self._force_file_sync(final_path)
                # Add validation check with retry
                if self._validate_atomic_success(temp_path, final_path):
                    logger.info(f"Atomic replace validated on attempt {attempt}")
                    return
            except OSError as e:
                logger.warning(f"Atomic replace attempt {attempt} failed: {e}")
                time.sleep(min(2, 0.25 * (attempt + 1)))  # Capped backoff and retry
        # Final validation attempt with fallback
        try:
            shutil.move(temp_path, final_path)
        except Exception as e:
            raise AudioProcessingError(f"Final fallback failed: {e}", AudioProcessingError.ErrorType.FILE_OPERATION)

    def _kill_all_handles(self, temp_path, final_path):
        # Windows-specific handle cleanup
        try:
            subprocess.run(["powershell", f"Get-Process *ffmpeg*,*ffprobe* | Where-Object {{ "f"($_.Path -like '*{temp_path.name}*') -or "f"($_.Path -like '*{final_path.name}*') "f"}} | Stop-Process -Force -ErrorAction SilentlyContinue"], shell=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as e:
            logger.debug(f"Cleanup error: {e}") 

    def _force_file_sync(self, path):
        # Robust file synchronization
        try:
            handle = win32file.CreateFile(str(path), win32con.GENERIC_READ, 0, None, win32con.OPEN_EXISTING, 0, None)
            try:
                int_handle = handle.handle
                windll.kernel32.FlushFileBuffers(int_handle)
            finally:
                win32file.CloseHandle(handle)
        except (pywintypes.error, TypeError) as e:
            logger.debug(f"File sync error: {str(e)}")
    
    def _validate_atomic_success(self, temp_path, final_path):
        # Non-destructive validation of atomic file operation success
        try:
            # Check if temp file was removed
            if os.path.exists(temp_path):
                logger.debug(f"Validation failed: Temp file still exists at {temp_path}")
                return False
            # Check if final file exists, with content
            if not os.path.exists(final_path):
                logger.debug(f"Validation failed: Final file missing at {final_path}")
                return False
            # Check file has content and valid header
            if os.path.getsize(final_path) < 1024:
                logger.debug(f"Validation failed: Final file too small: {os.path.getsize(final_path)} bytes")
                return False
            # Non-destructive header check
            with open(final_path, "rb") as f:
                header = f.read(8)
                if header[:4] not in [b"RIFF", b"ftyp"]:
                    logger.debug(f"Validation failed: Invalid file header: {header[:4]}")
                    return False
            return True
        except Exception as e:
            logger.debug(f"Validation exception: {str(e)}")
            return False
            
    @staticmethod
    def _safe_delete(path):
        #Robust deletion with retries and handle checks
        for attempt in range(5):
            if not os.path.exists(path):
                return
            try:
                os.remove(path)
                return
            except Exception as e:
                logger.warning(f"Delete attempt {attempt} failed: {str(e)}")
                time.sleep(0.2 * (attempt + 1))
                if sys.platform == "win32":
                    AudioRecorder._windows_file_removal(path)


class AudioRecorder:

    _active_processes = set()

    def __init__(self, output_file=None, sample_rate=16000):
        self.output_file = output_file or os.path.join(FileManager.get_data_path("recordings"), f"recording_{int(time.time())}.wav")
        self.temp_wav = self.get_temp_path()
        self.wav_file = self.temp_wav  # Single file ref.
        self.sample_rate = sample_rate
        self.is_recording = False
        self._stream = None
        self._frames = []
        self._recording_start_time = None
        self.is_paused = False
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        try:
            if not hasattr(AudioRecorder, '_ffmpeg_initialized'):
                static_ffmpeg.add_paths()
                AudioRecorder._ffmpeg_initialized = True
            if not hasattr(AudioRecorder, '_ffmpeg_verified'):
                subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                AudioRecorder._ffmpeg_verified = True
                logger.info("FFmpeg verified")
        except Exception as e:
            logger.critical(f"FFmpeg setup error: {e}")
            raise RuntimeError("FFmpeg is not installed or not configured correctly.")
    
    def __del__(self):
    # Explicit cleanup of resources when recorder is terminated
        try:
            if hasattr(self, "_stream") and self._stream:
                self._stream.close()
            # Kill any associated proc.
            try:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if any(x in proc.name().lower() for x in ['ffmpeg', 'ffprobe']):
                            logger.debug(f"Terminating audio process {proc.pid}")
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def get_temp_path(self, extension=".wav"):
        temp_dir = FileManager.get_data_path("temp")
        FileManager.ensure_directory_exists(temp_dir)
        return os.path.join(temp_dir, f"temp_recording_{int(time.time()*1000)}{extension}")
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_recording()
        self._cleanup_resources()
        
    def _check_system_resources(self):
        if psutil.cpu_percent() > 90:
            raise RuntimeError("CPU usage too high for recording")
        free_space = shutil.disk_usage(os.path.dirname(self.output_file)).free / (1024 ** 3)
        if free_space < 0.1:  # 100Mb
            raise IOError(f"Insufficient disk space: {free_space:.2f}GB")
            
    async def start_recording(self):
        try:
            self._check_system_resources()
            self.is_recording = True
            self._frames = []
            self._recording_start_time = time.time()
            await self._start_desktop_recording()
        except Exception as e:
            raise AudioProcessingError(str(e), AudioProcessingError.ErrorType.RECORDING_FAILED)
        
    async def _start_desktop_recording(self):
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self._audio_callback)
        self._stream.start()

    def _audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio status: {status}")
        if self.is_recording and not self.is_paused:
            self._frames.append(indata.copy())

    def pause_recording(self):
        if not self.is_recording:
            logger.warning("Cannot pause inactive recording")
            return
        self.is_paused = True
        logger.info("Recording paused")

    def resume_recording(self):
        if not self.is_recording:
            logger.warning("Cannot resume inactive recording")
            return
        self.is_paused = False
        logger.info("Recording resumed")

    async def stop_recording(self):
        preserved_frames = None
        with AtomicAudioFile() as temp_ctx:
            try:
                output_dir = os.path.dirname(self.output_file)
                if not os.path.exists(output_dir):
                    FileManager.ensure_directory_exists(output_dir)
                temp_path = temp_ctx.temp_path
                if not self.is_recording:
                    return
                self.is_recording = False
                # Stop desktop recording
                if self._stream:
                    self._stream.stop()
                    self._stream.close()
                if self._frames:
                    audio_data = np.concatenate(self._frames)
                    sf.write(temp_path, audio_data, self.sample_rate)
                    sf.write(self.temp_wav, audio_data, self.sample_rate)                    
                else:
                    # handle empty recording by creating an empty .wav file
                    empty_audio = np.zeros((1,), dtype=np.float32)
                    sf.write(temp_path, empty_audio, self.sample_rate)
                    sf.write(self.temp_wav, empty_audio, self.sample_rate)
                    logger.info("Empty recording found. Silent .wav created")
                if not os.path.exists(temp_path):
                    raise AudioProcessingError(f"Temporary recording file missing: {temp_path}", AudioProcessingError.ErrorType.RECORDING_FAILED)
                preserved_frames = self._frames.copy()
                temp_ctx.commit(self.output_file)
                self.wav_file = self.output_file
                if self.output_file.endswith('.mp4'):
                    await self._convert_to_mp4()
            except Exception as e:
                logger.error(f"Recording stop failed: {str(e)}")
                raise AudioProcessingError(str(e), AudioProcessingError.ErrorType.RECORDING_FAILED)
            finally:
                if preserved_frames is not None:
                    self._cleanup_resources()
        for _ in range(15):  # 0.5s delays = 7.5s total for file handlers
            if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                break
            await asyncio.sleep(0.5)
        else:
            raise FileNotFoundError(f"Final output never appeared: {self.output_file}")
        if preserved_frames:
                self._validate_output(preserved_frames)
        self._frames = []
        return True
        
    async def _convert_to_mp4(self):
        # Platform-specific FFmpeg configuration
        ffmpeg_args = ["-y", "-hwaccel", "auto", "-i", self.temp_wav]
        # Desktop/Windows streaming optimization
        ffmpeg_args += ["-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "-c:a", "aac", "-b:a", "192k", "-max_muxing_queue_size", "9999", "-fflags", "+genpts+discardcorrupt", "-strict", "experimental"]
        ffmpeg_args.append(self.output_file)
        # Platform-specific subprocess configuration
        kwargs = {}
        kwargs.update(creationflags=subprocess.CREATE_NO_WINDOW)
        for attempt in range(10):
            try:
                with open(self.temp_wav, 'rb') as f:
                    f.read(1024)
                    break
            except (PermissionError, OSError):
                self._kill_processes_locking_file(self.temp_wav)
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                raise AudioProcessingError("Input file locked after 10 attempts", AudioProcessingError.ErrorType.FILE_ACCESS)
        try:
            # Create process
            process = await asyncio.create_subprocess_exec("ffmpeg", *ffmpeg_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs)
        except FileNotFoundError:
            logger.critical("FFmpeg executable not found. Ensure FFmpeg is installed and in the system PATH.")
            raise AudioProcessingError("FFmpeg executable not found", AudioProcessingError.ErrorType.SYSTEM_ERROR)
        try:
            # Unified stream handling
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=45)
            # Enhanced error parsing
            if process.returncode != 0:
                error_output = stderr.decode(errors="replace")
                logger.error(f"FFmpeg error (exit {process.returncode}): {error_output[:1000]}")
                raise AudioProcessingError("Critical conversion failure", AudioProcessingError.ErrorType.CONVERSION_FAILED)
            elif stderr:
                logger.debug(f"FFmpeg info: {stderr.decode()[:500]}")
                error_output = stderr.decode(errors="replace")
                # Only clear error messages as failures
                error_lower = error_output.lower()
                if any(msg in error_lower for msg in ["error:", "invalid", "failed"]) and "error: http" not in error_lower: # Ignore http related mssgs
                    logger.error(f"FFmpeg error: {error_output[:1000]}") # Truncate error output to 1000 characters to avoid excessive log size
                    raise AudioProcessingError("Critical conversion failure", AudioProcessingError.ErrorType.CONVERSION_FAILED)
                else:
                    # Log stderr as info-level for debugging
                    logger.debug(f"FFmpeg output: {error_output[:500]}")
        except asyncio.TimeoutError:
            if process.returncode is None:
                self._terminate_process_tree(process.pid, self.output_file)
            logger.critical("Conversion timeout")
            raise AudioProcessingError("MP4 conversion timeout", AudioProcessingError.ErrorType.TIMEOUT)
        finally:
            # WIndows resource cleanup
            subprocess.run(["powershell", f"Get-Process *ffmpeg*,*ffprobe*"
                                          f"| Where-Object {{$_.Path -like '*{Path(self.output_file).name}*'}}" 
                                          f"| Stop-Process -Force -ErrorAction SilentlyContinue"], 
                                          shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
    
    def _validate_output(self, frames):
        # Validates output file with duration calculation
        if not os.path.exists(self.output_file):
            raise FileNotFoundError(f"Output file missing: {self.output_file}")
        if os.path.getsize(self.output_file) == 0:
            raise ValueError("Empty output file created")
        total_samples = sum(frame.shape[0] for frame in frames) if frames else 0
        source_duration = total_samples / self.sample_rate
        # Get converted duration from output file
        try:
            with sf.SoundFile(self.output_file) as f:
                converted_duration = f.frames / f.samplerate
            # Account for MP4 encoding headers
            tolerance = 0.5 if self.output_file.endswith('.mp4') else 0.15
            if abs(source_duration - converted_duration) > tolerance:
                logger.warning(f"Duration mismatch: Source {source_duration:.2f}s vs Converted {converted_duration:.2f}s")
                # Fall back to FFprobe for more accurate duration
                try:
                    ffprobe_duration = self.get_audio_duration(self.output_file)
                    if abs(source_duration - ffprobe_duration) <= tolerance:
                        return True
                except Exception:
                    pass
                raise ValueError(f"Duration mismatch: Source {source_duration:.1f}s vs Converted {converted_duration:.1f}s")
            return True
        except Exception as e:
            raise AudioProcessingError(f"Validation error: {str(e)}", AudioProcessingError.ErrorType.INVALID_FORMAT)
        
    def _cleanup_resources(self):
        temp_files = [self.temp_wav]
        for path in temp_files:
            if not os.path.exists(path):
                continue
            try:
                self._windows_file_removal(path)
            except Exception as e:
                logger.warning(f"Cleanup warning: {path} - {str(e)}")

    @staticmethod
    def _terminate_process_tree(pid, output_file):
        try:
            subprocess.run(["taskkill", "/F", "/T", "/FI", f"PID eq {pid}", "/IM", "ffmpeg*", "/IM", "ffprobe*"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["powershell", f"Get-ChildItem -Path {Path(output_file).drive}" 
                                          f"-Recurse -File | Where-Object {{$_.Name -like '*{Path(output_file).name}*'}}"
                                          f"| ForEach-Object {{$_.Delete()}}"], shell=True)
        except subprocess.CalledProcessError:
            pass

    @staticmethod
    def _windows_file_removal(path):
        if not path or not os.path.exists(path):
            return
        try:
            for attempt in range(5):
                try:
                    # Try standard removal first
                    os.remove(path)
                    logger.debug(f"Successfully removed file: {path}")
                    return
                except PermissionError:
                    logger.debug(f"Permission error removing {path}, attempt {attempt+1}/5")
                    try:
                        # Try to open with exclusive access to force closure of other handles
                        handle = win32file.CreateFile(path,
                                                      win32con.GENERIC_READ | win32con.GENERIC_WRITE,
                                                      0,  # No sharing
                                                      None,
                                                      win32con.OPEN_EXISTING,
                                                      win32con.FILE_ATTRIBUTE_NORMAL | win32con.FILE_FLAG_DELETE_ON_CLOSE,
                                                      None)
                        win32file.CloseHandle(handle)
                        # File should be deleted when handle is closed
                        return
                    except pywintypes.error as e:
                        # Wait with exponential backoff
                        time.sleep(0.5 * (2 ** attempt))
                except FileNotFoundError:
                    logger.debug(f"File already removed: {path}")
                    return
            logger.warning(f"Failed to remove file after multiple attempts: {path}")
        except ImportError:
            # Fallback if win32file is not available
            for attempt in range(10):
                try:
                    os.remove(path)
                    return
                except PermissionError:
                    time.sleep(0.5 * (attempt + 1))
                except FileNotFoundError:
                    return

    @staticmethod
    def _kill_processes_locking_file(path):
        # Kill processes that might be locking a file
        if not path or not os.path.exists(path):
            return
        try:
            subprocess.run(["taskkill", "/F", "/FI", f"MODULES eq {os.path.basename(path)}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception as e:
            logger.debug(f"Error killing processes: {e}")

    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        try:
            if file_path.endswith('.wav'):
                with wave.open(file_path, 'rb') as wf:
                    return float(wf.getnframes()) / wf.getframerate()
            else:
                cmd = ["ffprobe", "-v", 
                       "error", "-show_entries", 
                       "format=duration", "-of", 
                       "default=noprint_wrappers=1:nokey=1", file_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Failed to get duration: {e}")
            raise AudioProcessingError(f"Cannot determine duration: {str(e)}", AudioProcessingError.ErrorType.INVALID_FORMAT)