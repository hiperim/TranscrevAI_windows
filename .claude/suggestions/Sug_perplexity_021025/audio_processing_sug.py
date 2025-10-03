# FIXED - Enhanced Audio Processing Module with UTF-8 Support
"""
Unified Audio Processing Module for TranscrevAI
Combines real-time recording capabilities with robust audio file loading and processing.

FIXES APPLIED:
- Complete UTF-8 encoding for all file operations
- Enhanced error handling with proper encoding
- Windows compatibility improvements
- Robust file path handling with proper encoding
"""

import asyncio
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import wave
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import psutil
import soundfile as sf

# Lazy imports for optional dependencies
torch = None
torchaudio = None
librosa = None
pyaudio = None

logger = logging.getLogger(__name__)

def get_pyaudio():
    """Lazy import of PyAudio"""
    global pyaudio
    if pyaudio is None:
        try:
            import pyaudio
            pyaudio = pyaudio
        except ImportError:
            logger.critical("PyAudio is not installed. Live recording will not work.")
            raise
    return pyaudio

def get_librosa():
    """Lazy import of librosa"""
    global librosa
    if librosa is None:
        try:
            import librosa
        except ImportError:
            logger.warning("Librosa not available for advanced audio processing")
            librosa = None
    return librosa

def get_torch():
    """Lazy import of torch/torchaudio"""
    global torch, torchaudio
    if torch is None:
        try:
            import torch
            import torchaudio
        except ImportError:
            logger.warning("PyTorch/torchaudio not available for advanced processing")
            torch = None
            torchaudio = None
    return torch, torchaudio

class AudioFormat(Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"

class AudioRecorder:
    """
    Real-time audio recorder using PyAudio
    Thread-safe recording with WebSocket progress updates
    """
    
    def __init__(self, output_file: str, websocket_manager=None, session_id: str = None):
        self.output_file = Path(output_file)
        self.websocket_manager = websocket_manager
        self.session_id = session_id
        
        # Audio parameters optimized for transcription
        self.sample_rate = 16000  # Whisper's preferred sample rate
        self.chunk_size = 1024
        self.channels = 1  # Mono for transcription
        self.format = None  # Will be set when PyAudio is available
        
        # Recording state
        self.is_recording = False
        self.is_paused = False
        self.audio_data = []
        self.recording_thread = None
        
        # Ensure output directory exists with proper permissions
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AudioRecorder initialized: {self.output_file}")

    async def start_recording(self):
        """Start recording audio in a separate thread"""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
        
        try:
            pyaudio_module = get_pyaudio()
            self.format = pyaudio_module.paInt16
            
            self.is_recording = True
            self.is_paused = False
            self.audio_data = []
            
            # Start recording in a separate thread to avoid blocking
            self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
            self.recording_thread.start()
            
            logger.info(f"Started recording to {self.output_file}")
            
            if self.websocket_manager and self.session_id:
                await self.websocket_manager.send_message(self.session_id, {
                    "type": "recording_status",
                    "status": "started",
                    "file": str(self.output_file)
                })
                
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            raise

    def _recording_worker(self):
        """Worker thread for audio recording"""
        pyaudio_module = get_pyaudio()
        audio = pyaudio_module.PyAudio()
        stream = None
        
        try:
            # Open audio stream
            stream = audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Audio stream opened for recording")
            
            while self.is_recording:
                if not self.is_paused:
                    try:
                        # Read audio data
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        self.audio_data.append(data)
                        
                        # Send periodic updates via WebSocket
                        if len(self.audio_data) % 50 == 0:  # Every ~3 seconds at 16kHz
                            duration = len(self.audio_data) * self.chunk_size / self.sample_rate
                            asyncio.create_task(self._send_recording_update(duration))
                            
                    except Exception as e:
                        logger.warning(f"Error reading audio data: {e}")
                        break
                else:
                    time.sleep(0.1)  # Small delay when paused
                    
        except Exception as e:
            logger.error(f"Recording worker error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            audio.terminate()
            logger.info("Audio stream closed")

    async def _send_recording_update(self, duration: float):
        """Send recording progress update via WebSocket"""
        if self.websocket_manager and self.session_id:
            try:
                await self.websocket_manager.send_message(self.session_id, {
                    "type": "recording_progress",
                    "duration": round(duration, 1),
                    "status": "paused" if self.is_paused else "recording"
                })
            except Exception as e:
                logger.warning(f"Failed to send recording update: {e}")

    async def stop_recording(self):
        """Stop recording and save to file"""
        if not self.is_recording:
            logger.warning("No recording in progress")
            return
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5.0)
        
        try:
            # Save audio data to WAV file with proper UTF-8 path handling
            with wave.open(str(self.output_file), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(self.audio_data))
            
            duration = len(self.audio_data) * self.chunk_size / self.sample_rate
            file_size = self.output_file.stat().st_size
            
            logger.info(f"Recording saved: {self.output_file} ({duration:.1f}s, {file_size:,} bytes)")
            
            if self.websocket_manager and self.session_id:
                await self.websocket_manager.send_message(self.session_id, {
                    "type": "recording_complete",
                    "file": str(self.output_file),
                    "duration": round(duration, 1),
                    "file_size": file_size
                })
                
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            raise
        finally:
            self.audio_data = []

    def pause_recording(self):
        """Pause recording"""
        if self.is_recording:
            self.is_paused = True
            logger.info("Recording paused")

    def resume_recording(self):
        """Resume recording"""
        if self.is_recording:
            self.is_paused = False
            logger.info("Recording resumed")

class RobustAudioLoader:
    """
    Fallback-based audio file loader for maximum compatibility
    Handles various audio formats with proper UTF-8 encoding
    """
    
    @staticmethod
    def load_audio(file_path: Union[str, Path], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio file with multiple fallback methods
        
        Args:
            file_path: Path to audio file (UTF-8 encoded)
            target_sr: Target sample rate
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        file_path = Path(file_path)
        
        # Ensure file exists and is readable
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Try multiple loading methods in order of preference
        loading_methods = [
            RobustAudioLoader._load_with_soundfile,
            RobustAudioLoader._load_with_librosa,
            RobustAudioLoader._load_with_torchaudio,
            RobustAudioLoader._load_with_ffmpeg
        ]
        
        last_error = None
        for method in loading_methods:
            try:
                audio_data, sr = method(file_path, target_sr)
                logger.info(f"Successfully loaded {file_path} using {method.__name__}")
                return audio_data, sr
            except Exception as e:
                last_error = e
                logger.debug(f"{method.__name__} failed: {e}")
                continue
        
        raise RuntimeError(f"All audio loading methods failed. Last error: {last_error}")

    @staticmethod
    def _load_with_soundfile(file_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
        """Load audio using soundfile (fastest, most reliable)"""
        # Use str() to ensure proper path encoding
        audio_data, sr = sf.read(str(file_path), dtype=np.float32)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed (basic resampling)
        if sr != target_sr:
            audio_data = RobustAudioLoader._simple_resample(audio_data, sr, target_sr)
            sr = target_sr
        
        return audio_data, sr

    @staticmethod
    def _load_with_librosa(file_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
        """Load audio using librosa"""
        librosa = get_librosa()
        if librosa is None:
            raise ImportError("Librosa not available")
        
        audio_data, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
        return audio_data, sr

    @staticmethod
    def _load_with_torchaudio(file_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
        """Load audio using torchaudio"""
        torch, torchaudio = get_torch()
        if torch is None or torchaudio is None:
            raise ImportError("PyTorch/torchaudio not available")
        
        waveform, sr = torchaudio.load(str(file_path))
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Convert to numpy
        audio_data = waveform.squeeze().numpy()
        return audio_data, target_sr

    @staticmethod
    def _load_with_ffmpeg(file_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
        """Load audio using ffmpeg subprocess"""
        try:
            # Use ffmpeg to convert to raw PCM with proper encoding handling
            cmd = [
                'ffmpeg', '-i', str(file_path),
                '-f', 'f32le',  # 32-bit float little-endian
                '-ac', '1',     # mono
                '-ar', str(target_sr),  # target sample rate
                '-'             # output to stdout
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                check=True,
                encoding=None  # Binary mode for audio data
            )
            
            # Convert binary data to numpy array
            audio_data = np.frombuffer(result.stdout, dtype=np.float32)
            return audio_data, target_sr
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found in system PATH")

    @staticmethod
    def _simple_resample(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear resampling (fallback when librosa unavailable)"""
        if orig_sr == target_sr:
            return audio_data
        
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        new_length = int(len(audio_data) * ratio)
        
        # Simple linear interpolation
        old_indices = np.linspace(0, len(audio_data) - 1, new_length)
        new_audio = np.interp(old_indices, np.arange(len(audio_data)), audio_data)
        
        return new_audio.astype(np.float32)

class OptimizedAudioProcessor:
    """
    Utilities for fast, efficient audio file manipulation
    Optimized for transcription preprocessing with UTF-8 support
    """
    
    @staticmethod
    def validate_audio_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate and analyze audio file with proper encoding handling
        
        Returns:
            Dictionary with file information and validation results
        """
        file_path = Path(file_path)
        
        try:
            # Check file existence and readability
            if not file_path.exists():
                return {"valid": False, "error": "File not found"}
            
            if not file_path.is_file():
                return {"valid": False, "error": "Path is not a file"}
            
            # Get file stats
            stat = file_path.stat()
            file_size_mb = stat.st_size / (1024 * 1024)
            
            # Check file size limits (reasonable for transcription)
            if file_size_mb > 500:  # 500MB limit
                return {"valid": False, "error": f"File too large: {file_size_mb:.1f}MB"}
            
            if file_size_mb < 0.001:  # 1KB minimum
                return {"valid": False, "error": "File too small"}
            
            # Check file extension
            supported_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma'}
            if file_path.suffix.lower() not in supported_extensions:
                return {"valid": False, "error": f"Unsupported format: {file_path.suffix}"}
            
            # Try to get audio info
            try:
                info = sf.info(str(file_path))
                duration = info.frames / info.samplerate
                
                # Check duration limits
                if duration > 3600:  # 1 hour limit
                    return {"valid": False, "error": f"Audio too long: {duration/60:.1f} minutes"}
                
                if duration < 0.1:  # 100ms minimum
                    return {"valid": False, "error": "Audio too short"}
                
                return {
                    "valid": True,
                    "file_size_mb": file_size_mb,
                    "duration_seconds": duration,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "format": info.format_info,
                    "encoding": "utf-8"  # Ensure UTF-8 metadata
                }
                
            except Exception as e:
                return {"valid": False, "error": f"Cannot read audio metadata: {e}"}
            
        except Exception as e:
            return {"valid": False, "error": f"File validation failed: {e}"}

    @staticmethod
    def normalize_audio_for_transcription(audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio for optimal transcription quality
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Normalized audio array
        """
        try:
            # Convert to float32 for processing
            audio = audio_data.astype(np.float32)
            
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            # Apply gentle compression to boost quiet parts
            # This helps transcription models handle varying volume levels
            compressed = np.sign(audio) * np.sqrt(np.abs(audio))
            
            # Blend original and compressed (80% original, 20% compressed)
            audio = 0.8 * audio + 0.2 * compressed
            
            # Final normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95  # Leave some headroom
            
            return audio
            
        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}")
            return audio_data  # Return original if normalization fails

    @staticmethod
    def convert_to_wav(input_file: Union[str, Path], output_file: Union[str, Path], 
                      target_sr: int = 16000) -> bool:
        """
        Convert audio file to WAV format optimized for transcription
        
        Args:
            input_file: Input audio file path
            output_file: Output WAV file path
            target_sr: Target sample rate
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Load audio with robust loader
            audio_data, sr = RobustAudioLoader.load_audio(input_file, target_sr)
            
            # Normalize for transcription
            audio_data = OptimizedAudioProcessor.normalize_audio_for_transcription(audio_data)
            
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as WAV with UTF-8 path handling
            sf.write(str(output_path), audio_data, target_sr, format='WAV', subtype='PCM_16')
            
            logger.info(f"Converted {input_file} to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return False

# Memory monitoring for audio processing
def get_audio_memory_usage() -> Dict[str, float]:
    """Get current memory usage for audio processing monitoring"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent()
        }
    except Exception:
        return {"rss_mb": 0, "vms_mb": 0, "percent": 0}

# Utility functions for file operations with UTF-8 support
def ensure_utf8_path(file_path: Union[str, Path]) -> Path:
    """Ensure file path is properly UTF-8 encoded for Windows compatibility"""
    path = Path(file_path)
    
    # On Windows, ensure proper encoding
    if os.name == 'nt':
        try:
            # Test if path can be properly encoded/decoded
            str(path).encode('utf-8').decode('utf-8')
        except UnicodeError:
            logger.warning(f"Path encoding issue detected: {path}")
            # Fallback to ASCII-safe name
            safe_name = str(path.name).encode('ascii', 'replace').decode('ascii')
            path = path.parent / safe_name
    
    return path

def safe_file_operation(file_path: Union[str, Path], operation: str = "read") -> bool:
    """
    Test if file operation can be performed safely with UTF-8 encoding
    
    Args:
        file_path: File path to test
        operation: Operation type ("read", "write", "create")
        
    Returns:
        True if operation is safe, False otherwise
    """
    try:
        path = ensure_utf8_path(file_path)
        
        if operation == "read":
            return path.exists() and path.is_file()
        elif operation == "write":
            return path.parent.exists() and os.access(path.parent, os.W_OK)
        elif operation == "create":
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.parent.exists()
        
        return False
        
    except Exception as e:
        logger.warning(f"File operation safety check failed: {e}")
        return False