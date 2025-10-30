
"""
Unified Audio Processing Module for TranscrevAI.
Combines real-time recording, VAD pre-processing, dynamic quantization analysis,
and robust audio file loading into a single, comprehensive toolkit.
"""

import asyncio
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from fastapi import WebSocket
from datetime import datetime

import numpy as np
import soundfile as sf
import psutil

# Lazy imports for optional, heavy dependencies
_pyaudio = None
_librosa = None

logger = logging.getLogger(__name__)

# ============================================================================
# Adaptive Performance Configuration
# ============================================================================

def configure_adaptive_threads() -> Tuple[int, int]:
    """
    Automatically configure optimal thread counts for torch (diarization) and
    OpenMP (transcription) based on available hardware resources.

    This function intelligently allocates CPU threads to maximize performance
    across different hardware configurations (from 2-core laptops to 32+ core servers).

    Strategy:
    - Transcription (OpenMP/faster-whisper): CPU-intensive, benefits from more threads
    - Diarization (PyTorch/pyannote): Memory-bound, needs fewer threads to avoid contention

    Returns:
        Tuple[torch_threads, omp_threads]: Optimal thread counts for PyTorch and OpenMP
    """
    # Detect hardware
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = os.cpu_count() or 1
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

    logger.info(f"Hardware Detection: {physical_cores} physical cores, "
               f"{logical_cores} logical cores, {total_ram_gb:.1f}GB total RAM, "
               f"{available_ram_gb:.1f}GB available")

    # Adaptive allocation based on core count tiers
    if physical_cores <= 2:
        # Low-end systems (2 cores): Minimal threading
        torch_threads = 1
        omp_threads = 1
        tier = "Low-end (2 cores)"

    elif physical_cores <= 4:
        # Mid-range systems (4 cores): Conservative allocation
        # Transcription gets priority
        torch_threads = 1
        omp_threads = max(1, physical_cores - 1)
        tier = "Mid-range (4 cores)"

    elif physical_cores <= 8:
        # Standard systems (6-8 cores): Balanced allocation
        # This is the most common configuration
        torch_threads = 2
        omp_threads = max(1, physical_cores - 2)
        tier = "Standard (6-8 cores)"

    elif physical_cores <= 16:
        # High-end systems (12-16 cores): Aggressive allocation
        # Scale up both, but transcription gets majority
        torch_threads = min(4, physical_cores // 4)
        omp_threads = max(1, physical_cores - torch_threads)
        tier = "High-end (12-16 cores)"

    else:
        # Server systems (16+ cores): Maximum parallelism
        # Scale both significantly for server-grade CPUs
        torch_threads = min(8, physical_cores // 4)
        omp_threads = max(1, physical_cores - torch_threads)
        tier = "Server (16+ cores)"

    # Memory-based adjustments
    # If available RAM < 4GB, reduce threads to prevent thrashing
    if available_ram_gb < 4:
        logger.warning(f"Low available RAM ({available_ram_gb:.1f}GB). Reducing thread counts to prevent memory thrashing.")
        torch_threads = max(1, torch_threads // 2)
        omp_threads = max(1, omp_threads // 2)

    # Validate thread counts don't exceed available cores
    # This prevents oversubscription
    total_threads = torch_threads + omp_threads
    if total_threads > logical_cores:
        logger.warning(f"Total threads ({total_threads}) exceeds logical cores ({logical_cores}). Scaling down.")
        scale_factor = logical_cores / total_threads
        torch_threads = max(1, int(torch_threads * scale_factor))
        omp_threads = max(1, int(omp_threads * scale_factor))

    # Calculate allocation percentages for logging
    total = torch_threads + omp_threads
    torch_pct = (torch_threads / total) * 100
    omp_pct = (omp_threads / total) * 100

    logger.info(f"Adaptive Thread Configuration:")
    logger.info(f"  System Tier: {tier}")
    logger.info(f"  PyTorch (diarization): {torch_threads} threads ({torch_pct:.0f}%)")
    logger.info(f"  OpenMP (transcription): {omp_threads} threads ({omp_pct:.0f}%)")
    logger.info(f"  Rationale: Transcription (CPU-heavy) gets {omp_pct:.0f}%, Diarization (memory-bound) gets {torch_pct:.0f}%")

    return torch_threads, omp_threads

def _get_pyaudio():
    """Lazy import of PyAudio to avoid import errors if not installed."""
    global _pyaudio
    if _pyaudio is None:
        try:
            import pyaudio
            _pyaudio = pyaudio
        except ImportError:
            logger.critical("PyAudio is not installed. Live recording will not work.")
            raise
    return _pyaudio

def _get_librosa():
    """Lazy import of Librosa for audio analysis."""
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
        except ImportError:
            logger.warning("Librosa not available for advanced audio analysis.")
    return _librosa

def convert_wav_to_mp4(input_path: str, output_path: str, subtitle_path: Optional[str] = None) -> bool:
    """
    Convert a WAV file to MP4 video with black background.

    If subtitle_path is provided, creates a video MP4 with:
    - Black background video (1920x1080)
    - Audio from WAV
    - Subtitles as a separate track (can be toggled on/off in player)

    If subtitle_path is not provided, creates audio-only MP4.

    Args:
        input_path: Path to the input WAV file.
        output_path: Path to save the output MP4 file.
        subtitle_path: Optional path to SRT subtitle file.

    Returns:
        True if conversion was successful, False otherwise.
    """
    try:
        logger.info(f"Converting {input_path} to MP4 (with video: {subtitle_path is not None})...")

        if subtitle_path and os.path.exists(subtitle_path):
            # Get audio duration for video length
            import librosa
            duration = librosa.get_duration(path=input_path)

            # Create MP4 video with black background and subtitle track
            command = [
                "ffmpeg",
                "-y",  # Overwrite output file if exists
                "-f", "lavfi",
                "-i", f"color=c=black:s=1920x1080:d={duration}:r=30",  # Black video
                "-i", input_path,  # Audio input
                "-i", subtitle_path,  # Subtitle input
                "-c:v", "libx264",  # Video codec
                "-preset", "fast",  # Encoding speed
                "-crf", "23",  # Quality (lower = better, 23 is good default)
                "-c:a", "aac",  # Audio codec
                "-b:a", "192k",  # Audio bitrate
                "-c:s", "mov_text",  # Subtitle codec for MP4
                "-metadata:s:s:0", "language=por",  # Portuguese subtitle language
                "-metadata:s:s:0", "title=Legendas",  # Subtitle track title
                "-shortest",  # End when shortest input ends
                output_path
            ]
            logger.info(f"Creating MP4 video with black background and embedded subtitles (duration: {duration:.2f}s)")
        else:
            # Fallback: audio-only MP4 (original behavior)
            command = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-c:a", "aac",
                "-b:a", "192k",
                output_path
            ]
            logger.info("Creating audio-only MP4 (no subtitles provided)")

        result = subprocess.run(command, check=True, capture_output=True, text=True)

        logger.info(f"Successfully converted {input_path} to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed for {input_path}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during MP4 conversion of {input_path}: {e}")
        return False




# --- Dynamic Quantization --- #

class QuantizationLevel(Enum):
    INT8 = "int8"
    FLOAT16 = "float16"

@dataclass
class AudioQualityMetrics:
    clarity_score: float
    recommended_quantization: QuantizationLevel
    has_issues: bool = False
    warnings: List[str] = field(default_factory=list)
    rms_level: float = 0.0
    snr_estimate: float = 0.0
    clipping_detected: bool = False

class AudioQualityAnalyzer:
    """
    Analyzes audio quality and provides warnings to users about potential issues.

    Detects:
    - Low audio volume (RMS < threshold)
    - Excessive noise (poor SNR)
    - Audio clipping (peak detection)
    - Poor spectral clarity
    """

    def __init__(self):
        self.low_volume_threshold = 0.01  # RMS below this indicates very low volume
        self.normal_volume_threshold = 0.05  # RMS below this indicates low volume
        self.clipping_threshold = 0.95  # Peak amplitude above this indicates clipping
        self.poor_clarity_threshold = 0.4  # Clarity below this indicates poor quality

    def analyze_audio_quality(self, audio_path: str) -> AudioQualityMetrics:
        """
        Comprehensive audio quality analysis with user-facing warnings.

        Returns:
            AudioQualityMetrics with quality scores and user warnings
        """
        librosa = _get_librosa()
        if not librosa:
            logger.warning("Librosa not available, skipping quality analysis.")
            return AudioQualityMetrics(
                clarity_score=0.5,
                recommended_quantization=QuantizationLevel.INT8,
                has_issues=False,
                warnings=[]
            )

        try:
            # Load first 30 seconds for analysis (sufficient for quality assessment)
            y, sr = librosa.load(audio_path, sr=16000, duration=30, mono=True)

            warnings = []
            has_issues = False

            # 1. RMS (Root Mean Square) - Volume level analysis
            rms = np.sqrt(np.mean(y**2))

            if rms < self.low_volume_threshold:
                warnings.append("⚠️ Volume muito baixo detectado. Considere aumentar o volume da gravação.")
                has_issues = True
            elif rms < self.normal_volume_threshold:
                warnings.append("⚠️ Volume baixo detectado. A qualidade da transcrição pode ser afetada.")
                has_issues = True

            # 2. Clipping detection - Audio distortion
            peak_amplitude = np.max(np.abs(y))
            clipping_detected = peak_amplitude >= self.clipping_threshold

            if clipping_detected:
                warnings.append("⚠️ Distorção de áudio detectada (clipping). Reduza o volume da entrada.")
                has_issues = True

            # 3. Spectral clarity - Overall audio quality
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_mean = np.mean(spectral_centroids)
            clarity = 1.0 - min(1.0, np.std(spectral_centroids) / centroid_mean) if centroid_mean > 0 else 0.5

            if clarity < self.poor_clarity_threshold:
                warnings.append("⚠️ Qualidade de áudio ruim detectada. Verifique o ambiente e o microfone.")
                has_issues = True

            # 4. Noise estimation (simple SNR estimate using energy ratio)
            # Calculate energy in high-frequency bands (noise) vs low-frequency (speech)
            frame_length = 2048
            hop_length = 512
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
            snr_estimate = np.mean(spectral_rolloff) / (sr / 2)  # Normalized estimate

            if snr_estimate < 0.3:  # Poor SNR indicates excessive noise
                warnings.append("⚠️ Ruído de fundo excessivo detectado. A transcrição pode ter erros.")
                has_issues = True

            # Determine quantization based on overall quality
            quantization = QuantizationLevel.FLOAT16 if clarity > 0.7 and not has_issues else QuantizationLevel.INT8

            logger.info(f"Audio quality analysis: RMS={rms:.4f}, Clarity={clarity:.2f}, SNR_Est={snr_estimate:.2f}, Clipping={clipping_detected}, Issues={has_issues}")

            return AudioQualityMetrics(
                clarity_score=clarity,
                recommended_quantization=quantization,
                has_issues=has_issues,
                warnings=warnings,
                rms_level=float(rms),
                snr_estimate=float(snr_estimate),
                clipping_detected=clipping_detected
            )

        except Exception as e:
            logger.error(f"Audio quality analysis failed: {e}")
            return AudioQualityMetrics(
                clarity_score=0.5,
                recommended_quantization=QuantizationLevel.INT8,
                has_issues=False,
                warnings=[]
            )

# --- Real-time Audio Recording --- #

class AudioRecorder:
    """A functional audio recorder using PyAudio for real-time microphone capture."""
    def __init__(self, output_file: str, sample_rate: int = 16000):
        self.output_file = output_file
        self.sample_rate = sample_rate
        self._is_recording = False
        self._audio_queue = queue.Queue()
        self._recording_thread: Optional[threading.Thread] = None
        self.CHUNK = 1024
        self.FORMAT = _get_pyaudio().paInt16
        self.CHANNELS = 1
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

    def _recording_loop(self):
        p = _get_pyaudio().PyAudio()
        try:
            stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.sample_rate, input=True, frames_per_buffer=self.CHUNK)
        except Exception as e:
            logger.critical(f"Failed to open microphone stream: {e}")
            return

        while self._is_recording:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self._audio_queue.put(data)
            except IOError as e:
                logger.warning(f"IOError during recording: {e}")
        
        stream.stop_stream()
        stream.close()
        p.terminate()

    def start_recording(self):
        if self._is_recording: return
        self._is_recording = True
        self._recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self._recording_thread.start()
        logger.info(f"AudioRecorder started, writing to: {self.output_file}")

    def stop_recording(self):
        if not self._is_recording: return
        self._is_recording = False
        if self._recording_thread: self._recording_thread.join(timeout=2.0)
        
        with wave.open(self.output_file, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(_get_pyaudio().PyAudio().get_sample_size(self.FORMAT))
            wf.setframerate(self.sample_rate)
            while not self._audio_queue.empty():
                wf.writeframes(self._audio_queue.get())
        logger.info(f"Audio file saved: {self.output_file}")


# --- Live Audio Processor with Disk Buffering --- #

class RecordingState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    PROCESSING = "processing"
    COMPLETE = "complete"

class LiveAudioProcessor:
    """
    Manages live audio recording with disk buffering (Option A - Compliance validated).
    Uses temporary file storage to maintain low RAM usage during recording.
    """
    def __init__(self, temp_dir: Optional[str] = None):
        if temp_dir is None:
            from src.file_manager import FileManager
            temp_dir = FileManager.get_data_path("temp")
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        logger.info("LiveAudioProcessor initialized with disk buffering strategy")

    async def start_recording(self, session_id: str, sample_rate: int = 16000) -> Dict[str, str]:
        """Start a new recording session with disk buffering"""
        with self._lock:
            if session_id in self.sessions:
                current_state = self.sessions[session_id].get("state")
                if current_state and current_state != RecordingState.IDLE:
                    raise ValueError(f"Session {session_id} already active with state: {current_state.value}")

            temp_file = self.temp_dir / f"{session_id}_{int(time.time())}.wav"

            self.sessions[session_id] = {
                "state": RecordingState.RECORDING,
                "temp_file": temp_file,
                "sample_rate": sample_rate,
                "start_time": time.time(),
                "chunks_received": 0,
                "total_bytes": 0
            }

            logger.info(f"Recording started for session {session_id}, buffering to: {temp_file}")
            return {"status": "recording", "session_id": session_id, "temp_file": str(temp_file)}

    async def pause_recording(self, session_id: str) -> Dict[str, str]:
        """Pause active recording"""
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            session = self.sessions[session_id]
            current_state = session.get("state")
            if current_state and current_state != RecordingState.RECORDING:
                raise ValueError(f"Cannot pause session in state: {current_state.value}")

            session["state"] = RecordingState.PAUSED
            session["pause_time"] = time.time()
            logger.info(f"Recording paused for session {session_id}")
            return {"status": "paused", "session_id": session_id}

    async def resume_recording(self, session_id: str) -> Dict[str, str]:
        """Resume paused recording"""
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            session = self.sessions[session_id]
            current_state = session.get("state")
            if current_state and current_state != RecordingState.PAUSED:
                raise ValueError(f"Cannot resume session in state: {current_state.value}")

            session["state"] = RecordingState.RECORDING
            session["resume_time"] = time.time()
            logger.info(f"Recording resumed for session {session_id}")
            return {"status": "recording", "session_id": session_id}

    async def process_audio_chunk(self, session_id: str, audio_chunk: bytes) -> Dict[str, Any]:
        """
        Store audio chunk in memory buffer for proper WAV file generation.
        Compliance: Uses in-memory buffering for valid WAV structure.
        """
        with self._lock:
            if session_id not in self.sessions:
                # Session already completed/deleted - ignore late chunks from frontend
                logger.debug(f"Ignoring chunk for completed/deleted session {session_id}")
                return {"status": "session_completed", "chunks_received": 0}

            session = self.sessions[session_id]
            current_state = session.get("state")
            if current_state and current_state not in [RecordingState.RECORDING, RecordingState.PAUSED]:
                # Session in PROCESSING/COMPLETE state - ignore late chunks
                logger.debug(f"Ignoring chunk for session {session_id} in state {current_state.value}")
                return {"status": "session_not_recording", "chunks_received": session.get("chunks_received", 0)}

            # Store chunks in memory buffer
            if "audio_buffer" not in session:
                session["audio_buffer"] = []

            session["audio_buffer"].append(audio_chunk)
            session["chunks_received"] = session.get("chunks_received", 0) + 1
            session["total_bytes"] = session.get("total_bytes", 0) + len(audio_chunk)

            return {
                "status": "chunk_buffered",
                "chunks_received": session["chunks_received"],
                "total_bytes": session["total_bytes"]
            }

    async def stop_recording(self, session_id: str) -> str:
        """
        Stop recording, convert WebM to WAV, and return path for processing.
        Transitions to PROCESSING state.
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            session = self.sessions[session_id]
            current_state = session.get("state")
            if current_state and current_state not in [RecordingState.RECORDING, RecordingState.PAUSED]:
                raise ValueError(f"Cannot stop session in state: {current_state.value}")

            session["state"] = RecordingState.PROCESSING
            session["stop_time"] = time.time()
            duration = session["stop_time"] - session.get("start_time", session["stop_time"])

            # Get audio chunks from buffer
            audio_buffer = session.get("audio_buffer", [])
            if not audio_buffer:
                logger.warning(f"Session {session_id} has no audio data")
                raise ValueError("No audio data recorded")

            audio_data = b''.join(audio_buffer)
            temp_file_path = str(session.get("temp_file"))

            # Write raw WebM data to temporary file
            webm_temp = temp_file_path.replace('.wav', '_raw.webm')
            with open(webm_temp, 'wb') as f:
                f.write(audio_data)

            logger.info(f"WebM data written: {len(audio_data)} bytes to {webm_temp}")

            # Convert WebM to WAV using FFMPEG
            try:
                await self._convert_webm_to_wav(webm_temp, temp_file_path, session.get("sample_rate", 16000))
                logger.info(f"Successfully converted WebM to WAV: {temp_file_path}")

                # Clean up temporary WebM file
                Path(webm_temp).unlink(missing_ok=True)

            except Exception as e:
                logger.error(f"Failed to convert WebM to WAV: {e}")
                # Clean up
                Path(webm_temp).unlink(missing_ok=True)
                raise ValueError(f"Audio conversion failed: {e}")

            # Clear buffer to free memory
            if "audio_buffer" in session:
                session["audio_buffer"].clear()

            logger.info(f"Recording stopped for session {session_id}. Duration: {duration:.2f}s, File: {temp_file_path}")

            return temp_file_path

    async def _convert_webm_to_wav(self, input_path: str, output_path: str, sample_rate: int = 16000):
        """
        Convert WebM audio to WAV using FFMPEG.
        Uses ffmpeg from static_ffmpeg package or system installation.
        """
        try:
            # Try using static_ffmpeg first (bundled with app)
            try:
                from static_ffmpeg import run
                ffmpeg_cmd = run.get_or_fetch_platform_executables_else_raise()
                ffmpeg_path = ffmpeg_cmd[0]  # Get ffmpeg executable path
                logger.debug(f"Using static_ffmpeg from: {ffmpeg_path}")
            except Exception as e:
                # Fallback to system ffmpeg
                ffmpeg_path = "ffmpeg"
                logger.debug(f"Using system ffmpeg: {e}")

            # FFMPEG command to convert WebM → WAV (mono, 16kHz, 16-bit PCM)
            cmd = [
                ffmpeg_path,
                '-i', input_path,           # Input file
                '-vn',                       # No video
                '-acodec', 'pcm_s16le',     # 16-bit PCM
                '-ar', str(sample_rate),    # Sample rate
                '-ac', '1',                  # Mono channel
                '-y',                        # Overwrite output
                output_path                  # Output file
            ]

            logger.debug(f"Running FFMPEG: {' '.join(cmd)}")

            # Run ffmpeg synchronously (but within async context)
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,  # 30 second timeout
                check=True
            )

            logger.info(f"FFMPEG conversion successful: {input_path} → {output_path}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("FFMPEG conversion timed out (>30s)")
        except subprocess.CalledProcessError as e:
            stderr_bytes = e.stderr
            if stderr_bytes:
                stderr = stderr_bytes.decode('utf-8', errors='ignore')
            else:
                stderr = 'No error output'
            raise RuntimeError(f"FFMPEG failed: {stderr}")
        except FileNotFoundError:
            raise RuntimeError("FFMPEG not found. Please install ffmpeg or check static_ffmpeg installation.")

    async def complete_session(self, session_id: str) -> None:
        """Mark session as complete and cleanup temporary file"""
        with self._lock:
            if session_id not in self.sessions:
                return

            session = self.sessions[session_id]
            session["state"] = RecordingState.COMPLETE

            # Cleanup temp file
            temp_file = session.get("temp_file")
            if temp_file and temp_file.exists():
                temp_file.unlink()
                logger.info(f"Temporary file deleted for session {session_id}")

            # Remove session after brief retention
            del self.sessions[session_id]

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state"""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None

            state = session.get("state")
            return {
                "state": state.value if state else None,
                "chunks_received": session.get("chunks_received", 0),
                "total_bytes": session.get("total_bytes", 0),
                "duration": time.time() - session.get("start_time", time.time())
            }


# ============================================================================
# Session Management for Live Recording
# ============================================================================

@dataclass
class SessionData:
    """Represents all data associated with a single recording session."""
    session_id: str
    websocket: "WebSocket"
    format: Optional[str]
    started_at: "datetime"
    temp_file: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)
    status: str = "idle"

class SessionManager:
    """Centralized session lifecycle management.

    Single source of truth for all active recording sessions.
    Thread-safe operations using asyncio.Lock.
    """

    def __init__(self, session_timeout_hours: int = 24):
        self.sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
        self.session_timeout_hours = session_timeout_hours # Keep timeout for future use
        logger.info(f"SessionManager initialized (timeout: {session_timeout_hours}h)")

    async def create_session(self, session_id: str, session_data: SessionData) -> None:
        """Create a new recording session."""
        async with self._lock:
            if session_id in self.sessions:
                raise ValueError(f"Session {session_id} already exists")
            self.sessions[session_id] = session_data
            logger.info(f"Session created: {session_id}")

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve session data by ID."""
        async with self._lock:
            return self.sessions.get(session_id)

    async def remove_session(self, session_id: str) -> bool:
        """Remove session and cleanup resources."""
        async with self._lock:
            if session_id in self.sessions:
                session_data = self.sessions.pop(session_id, None)
                if session_data and session_data.temp_file:
                    try:
                        if os.path.exists(session_data.temp_file):
                            os.remove(session_data.temp_file)
                            logger.debug(f"Cleaned up temp file: {session_data.temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp file for session {session_id}: {e}")
                logger.info(f"Session removed: {session_id}")
                return True
            return False

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions

    async def get_active_session_count(self) -> int:
        """Get the number of currently active sessions."""
        async with self._lock:
            return len(self.sessions)

    async def get_all_session_ids(self) -> List[str]:
        """Get a list of all active session IDs."""
        async with self._lock:
            return list(self.sessions.keys())
