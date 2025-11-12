import asyncio
import logging
import os
import subprocess
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from fastapi import WebSocket
from datetime import datetime
import numpy as np
import psutil
from src.file_manager import FileManager
# Lazy imports for heavy dependencies
_pyaudio = None
_librosa = None

logger = logging.getLogger(__name__)

def configure_adaptive_threads() -> Tuple[int, int]:
    """
    Automatically configure optimal thread counts for torch (diarization) and OpenMP (transcription) based on available hardware resources - Returns: Tuple[torch_threads, omp_threads]: Optimal thread counts for PyTorch and OpenMP
    """
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
        torch_threads = 2
        omp_threads = max(1, physical_cores - 2)
        tier = "Standard (6-8 cores)"

    elif physical_cores <= 16:
        # High-end systems (12-16 cores): Aggressive allocation
        torch_threads = min(4, physical_cores // 4)
        omp_threads = max(1, physical_cores - torch_threads)
        tier = "High-end (12-16 cores)"

    else:
        # Server systems (16+ cores): Maximum parallelism
        torch_threads = min(8, physical_cores // 4)
        omp_threads = max(1, physical_cores - torch_threads)
        tier = "Server (16+ cores)"

    # Memory-based adjustments
    if available_ram_gb < 4:
        logger.warning(f"Low available RAM ({available_ram_gb:.1f}GB). Reducing thread counts to prevent memory thrashing.")
        torch_threads = max(1, torch_threads // 2)
        omp_threads = max(1, omp_threads // 2)

    # Validate thread counts don't exceed available cores
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
    """Convert a WAV file to MP4 video with black background"""
    try:
        logger.info(f"Converting {input_path} to MP4 (with video: {subtitle_path is not None})...")

        if subtitle_path and os.path.exists(subtitle_path):
            import librosa
            duration = librosa.get_duration(path=input_path)

            # .mp4 video with black background and subtitle track if available
            command = [
                "ffmpeg",
                "-y",  # Overwrite output file if exists
                "-f", "lavfi",
                "-i", f"color=c=black:s=1920x1080:d={duration}:r=30",  # Black video
                "-i", input_path,  # Audio input
                "-i", subtitle_path,  # Subtitle input
                "-c:v", "libx264",  # Video codec
                "-preset", "fast",  # Encoding speed
                "-crf", "23",  # Quality (lower = better)
                "-c:a", "aac",  # Audio codec
                "-b:a", "192k",  # Audio bitrate
                "-c:s", "mov_text",  # Subtitle codec for MP4
                "-metadata:s:s:0", "language=por",  # Portuguese subtitle
                "-metadata:s:s:0", "title=Legendas",  # Subtitle track title
                "-shortest",  # End when shortest input ends
                output_path
            ]
            logger.info(f"Creating MP4 video with black background and embedded subtitles (duration: {duration:.2f}s)")
        else:
            # Fallback: audio-only .mp4
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

@dataclass
class AudioQualityMetrics:
    clarity_score: float
    has_issues: bool = False
    warnings: List[str] = field(default_factory=list)
    rms_level: float = 0.0
    snr_estimate: float = 0.0
    clipping_detected: bool = False

class AudioQualityAnalyzer:
    """Analyzes audio quality and provides warnings to users"""
    def __init__(self):
        self.low_volume_threshold = 0.01  # RMS below this indicates very low volume
        self.normal_volume_threshold = 0.05  # RMS below this indicates low volume
        self.clipping_threshold = 0.95  # Peak amplitude above this indicates clipping
        self.poor_clarity_threshold = 0.4  # Clarity below this indicates poor quality

    def analyze_audio_quality(self, audio_path: str) -> AudioQualityMetrics:
        """Comprehensive audio quality analysis with user-facing warnings"""
        librosa = _get_librosa()
        if not librosa:
            logger.warning("Librosa not available, skipping quality analysis.")
            return AudioQualityMetrics(
                clarity_score=0.5,
                has_issues=False,
                warnings=[]
            )

        try:
            # Load first 30 seconds for analysis for quality assessment
            y, sr = librosa.load(audio_path, sr=16000, duration=30, mono=True)

            warnings = []
            has_issues = False

            # Root Mean Square (RMS) - volume level analysis
            rms = np.sqrt(np.mean(y**2))

            if rms < self.low_volume_threshold:
                warnings.append("Volume muito baixo detectado. Considere aumentar o volume da gravação.")
                has_issues = True
            elif rms < self.normal_volume_threshold:
                warnings.append("Volume baixo detectado. A qualidade da transcrição pode ser afetada.")
                has_issues = True

            # Clipping detection - audio distortion
            peak_amplitude = np.max(np.abs(y))
            clipping_detected = peak_amplitude >= self.clipping_threshold

            if clipping_detected:
                warnings.append("Distorção de áudio detectada (clipping). Reduza o volume da entrada.")
                has_issues = True

            # Spectral clarity - overall audio quality
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_mean = np.mean(spectral_centroids)
            clarity = 1.0 - min(1.0, np.std(spectral_centroids) / centroid_mean) if centroid_mean > 0 else 0.5

            if clarity < self.poor_clarity_threshold:
                warnings.append("AVISO: Qualidade de áudio ruim detectada. Verifique o ambiente e o microfone.")
                has_issues = True

            # Noise estimation - simple SNR estimate using energy ratio - calculate energy in high-frequency bands (noise) vs. low-frequency (speech)
            frame_length = 2048
            hop_length = 512
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
            snr_estimate = np.mean(spectral_rolloff) / (sr / 2)  # Normalized estimate
            if snr_estimate < 0.3:  
                warnings.append("Ruído de fundo excessivo detectado. A transcrição pode ter erros.")
                has_issues = True

            logger.info(f"Audio quality analysis: RMS={rms:.4f}, Clarity={clarity:.2f}, SNR_Est={snr_estimate:.2f}, Clipping={clipping_detected}, Issues={has_issues}")

            return AudioQualityMetrics(
                clarity_score=clarity,
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
                has_issues=False,
                warnings=[]
            )

# --- Live audio processor - disk buffering 

class RecordingState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    PROCESSING = "processing"
    COMPLETE = "complete"

class LiveAudioProcessor:
    """Temporary file storage to maintain low RAM uring recording"""
    def __init__(self, file_manager: FileManager):
        self.temp_dir = file_manager.get_data_path("temp")
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        logger.info("LiveAudioProcessor initialized with disk buffering strategy")

    def start_recording(self, session_id: str, sample_rate: int = 16000) -> Dict[str, Any]:
        """Start a new recording session with disk buffering"""
        with self._lock:
            # Allow restarting session (override any previous state)
            # This handles reconnection after abrupt disconnect
            if session_id in self.sessions:
                logger.info(f"Overriding existing session state for {session_id}")

            temp_file = self.temp_dir / f"{session_id}_{int(time.time())}.wav"

            start_time = time.time()
            self.sessions[session_id] = {
                "state": RecordingState.RECORDING,
                "temp_file": temp_file,
                "sample_rate": sample_rate,
                "start_time": start_time,
                "chunks_received": 0,
                "total_bytes": 0
            }

            logger.info(f"Recording started for session {session_id}, buffering to: {temp_file}")
            return {"status": "recording", "session_id": session_id, "temp_file": str(temp_file), "start_time": start_time}

    def pause_recording(self, session_id: str) -> Dict[str, str]:
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

    def resume_recording(self, session_id: str) -> Dict[str, str]:
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

    def process_audio_chunk(self, session_id: str, audio_chunk: bytes) -> Dict[str, Any]:
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

    def stop_recording(self, session_id: str) -> str:
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
                self._convert_webm_to_wav(webm_temp, temp_file_path, session.get("sample_rate", 16000))
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

    def _convert_webm_to_wav(self, input_path: str, output_path: str, sample_rate: int = 16000):
        """Convert WebM audio to WAV using FFMPEG"""
        try:
            # Try using static_ffmpeg first
            try:
                from static_ffmpeg import run
                ffmpeg_cmd = run.get_or_fetch_platform_executables_else_raise()
                ffmpeg_path = ffmpeg_cmd[0]  # Get ffmpeg executable path
                logger.debug(f"Using static_ffmpeg from: {ffmpeg_path}")
            except Exception as e:
                # Fallback to system ffmpeg
                ffmpeg_path = "ffmpeg"
                logger.debug(f"Using system ffmpeg: {e}")

            # FFMPEG convert WebM → WAV (mono, 16kHz, 16-bit PCM)
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

            # Run ffmpeg synchronously within async context
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,  # 30 sec timeout
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

    def complete_session(self, session_id: str) -> None:
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
        """Thread-safely get the state of a session"""
        with self._lock:
            if session_id in self.sessions:
                # Return a copy of the state-related parts of the session
                session = self.sessions[session_id]
                return {
                    "state": session.get("state"),
                    "total_bytes": session.get("total_bytes", 0)
                }
            return None

# --- Session management for live recording 

@dataclass
class SessionData:
    """All data associated with a single recording session"""
    session_id: str
    websocket: Optional["WebSocket"]
    format: Optional[str]
    started_at: "datetime"
    completed_at: Optional["datetime"] = None # Adicionado para o cleanup
    temp_file: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)
    status: str = "idle"

class SessionManager:
    """Centralized session lifecycle management - single source of truth for all active recording sessions"""

    def __init__(self, session_timeout_hours: int = 24):
        self.sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
        self.session_timeout_hours = session_timeout_hours
        logger.info(f"SessionManager initialized (cleanup timeout: {session_timeout_hours}h)")

    async def create_session(self, session_id: str, session_data: SessionData) -> None:
        async with self._lock:
            if session_id in self.sessions:
                raise ValueError(f"Session {session_id} already exists")
            self.sessions[session_id] = session_data
            logger.info(f"Session created: {session_id}")

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        async with self._lock:
            return self.sessions.get(session_id)

    async def remove_session(self, session_id: str) -> bool:
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
        return session_id in self.sessions

    async def get_active_session_count(self) -> int:
        """Number of currently active sessions"""
        async with self._lock:
            return len(self.sessions)

    async def get_all_session_ids(self) -> List[str]:
        """List of all active session IDs"""
        async with self._lock:
            return list(self.sessions.keys())

    async def cleanup_old_sessions(self):
        """
        Background task to cleanup expired sessions.
        - Removes sessions with status 'completed' after 1 hour.
        - Removes any other session older than 24 hours.
        """
        while True:
            try:
                now = datetime.now()
                one_hour_in_seconds = 3600
                twenty_four_hours_in_seconds = self.session_timeout_hours * 3600
                
                expired_sessions = []
                async with self._lock:
                    for sid, data in self.sessions.items():
                        # Rule 1: Cleanup completed sessions after 1 hour
                        if data.status == "completed" and data.completed_at:
                            if (now - data.completed_at).total_seconds() > one_hour_in_seconds:
                                expired_sessions.append(sid)
                                logger.info(f"Marking completed session for cleanup (1h expired): {sid}")
                        # Rule 2: Cleanup any other session older than 24 hours (failsafe)
                        elif (now - data.started_at).total_seconds() > twenty_four_hours_in_seconds:
                            expired_sessions.append(sid)
                            logger.info(f"Marking old session for cleanup (24h expired): {sid}")

                for session_id in set(expired_sessions): # Use set to avoid duplicates
                    await self.remove_session(session_id)

                # Run cleanup every 30 minutes
                await asyncio.sleep(1800)

            except Exception as e:
                logger.error(f"Session cleanup error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Retry in 1 minute
