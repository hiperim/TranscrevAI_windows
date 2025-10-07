# FINALIZED AND CORRECTED - Enhanced Audio Processing Module
"""
Unified Audio Processing Module for TranscrevAI.
Combines real-time recording, VAD pre-processing, dynamic quantization analysis,
and robust audio file loading into a single, comprehensive toolkit.
"""

import asyncio
import logging
import os
import queue
import threading
import time
import wave
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import soundfile as sf

# Lazy imports for optional, heavy dependencies
_pyaudio = None
_librosa = None

logger = logging.getLogger(__name__)

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

# --- VAD (Voice Activity Detection) Pre-processor --- #

class VADMode(Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"

@dataclass
class VADSegment:
    start: float
    end: float

class VADPreprocessor:
    """Uses a mock Silero VAD model to detect speech timestamps in an audio file."""
    def __init__(self):
        self.sample_rate = 16000

    async def get_speech_timestamps(self, audio_path: str, vad_mode: VADMode = VADMode.BALANCED) -> List[Dict[str, float]]:
        librosa = _get_librosa()
        if not librosa:
            raise ImportError("Librosa is required for VAD processing.")
        
        try:
            audio_data, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            # Use librosa's built-in effect to split by non-silent parts.
            # This is a robust way to simulate a VAD.
            speech_chunks = librosa.effects.split(audio_data, top_db=30) # 30dB is a reasonable threshold
            timestamps = [{'start': start / self.sample_rate, 'end': end / self.sample_rate} for start, end in speech_chunks]
            
            logger.info(f"VAD processing found {len(timestamps)} speech segments.")
            return timestamps
        except Exception as e:
            logger.error(f"VAD processing failed: {e}")
            return []

# --- Dynamic Quantization --- #

class QuantizationLevel(Enum):
    INT8 = "int8"
    FLOAT16 = "float16"

@dataclass
class AudioQualityMetrics:
    clarity_score: float
    recommended_quantization: QuantizationLevel

class AudioQualityAnalyzer:
    """Analyzes audio quality to recommend an optimal quantization level."""
    def analyze_audio_quality(self, audio_path: str) -> AudioQualityMetrics:
        librosa = _get_librosa()
        if not librosa:
            logger.warning("Librosa not available, defaulting to INT8 quantization.")
            return AudioQualityMetrics(clarity_score=0.5, recommended_quantization=QuantizationLevel.INT8)

        try:
            y, sr = librosa.load(audio_path, sr=16000, duration=30)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_mean = np.mean(spectral_centroids)
            clarity = 1.0 - min(1.0, np.std(spectral_centroids) / centroid_mean) if centroid_mean > 0 else 0.5
            quantization = QuantizationLevel.FLOAT16 if clarity > 0.8 else QuantizationLevel.INT8
            return AudioQualityMetrics(clarity_score=clarity, recommended_quantization=quantization)
        except Exception as e:
            logger.error(f"Audio quality analysis failed: {e}")
            return AudioQualityMetrics(clarity_score=0.5, recommended_quantization=QuantizationLevel.INT8)

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

# --- Audio File Loading --- #

class RobustAudioLoader:
    """Robust audio loading with fallback strategies for maximum compatibility."""
    def load_audio(self, file_path: Union[str, Path], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        file_path = Path(file_path)
        if not file_path.exists(): raise FileNotFoundError(f"Audio file not found: {file_path}")
        if file_path.stat().st_size < 100: raise ValueError(f"Audio file is too small: {file_path}")
        
        try:
            data, sr = sf.read(str(file_path), dtype='float32')
        except Exception as e1:
            logger.warning(f"Soundfile failed to load {file_path}: {e1}. Falling back to librosa.")
            librosa = _get_librosa()
            if not librosa: raise RuntimeError("Audio loading failed: librosa not available.")
            data, sr = librosa.load(str(file_path), sr=None, mono=False)
            if len(data.shape) > 1: data = data.mean(axis=0)

        if len(data.shape) > 1: data = np.mean(data, axis=1)
        if sr != target_sr:
            librosa = _get_librosa()
            if not librosa: raise RuntimeError("Resampling failed: librosa not available.")
            data = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return data.astype(np.float32), int(sr)
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
    def __init__(self, temp_dir: str = "data/temp"):
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
                if current_state != RecordingState.IDLE:
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
            if session["state"] != RecordingState.RECORDING:
                raise ValueError(f"Cannot pause session in state: {session['state'].value}")

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
            if session["state"] != RecordingState.PAUSED:
                raise ValueError(f"Cannot resume session in state: {session['state'].value}")

            session["state"] = RecordingState.RECORDING
            session["resume_time"] = time.time()
            logger.info(f"Recording resumed for session {session_id}")
            return {"status": "recording", "session_id": session_id}

    async def process_audio_chunk(self, session_id: str, audio_chunk: bytes) -> Dict[str, Any]:
        """
        Write audio chunk to disk buffer (low memory operation).
        Compliance: Keeps RAM usage minimal during recording.
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            session = self.sessions[session_id]
            if session["state"] not in [RecordingState.RECORDING, RecordingState.PAUSED]:
                raise ValueError(f"Cannot process chunk in state: {session['state'].value}")

            # Write to disk buffer (not RAM)
            with open(session["temp_file"], "ab") as f:
                f.write(audio_chunk)

            session["chunks_received"] += 1
            session["total_bytes"] += len(audio_chunk)

            return {
                "status": "chunk_saved",
                "chunks_received": session["chunks_received"],
                "total_bytes": session["total_bytes"]
            }

    async def stop_recording(self, session_id: str) -> str:
        """
        Stop recording and return path to complete audio file for processing.
        Transitions to PROCESSING state.
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            session = self.sessions[session_id]
            if session["state"] not in [RecordingState.RECORDING, RecordingState.PAUSED]:
                raise ValueError(f"Cannot stop session in state: {session['state'].value}")

            session["state"] = RecordingState.PROCESSING
            session["stop_time"] = time.time()
            duration = session["stop_time"] - session["start_time"]

            temp_file_path = str(session["temp_file"])
            logger.info(f"Recording stopped for session {session_id}. Duration: {duration:.2f}s, File: {temp_file_path}")

            return temp_file_path

    async def complete_session(self, session_id: str) -> None:
        """Mark session as complete and cleanup temporary file"""
        with self._lock:
            if session_id not in self.sessions:
                return

            session = self.sessions[session_id]
            session["state"] = RecordingState.COMPLETE

            # Cleanup temp file
            if session["temp_file"].exists():
                session["temp_file"].unlink()
                logger.info(f"Temporary file deleted for session {session_id}")

            # Remove session after brief retention
            del self.sessions[session_id]

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state"""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None

            return {
                "state": session["state"].value,
                "chunks_received": session.get("chunks_received", 0),
                "total_bytes": session.get("total_bytes", 0),
                "duration": time.time() - session["start_time"] if "start_time" in session else 0
            }
