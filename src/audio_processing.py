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