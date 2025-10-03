"""
Unified Audio Processing Module for TranscrevAI

Combines real-time recording capabilities with robust audio file loading and processing.
- AudioRecorder: A functional, thread-safe recorder using PyAudio for live capture.
- OptimizedAudioProcessor: Utilities for fast, efficient audio file manipulation using torchaudio.
- RobustAudioLoader: Fallback-based audio file loader for maximum compatibility.
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

# Lazy imports for performance and optional dependencies
_torch = None
_torchaudio = None
_librosa = None
_pyaudio = None

logger = logging.getLogger(__name__)

def _get_pyaudio():
    global _pyaudio
    if _pyaudio is None:
        try:
            import pyaudio
            _pyaudio = pyaudio
        except ImportError:
            logger.critical("PyAudio is not installed. Live recording will not work.")
            raise
    return _pyaudio

# --- Real-time Audio Recording --- #

class AudioRecorder:
    """A functional audio recorder using PyAudio for real-time microphone capture."""

    def __init__(self, output_file: str, sample_rate: int = 16000, websocket_manager=None, session_id=None):
        self.output_file = output_file
        self.sample_rate = sample_rate
        self.websocket_manager = websocket_manager
        self.session_id = session_id

        self._is_recording = False
        self._is_paused = False
        self._audio_queue = queue.Queue()
        self._recording_thread: Optional[threading.Thread] = None

        # Audio format configuration
        self.CHUNK = 1024
        self.FORMAT = _get_pyaudio().paInt16
        self.CHANNELS = 1

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

    def _recording_loop(self):
        """The main loop for the recording thread. Captures audio from the microphone."""
        try:
            p = _get_pyaudio().PyAudio()
            stream = p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK)
            logger.info(f"Microphone stream opened for session {self.session_id}")
        except Exception as e:
            logger.critical(f"Failed to open microphone stream: {e}")
            if self.websocket_manager and self.session_id:
                asyncio.run(self.websocket_manager.send_message(self.session_id, {
                    "type": "error",
                    "message": "Falha ao acessar o microfone. Verifique as permissões."
                }))
            return

        while self._is_recording:
            if not self._is_paused:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    self._audio_queue.put(data)
                except IOError as e:
                    logger.warning(f"IOError during recording for session {self.session_id}: {e}")
            else:
                time.sleep(0.1)

        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info(f"Microphone stream closed for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error closing microphone stream: {e}")

    async def start_recording(self):
        if self._is_recording:
            logger.warning("Recording is already active.")
            return
        self._is_recording = True
        self._is_paused = False
        self._recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self._recording_thread.start()
        logger.info(f"AudioRecorder started for: {self.output_file}")

    def pause_recording(self):
        if self._is_recording and not self._is_paused:
            self._is_paused = True
            logger.info(f"AudioRecorder paused: {self.output_file}")

    def resume_recording(self):
        if self._is_recording and self._is_paused:
            self._is_paused = False
            logger.info(f"AudioRecorder resumed: {self.output_file}")

    async def stop_recording(self):
        if not self._is_recording:
            return
        self._is_recording = False
        if self._recording_thread:
            self._recording_thread.join(timeout=2.0)

        logger.info(f"Recording stopped. Writing {self._audio_queue.qsize()} chunks to {self.output_file}")
        try:
            with wave.open(self.output_file, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(_get_pyaudio().PyAudio().get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                while not self._audio_queue.empty():
                    wf.writeframes(self._audio_queue.get())
            logger.info(f"Audio file saved successfully: {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save WAV file: {e}")
            raise

# --- Audio File Loading and Processing --- #

class OptimizedAudioProcessor:
    """Utilities for fast, efficient audio file manipulation using torchaudio."""
    @staticmethod
    def torchaudio_get_duration(audio_path: str) -> float:
        try:
            import torchaudio
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception:
            with sf.SoundFile(audio_path) as f:
                return len(f) / f.samplerate

class RobustAudioLoader:
    """Robust audio loading with fallback strategies for maximum compatibility."""
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        logger.info(f"Loading audio: {audio_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if os.path.getsize(audio_path) < 100:
            raise ValueError(f"Audio file is too small: {audio_path}")
        
        try: # Primary strategy: soundfile
            data, sr = sf.read(audio_path, dtype='float32')
        except Exception as e1:
            logger.warning(f"Soundfile failed: {e1}. Falling back to librosa.")
            try: # Fallback strategy: librosa
                import librosa
                data, sr = librosa.load(audio_path, sr=None, mono=False)
                data = data.T # Librosa loads as (channels, samples)
            except Exception as e2:
                raise RuntimeError(f"All audio loading strategies failed. Last error: {e2}")

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            import librosa
            data = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return data, sr