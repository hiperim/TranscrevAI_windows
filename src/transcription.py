import asyncio
import wave
import json
import logging
import os
import tempfile
import soundfile as sf
import numpy as np
import librosa  # Added missing import
from scipy import signal

# Enhanced audio preprocessing imports
try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    nr = None  # Define nr to avoid unbound variable error

from typing import AsyncGenerator, Tuple, List, Dict, Any
from pathlib import Path

from src.file_manager import FileManager
from config.app_config import WHISPER_MODEL_DIR, WHISPER_MODELS, WHISPER_CONFIG
from src.logging_setup import setup_app_logging

# Use proper logging setup first
logger = setup_app_logging(logger_name="transcrevai.transcription")

# Log library availability after logger is set up
if not PYLOUDNORM_AVAILABLE:
    logger.warning("pyloudnorm not available - LUFS normalization disabled")

if not NOISEREDUCE_AVAILABLE:
    logger.warning("noisereduce not available - noise reduction disabled")

# Lazy import for heavy ML dependencies
WHISPER_AVAILABLE = False
whisper = None
torch = None
_ml_imports_attempted = False

def _ensure_ml_imports():
    """Lazy import of heavy ML dependencies"""
    global WHISPER_AVAILABLE, whisper, torch, _ml_imports_attempted
    
    if _ml_imports_attempted:
        return WHISPER_AVAILABLE
    
    _ml_imports_attempted = True
    try:
        import whisper as _whisper
        import torch as _torch
        whisper = _whisper
        torch = _torch
        WHISPER_AVAILABLE = True
        logger.info("ML dependencies loaded successfully")
    except ImportError as e:
        logger.warning(f"ML dependencies not available: {e}")
        WHISPER_AVAILABLE = False
        whisper = None
        torch = None
    
    return WHISPER_AVAILABLE

class TranscriptionError(Exception):
    pass

# Model management removed - handled by main.py

class WhisperTranscriptionService:
    """Whisper-based transcription service with automatic model management"""

    def __init__(self):
        self._models = {}
        self._model_lock = asyncio.Lock()
        self._device = None  # Lazy device detection
        
    @property
    def device(self):
        """Lazy device detection only when needed"""
        if self._device is None:
            if _ensure_ml_imports() and torch:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Whisper device: {self._device}")
            else:
                self._device = "cpu"
        return self._device

    async def load_whisper_model(self, language_code: str) -> Any:
        """Load and cache Whisper model"""
        # Ensure ML dependencies are loaded
        if not _ensure_ml_imports():
            logger.error("Whisper not available")
            raise TranscriptionError("Whisper dependencies not installed")

        async with self._model_lock:
            if language_code not in self._models:
                try:
                    model_name = WHISPER_MODELS.get(language_code, "small")
                    # Download model in thread pool
                    loop = asyncio.get_event_loop()
                    model = await loop.run_in_executor(
                        None,
                        whisper.load_model,
                        model_name,
                        self.device,
                        str(WHISPER_MODEL_DIR)
                    )
                    self._models[language_code] = model
                    logger.info(f"Whisper model '{model_name}' loaded for {language_code}")
                except Exception as e:
                    logger.error(f"Failed to load Whisper model for {language_code}: {e}")
                    raise TranscriptionError(f"Whisper model loading failed: {str(e)}")

        return self._models[language_code]

# Global service instance
transcription_service = WhisperTranscriptionService()

async def transcribe_audio_with_progress(
    wav_file: str,
    model_path: str,
    language_code: str,
    sample_rate: int = 16000
) -> AsyncGenerator[Tuple[int, List[Dict]], None]:
    """Whisper-based transcription with progress tracking"""
    try:
        logger.info(f"Starting Whisper transcription for {wav_file} with language {language_code}")

        # Check if Whisper is available
        if not WHISPER_AVAILABLE:
            logger.error("Whisper not available")
            yield 100, [{"start": 0.0, "end": 1.0, "text": "Whisper not available"}]
            return

        # Load Whisper model
        model = await transcription_service.load_whisper_model(language_code)

        # Validate audio file
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"Audio file not found: {wav_file}")

        if os.path.getsize(wav_file) == 0:
            raise ValueError("Audio file is empty")

        # Yield initial progress
        yield 10, []

        # Process audio with Whisper
        loop = asyncio.get_event_loop()

        def transcribe_with_whisper():
            try:
                # Load and preprocess audio
                audio_data, sr = librosa.load(wav_file, sr=16000, mono=True)
                if len(audio_data) == 0:
                    raise ValueError("No audio data loaded")

                # Apply audio preprocessing for better Whisper performance
                if NOISEREDUCE_AVAILABLE and nr is not None:  # Fixed unbound variable check
                    try:
                        audio_data = nr.reduce_noise(y=audio_data, sr=sr)
                        logger.info("Applied noise reduction")
                    except Exception as e:
                        logger.warning(f"Noise reduction failed: {e}")

                # Normalize audio
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8

                # Transcribe with Whisper
                language = language_code if language_code in ['en'] else None  # Use None for auto-detect
                result = model.transcribe(
                    audio_data,
                    language=language,
                    word_timestamps=WHISPER_CONFIG["word_timestamps"],
                    condition_on_previous_text=WHISPER_CONFIG["condition_on_previous_text"],
                    temperature=WHISPER_CONFIG["temperature"],
                    best_of=WHISPER_CONFIG["best_of"],
                    beam_size=WHISPER_CONFIG["beam_size"]
                )

                # Convert Whisper result to TranscrevAI format
                transcription_data = []
                if "segments" in result:
                    for segment in result["segments"]:
                        # Extract word-level timestamps if available
                        if "words" in segment and segment["words"]:
                            for word in segment["words"]:
                                transcription_data.append({
                                    "start": word["start"],
                                    "end": word["end"],
                                    "text": word["word"].strip(),
                                    "confidence": word.get("probability", 1.0)
                                })
                        else:
                            # Fallback to segment-level
                            transcription_data.append({
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": segment["text"].strip(),
                                "confidence": 1.0
                            })
                else:
                    # Fallback: single segment
                    transcription_data.append({
                        "start": 0.0,
                        "end": len(audio_data) / sr,
                        "text": result.get("text", "").strip(),
                        "confidence": 1.0
                    })

                return transcription_data

            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
                raise TranscriptionError(f"Whisper error: {str(e)}")

        # Yield progress updates
        yield 50, []

        # Execute transcription
        transcription_data = await loop.run_in_executor(None, transcribe_with_whisper)

        yield 90, transcription_data

        # Apply post-processing
        def post_process_whisper_results(segments):
            """Post-process Whisper transcription results"""
            if not segments:
                return segments

            processed = []
            for segment in segments:
                text = segment.get('text', '').strip()
                if text and len(text) > 1:  # Filter very short segments
                    processed.append(segment)

            return processed

        # Apply post-processing
        transcription_data = post_process_whisper_results(transcription_data)

        # Final yield
        yield 100, transcription_data

        logger.info(f"Whisper transcription completed: {len(transcription_data)} segments")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise TranscriptionError(f"Transcription error: {str(e)}")