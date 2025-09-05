import asyncio
import wave
import json
import logging
import os
import tempfile
import soundfile as sf
import numpy as np
import librosa # Added missing import
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
    nr = None # Define nr to avoid unbound variable error

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
        # Try openai-whisper package first (which should be available)
        import whisper as _whisper
        import torch as _torch

        whisper = _whisper
        torch = _torch
        WHISPER_AVAILABLE = True
        logger.info("ML dependencies (OpenAI Whisper) loaded successfully")
        
        # Test that whisper is working properly
        _ = whisper.available_models()
        logger.info(f"Available Whisper models: {whisper.available_models()}")
        
    except ImportError as e:
        logger.error(f"ML dependencies not available - Whisper import failed: {e}")
        logger.error("Please ensure 'openai-whisper' is properly installed: pip install openai-whisper")
        WHISPER_AVAILABLE = False
        whisper = None
        torch = None
    except Exception as e:
        logger.error(f"ML dependencies loaded but Whisper test failed: {e}")
        WHISPER_AVAILABLE = False
        whisper = None
        torch = None

    return WHISPER_AVAILABLE

def load_audio_librosa(audio_file, sr=16000, mono=True):
    """
    Load audio using librosa
    
    Args:
        audio_file: Path to audio file
        sr: Target sample rate (default 16000 for Whisper)
        mono: Convert to mono if True
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    audio_data, sample_rate = librosa.load(audio_file, sr=sr, mono=mono, dtype=np.float32)
    return audio_data, sample_rate

def preprocess_audio_advanced(audio_data, sample_rate):
    """
    Advanced audio preprocessing for better transcription quality
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: sample rate of audio
        
    Returns:
        numpy array: preprocessed audio
    """
    try:
        # Normalize audio level
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        # Apply LUFS normalization if available
        if PYLOUDNORM_AVAILABLE:
            try:
                meter = pyln.Meter(sample_rate)
                loudness = meter.integrated_loudness(audio_data)
                
                # Target -23 LUFS for speech with clipping protection  
                if not np.isinf(loudness) and not np.isnan(loudness):
                    # Only normalize if current loudness is significantly different
                    if abs(loudness - (-23.0)) > 3.0:  # 3 dB threshold
                        audio_data = pyln.normalize.loudness(audio_data, loudness, -23.0)
                        # Apply soft limiter to prevent clipping
                        audio_data = np.clip(audio_data, -0.95, 0.95)
                        logger.debug(f"Applied LUFS normalization with limiter: {loudness:.2f} -> -23.0 LUFS")
            except Exception as e:
                logger.warning(f"LUFS normalization failed: {e}")
        
        # Apply noise reduction if available
        if NOISEREDUCE_AVAILABLE and nr is not None:
            try:
                # Only apply if audio is long enough for stationary noise estimation
                if len(audio_data) / sample_rate > 2.0:
                    audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=True)
                    logger.debug("Applied stationary noise reduction")
                else:
                    # Use non-stationary for shorter audio
                    audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=False)
                    logger.debug("Applied non-stationary noise reduction")
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")
        
        # Apply high-pass filter to remove low-frequency noise
        try:
            # 80 Hz high-pass filter for speech
            sos = signal.butter(5, 80, btype='highpass', fs=sample_rate, output='sos')
            audio_data = signal.sosfilt(sos, audio_data)
            logger.debug("Applied high-pass filter at 80Hz")
        except Exception as e:
            logger.warning(f"High-pass filter failed: {e}")
        
        # Apply gentle compression to even out dynamics
        try:
            # Simple soft compression
            threshold = 0.5
            ratio = 4.0
            
            # Find samples above threshold
            above_threshold = np.abs(audio_data) > threshold
            
            # Apply compression
            compressed = np.copy(audio_data)
            compressed[above_threshold] = np.sign(audio_data[above_threshold]) * (
                threshold + (np.abs(audio_data[above_threshold]) - threshold) / ratio
            )
            
            audio_data = compressed
            logger.debug("Applied soft compression")
        except Exception as e:
            logger.warning(f"Audio compression failed: {e}")
        
        # Final normalization
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        # Ensure float32 consistency for Whisper
        return audio_data.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        return audio_data

class TranscriptionError(Exception):
    pass

# Model management removed - handled by main.py

class WhisperTranscriptionService:
    """Whisper-based transcription service with automatic model management"""

    def __init__(self):
        self._models = {}
        self._model_lock = asyncio.Lock()
        self._device = None # Lazy device detection

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

# Global service instance - lazy initialization
transcription_service = None

def get_transcription_service():
    """Get or create transcription service instance"""
    global transcription_service
    if transcription_service is None:
        transcription_service = WhisperTranscriptionService()
    return transcription_service

async def transcribe_audio_with_progress(
    wav_file: str,
    model_path: str,
    language_code: str,
    sample_rate: int = 16000
) -> AsyncGenerator[Tuple[int, List[Dict]], None]:
    """Whisper-based transcription with progress tracking"""
    try:
        logger.info(f"Starting Whisper transcription for {wav_file} with language {language_code}")

        # Check if Whisper is available - ensure ML imports are attempted first
        if not _ensure_ml_imports():
            logger.error("Whisper not available - ML dependencies failed to load")
            yield 100, [{"start": 0.0, "end": 1.0, "text": "Whisper not available"}]
            return

        # Load Whisper model
        service = get_transcription_service()
        model = await service.load_whisper_model(language_code)

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
                # Load and preprocess audio using librosa
                logger.info("Loading audio with librosa")
                audio_data, sr = load_audio_librosa(wav_file, sr=16000, mono=True)
                
                # Ensure consistent dtype for Whisper (fix dtype mismatch)
                audio_data = audio_data.astype(np.float32)
                
                if len(audio_data) == 0:
                    raise ValueError("No audio data loaded")

                logger.info(f"Audio loaded: {len(audio_data)} samples at {sr}Hz")

                # Apply advanced preprocessing
                audio_data = preprocess_audio_advanced(audio_data, sr)
                
                logger.info("Advanced audio preprocessing completed")

                # Transcribe with Whisper
                language = language_code if language_code in ['en'] else None # Use None for auto-detect

                result = model.transcribe(
                    audio_data,
                    language=language,
                    word_timestamps=WHISPER_CONFIG["word_timestamps"],
                    condition_on_previous_text=WHISPER_CONFIG["condition_on_previous_text"],
                    temperature=WHISPER_CONFIG["temperature"],
                    best_of=WHISPER_CONFIG["best_of"],
                    beam_size=WHISPER_CONFIG["beam_size"],
                    fp16=False  # Force FP32 to avoid CPU warnings
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
                if text and len(text) > 1: # Filter very short segments
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

# Additional utility functions

def get_audio_info(audio_file):
    """
    Get audio file information using soundfile
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        dict: Audio file information (duration, sample_rate, channels)
    """
    try:
        info = sf.info(audio_file)
        return {
            "duration": float(info.duration),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "method": "soundfile"
        }
        
    except Exception as e:
        logger.error(f"Failed to get audio info: {e}")
        return {
            "duration": 0.0,
            "sample_rate": 16000,
            "channels": 1,
            "method": "fallback"
        }

def convert_audio_format(input_file, output_file, target_sr=16000, mono=True):
    """
    Convert audio format using librosa
    
    Args:
        input_file: Input audio file path
        output_file: Output audio file path
        target_sr: Target sample rate
        mono: Convert to mono
        
    Returns:
        bool: Success status
    """
    try:
        # Load audio using librosa
        audio_data, sr = load_audio_librosa(input_file, sr=target_sr, mono=mono)
        
        # Save using soundfile
        sf.write(output_file, audio_data, target_sr)
        
        logger.info(f"Audio conversion successful: {input_file} -> {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return False

# Maintain backward compatibility with existing functions
def load_audio_librosa_fallback(audio_file, sr=16000, mono=True):
    """
    DEPRECATED: Legacy function for loading audio with librosa.
    Use load_audio_librosa instead.
    """
    logger.warning("Using deprecated load_audio_librosa_fallback, use load_audio_librosa instead")
    return load_audio_librosa(audio_file, sr=sr, mono=mono)