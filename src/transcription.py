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
import re

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

class ContextualCorrector:
    """Lightweight contextual corrections for common transcription errors"""
    
    def __init__(self):
        self.corrections = {
            "pt": {
                # Portuguese (Brazilian) common corrections
                r'\bvoce\b': 'você',
                r'\besta\b': 'está',  
                r'\bmedico\b': 'médico',
                r'\btelefone\b': 'telefone',
                r'\bopcao\b': 'opção',
                r'\binformacao\b': 'informação',
                r'\boperacao\b': 'operação',
                r'\bcomunicacao\b': 'comunicação',
                r'\beducacao\b': 'educação',
                r'\bsituacao\b': 'situação',
                r'\brapido\b': 'rápido',
                r'\bbasico\b': 'básico',
                r'\bpratico\b': 'prático',
                r'\bautomatico\b': 'automático'
            },
            "en": {
                # English common corrections
                r'\bthere going\b': "they're going",
                r'\byour going\b': "you're going",
                r'\bits going\b': "it's going",
                r'\bthats\b': "that's",
                r'\bdont\b': "don't",
                r'\bcant\b': "can't",
                r'\bwont\b': "won't",
                r'\byour\b(?=\s+(are|going|looking|taking))': "you're",
                r'\btheir\b(?=\s+(going|looking|taking))': "they're",
                r'\bits\b(?=\s+(going|going|looking|taking))': "it's"
            },
            "es": {
                # Spanish common corrections
                r'\bmedico\b': 'médico',
                r'\btelefono\b': 'teléfono',
                r'\brapido\b': 'rápido',
                r'\bmusica\b': 'música',
                r'\bingles\b': 'inglés',
                r'\bfacil\b': 'fácil',
                r'\butil\b': 'útil',
                r'\binformacion\b': 'información',
                r'\boperacion\b': 'operación',
                r'\bcomunicacion\b': 'comunicación',
                r'\beducacion\b': 'educación',
                r'\bsituacion\b': 'situación',
                r'\bautomatico\b': 'automático',
                r'\bpractico\b': 'práctico'
            }
        }
        
        # Compile regex patterns for better performance
        self.compiled_patterns = {}
        for language, patterns in self.corrections.items():
            self.compiled_patterns[language] = {
                re.compile(pattern, re.IGNORECASE): replacement
                for pattern, replacement in patterns.items()
            }
    
    def apply_corrections(self, text: str, language: str, confidence: float = 1.0) -> str:
        """
        Apply language-specific corrections to text
        
        Args:
            text: Text to correct
            language: Language code (pt, en, es)
            confidence: Confidence score (apply corrections only to low confidence words)
        
        Returns:
            Corrected text
        """
        try:
            # Only apply corrections to low-confidence transcriptions
            if confidence > 0.7:
                return text
            
            if language not in self.compiled_patterns:
                return text
            
            corrected_text = text
            for pattern, replacement in self.compiled_patterns[language].items():
                corrected_text = pattern.sub(replacement, corrected_text)
            
            # Log corrections if any were made
            if corrected_text != text:
                logger.debug(f"Contextual correction applied [{language}]: '{text}' -> '{corrected_text}'")
                
            return corrected_text
            
        except Exception as e:
            logger.warning(f"Contextual correction failed: {e}")
            return text

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

def preprocess_audio_realtime(audio_data, sample_rate):
    """
    Lightweight preprocessing optimized for real-time performance
    Replaces heavy pipeline with minimal, fast operations
    """
    try:
        # Simple normalization only (fastest)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        # Optional: Simple high-pass filter only for real-time
        if sample_rate >= 16000:
            from scipy import signal
            # Lightweight 2nd order filter instead of complex pipeline
            b, a = signal.butter(2, 80, btype='highpass', fs=sample_rate)
            audio_data = signal.filtfilt(b, a, audio_data)
        
        return audio_data.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"Real-time preprocessing failed: {e}, using simple normalization")
        # Fallback to simple normalization
        if np.max(np.abs(audio_data)) > 0:
            return (audio_data / np.max(np.abs(audio_data)) * 0.8).astype(np.float32)
        return audio_data.astype(np.float32)

async def preprocess_audio_advanced(audio_data, sample_rate):
    """
    Advanced audio preprocessing for better transcription quality - now properly async
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: sample rate of audio
        
    Returns:
        numpy array: preprocessed audio
    """
    try:
        # Run CPU-intensive operations in thread pool
        loop = asyncio.get_event_loop()
        
        def _cpu_intensive_processing():
            processed_data = audio_data.copy()
            
            # Normalize audio level
            if np.max(np.abs(processed_data)) > 0:
                processed_data = processed_data / np.max(np.abs(processed_data)) * 0.8
            
            # Apply LUFS normalization if available
            if PYLOUDNORM_AVAILABLE:
                try:
                    meter = pyln.Meter(sample_rate)
                    loudness = meter.integrated_loudness(processed_data)
                    
                    # Target -23 LUFS for speech with clipping protection  
                    if not np.isinf(loudness) and not np.isnan(loudness):
                        # Only normalize if current loudness is significantly different
                        if abs(loudness - (-23.0)) > 3.0:  # 3 dB threshold
                            processed_data = pyln.normalize.loudness(processed_data, loudness, -23.0)
                            # Apply soft limiter to prevent clipping
                            processed_data = np.clip(processed_data, -0.95, 0.95)
                except Exception as e:
                    logger.warning(f"LUFS normalization failed: {e}")
            
            # Apply noise reduction if available
            if NOISEREDUCE_AVAILABLE and nr is not None:
                try:
                    # Only apply if audio is long enough for stationary noise estimation
                    if len(processed_data) / sample_rate > 2.0:
                        processed_data = nr.reduce_noise(y=processed_data, sr=sample_rate, stationary=True)
                    else:
                        # Use non-stationary for shorter audio
                        processed_data = nr.reduce_noise(y=processed_data, sr=sample_rate, stationary=False)
                except Exception as e:
                    logger.warning(f"Noise reduction failed: {e}")
            
            # Apply high-pass filter to remove low-frequency noise
            try:
                # 80 Hz high-pass filter for speech
                sos = signal.butter(5, 80, btype='highpass', fs=sample_rate, output='sos')
                processed_data = signal.sosfilt(sos, processed_data)
            except Exception as e:
                logger.warning(f"High-pass filter failed: {e}")
            
            # Apply gentle compression to even out dynamics
            try:
                # Simple soft compression
                threshold = 0.5
                ratio = 4.0
                
                # Find samples above threshold
                above_threshold = np.abs(processed_data) > threshold
                
                # Apply compression
                compressed = np.copy(processed_data)
                compressed[above_threshold] = np.sign(processed_data[above_threshold]) * (
                    threshold + (np.abs(processed_data[above_threshold]) - threshold) / ratio
                )
                
                processed_data = compressed
            except Exception as e:
                logger.warning(f"Audio compression failed: {e}")
            
            # Final normalization
            if np.max(np.abs(processed_data)) > 0:
                processed_data = processed_data / np.max(np.abs(processed_data)) * 0.8
            
            # Ensure float32 consistency for Whisper
            return processed_data.astype(np.float32)
        
        # Execute in thread pool to avoid blocking
        return await loop.run_in_executor(None, _cpu_intensive_processing)
        
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

                # Transcribe with Whisper - Enhanced with language-specific optimization
                from config.app_config import WHISPER_MODELS
                language = language_code if language_code in WHISPER_MODELS else None
                logger.info(f"Using Whisper with language: {language} (requested: {language_code})")

                # Get language-specific configuration or fallback
                language_configs = WHISPER_CONFIG.get("language_configs", {})
                if language_code in language_configs:
                    config = language_configs[language_code]
                    logger.info(f"Using optimized configuration for {language_code}")
                else:
                    config = WHISPER_CONFIG.get("fallback_config", {})
                    logger.info(f"Using fallback configuration for {language_code}")

                # Enhanced diagnostic logging
                logger.info(f"Whisper transcription parameters:")
                logger.info(f"  - Language: {language}")
                logger.info(f"  - Model: {WHISPER_MODELS.get(language_code, 'auto')}")
                logger.info(f"  - Temperature: {config.get('temperature', (0.0, 0.2))}")
                logger.info(f"  - Best of: {config.get('best_of', 5)}")
                logger.info(f"  - Beam size: {config.get('beam_size', 5)}")
                logger.info(f"  - No speech threshold: {config.get('no_speech_threshold', 0.6)}")
                logger.info(f"  - Initial prompt: {config.get('initial_prompt', '')}")

                result = model.transcribe(
                    audio_data,
                    language=language,
                    word_timestamps=WHISPER_CONFIG["word_timestamps"],
                    condition_on_previous_text=WHISPER_CONFIG["condition_on_previous_text"],
                    temperature=config.get("temperature", (0.0, 0.2)),
                    best_of=config.get("best_of", 5),
                    beam_size=config.get("beam_size", 5),
                    patience=config.get("patience", 1.0),
                    length_penalty=config.get("length_penalty", 1.0),
                    no_speech_threshold=config.get("no_speech_threshold", 0.6),
                    initial_prompt=config.get("initial_prompt", ""),
                    fp16=False  # Force FP32 to avoid CPU warnings
                )
                
                logger.info(f"Whisper detected language: {result.get('language', 'unknown')}")

                # Initialize contextual corrector
                corrector = ContextualCorrector()

                # Convert Whisper result to TranscrevAI format with contextual corrections
                transcription_data = []

                if "segments" in result:
                    for segment in result["segments"]:
                        # Extract word-level timestamps if available
                        if "words" in segment and segment["words"]:
                            for word in segment["words"]:
                                original_text = word["word"].strip()
                                confidence = word.get("probability", 1.0)
                                
                                # Apply contextual corrections
                                corrected_text = corrector.apply_corrections(
                                    original_text, language_code, confidence
                                )
                                
                                transcription_data.append({
                                    "start": word["start"],
                                    "end": word["end"],
                                    "text": corrected_text,
                                    "confidence": confidence
                                })
                        else:
                            # Fallback to segment-level
                            original_text = segment["text"].strip()
                            corrected_text = corrector.apply_corrections(
                                original_text, language_code, 1.0
                            )
                            
                            transcription_data.append({
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": corrected_text,
                                "confidence": 1.0
                            })
                else:
                    # Fallback: single segment
                    original_text = result.get("text", "").strip()
                    corrected_text = corrector.apply_corrections(
                        original_text, language_code, 1.0
                    )
                    
                    transcription_data.append({
                        "start": 0.0,
                        "end": len(audio_data) / sr,
                        "text": corrected_text,
                        "confidence": 1.0
                    })

                # Log total number of corrections applied
                corrections_count = sum(1 for item in transcription_data if item["confidence"] <= 0.7)
                if corrections_count > 0:
                    logger.info(f"Applied contextual corrections to {corrections_count} low-confidence segments")

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