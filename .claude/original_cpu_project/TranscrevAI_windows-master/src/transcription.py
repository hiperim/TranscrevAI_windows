# Enhanced transcription with medium model support
import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, AsyncGenerator, Tuple
import tempfile
import os

from config.app_config import (
    WHISPER_CONFIG, 
    WHISPER_MODELS,
    ADAPTIVE_PROMPTS,
    QUALITY_CONFIG
)
from src.logging_setup import setup_app_logging

logger = setup_app_logging(logger_name="transcrevai.transcription")

# Lazy imports for better performance
_whisper = None
_librosa = None
_soundfile = None

# Global model cache for faster loading
_model_cache = {}
_cache_lock = asyncio.Lock()

def get_whisper():
    """Lazy import whisper"""
    global _whisper
    if _whisper is None:
        import whisper
        _whisper = whisper
    return _whisper

def get_librosa():
    """Lazy import librosa"""
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa

def get_soundfile():
    """Lazy import soundfile"""
    global _soundfile
    if _soundfile is None:
        import soundfile
        _soundfile = soundfile
    return _soundfile

async def get_cached_model(model_name):
    """Get cached Whisper model or load if not cached"""
    async with _cache_lock:
        if model_name not in _model_cache:
            logger.info(f"Loading and caching model: {model_name}")
            whisper = get_whisper()
            try:
                _model_cache[model_name] = whisper.load_model(model_name)
            except Exception as e:
                logger.warning(f"Failed to load {model_name}, falling back to medium: {e}")
                model_name = "medium"
                _model_cache[model_name] = whisper.load_model("medium")
        else:
            logger.info(f"Using cached model: {model_name}")
        
        return _model_cache[model_name]

# CRITICAL FIX: Enhanced transcription with adaptive processing
async def transcribe_audio_with_progress(
    audio_file: str, 
    language: str = "en", 
    sample_rate: int = 16000,
    audio_input_type: str = "neutral",
    processing_profile: str = "balanced"
) -> AsyncGenerator[Tuple[int, List[Dict[str, Any]]], None]:
    """
    Enhanced transcription using medium models for pt, en, es
    """
    try:
        logger.info(f"Starting enhanced transcription: {audio_file}, language: {language}, type: {audio_input_type}")
        
        # Step 1: Get configuration for medium model
        yield 10, []
        whisper_config = get_whisper_config(language, audio_input_type)
        model_name = WHISPER_MODELS.get(language, "medium")
        
        logger.info(f"Using model: {model_name} for {language}")
        
        # Step 3 & 4: Parallel model loading and audio preparation
        yield 30, []
        model_task = asyncio.create_task(get_cached_model(model_name))
        audio_task = asyncio.create_task(prepare_audio_for_transcription(audio_file, sample_rate))
        
        # Wait for both to complete
        yield 40, []
        model, audio_data = await asyncio.gather(model_task, audio_task)
        
        # Step 5: Transcribe with adaptive settings
        yield 50, []
        
        # Get the appropriate initial prompt from whisper_config
        initial_prompt = whisper_config.get("initial_prompt", "")
        
        transcribe_options = {
            "language": language,
            "word_timestamps": True,
            "initial_prompt": initial_prompt,
            **{k: v for k, v in whisper_config.items() if k not in ["model", "initial_prompt"]}
        }
        
        logger.info(f"Transcription options: {transcribe_options}")
        
        # Perform transcription
        yield 70, []
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: model.transcribe(audio_data, **transcribe_options)
        )
        
        # Step 6: Process and enhance results
        yield 90, []
        segments = process_transcription_result(result, language)
        
        # Step 7: Apply contextual corrections
        enhanced_segments = apply_contextual_corrections(segments, language)
        
        yield 100, enhanced_segments
        
        logger.info(f"Enhanced transcription completed: {len(enhanced_segments)} segments")
        
    except Exception as e:
        logger.error(f"Enhanced transcription failed: {e}")
        yield 100, []

async def prepare_audio_for_transcription(audio_file: str, target_sr: int = 16000) -> np.ndarray:
    """
    Enhanced audio preparation optimized for Whisper performance
    """
    try:
        sf = get_soundfile()
        librosa = get_librosa()
        
        # Load audio file
        audio_data, sr = sf.read(audio_file)
        
        # Convert to mono if stereo and ensure consistent dtype
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Ensure consistent float32 dtype to avoid dtype conflicts
        audio_data = audio_data.astype(np.float32)
        
        # Smart resampling - only if significantly different
        if abs(sr - target_sr) > 2000:  # Only resample if very different
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            audio_data = audio_data.astype(np.float32)
        
        # Enhanced normalization for optimal Whisper performance
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            # Normalize to optimal RMS level for Whisper (around 0.15-0.25)
            target_rms = 0.2
            audio_data = audio_data * (target_rms / rms)
            
            # Clip to prevent distortion
            audio_data = np.clip(audio_data, -0.9, 0.9)
        
        # Adaptive filtering based on audio characteristics
        audio_variance = np.var(audio_data)
        if audio_variance < 0.001:  # Very quiet audio
            # Amplify quiet sections while preserving dynamics
            audio_data = audio_data * 2.0
        
        # High-pass filter for noise reduction (optimized for speech)
        from scipy import signal
        if target_sr >= 16000:  # Only apply if sufficient sample rate
            sos = signal.butter(3, 85, btype='highpass', fs=target_sr, output='sos')
            audio_data = signal.sosfilt(sos, audio_data)
        
        # Final dtype conversion to float32 for compatibility
        return audio_data.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Audio preparation failed: {e}")
        # Fallback: return raw audio with consistent dtype
        sf = get_soundfile()
        audio_data, _ = sf.read(audio_file)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        return audio_data.astype(np.float32)

def get_whisper_config(language_code, audio_input_type="neutral"):
    """
    Get Whisper configuration for medium model
    """
    base_config = WHISPER_CONFIG["language_configs"].get(language_code, WHISPER_CONFIG["language_configs"]["en"])
    
    config = {
        **base_config,
        "model": WHISPER_MODELS.get(language_code, "medium")
    }
    
    # Adjust prompt based on audio type
    if language_code in ADAPTIVE_PROMPTS and audio_input_type in ADAPTIVE_PROMPTS[language_code]:
        config["initial_prompt"] = ADAPTIVE_PROMPTS[language_code][audio_input_type]
    
    return config

def process_transcription_result(result, language):
    """
    CRITICAL FIX: Enhanced processing of transcription results
    """
    segments = []
    
    if not result.get("segments"):
        return segments
    
    for i, segment in enumerate(result["segments"]):
        # Basic segment processing
        processed_segment = {
            "id": i,
            "start": segment.get("start", 0),
            "end": segment.get("end", 0), 
            "text": segment.get("text", "").strip(),
            "confidence": calculate_segment_confidence(segment),
            "language": language,
        }
        
        # Add word-level timestamps if available
        if segment.get("words"):
            processed_segment["words"] = [
                {
                    "word": word.get("word", ""),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "confidence": word.get("probability", 0.5)
                }
                for word in segment["words"]
            ]
        
        # Quality filtering
        if should_include_segment(processed_segment):
            segments.append(processed_segment)
    
    return segments

def calculate_segment_confidence(segment):
    """Calculate confidence score for a segment"""
    # If we have word-level probabilities, use those
    if segment.get("words"):
        probs = [word.get("probability", 0.5) for word in segment["words"]]
        return sum(probs) / len(probs) if probs else 0.5
    
    # Fallback to segment-level confidence
    return segment.get("avg_logprob", -1.0) + 1.0  # Convert from log prob

def should_include_segment(segment):
    """Determine if a segment should be included based on quality thresholds"""
    min_confidence = QUALITY_CONFIG["transcription"]["min_confidence"]
    min_duration = QUALITY_CONFIG["transcription"]["segment_min_duration"]
    
    if segment["confidence"] < min_confidence:
        return False
        
    duration = segment["end"] - segment["start"]
    if duration < min_duration:
        return False
        
    if not segment["text"] or len(segment["text"].strip()) < 2:
        return False
        
    return True

# CRITICAL FIX: Contextual corrections for Portuguese, English, and Spanish
def apply_contextual_corrections(segments, language):
    """
    CRITICAL FIX: Apply language-specific contextual corrections
    """
    if language == "pt":
        return apply_portuguese_corrections(segments)
    elif language == "en":
        return apply_english_corrections(segments)
    elif language == "es":
        return apply_spanish_corrections(segments)
    else:
        return segments

def apply_portuguese_corrections(segments):
    """Apply Portuguese-specific corrections"""
    pt_corrections = {
        # Common transcription errors in Portuguese
        "mas": "mas",  # conjunction vs "more"
        "mais": "mais",  # "more" 
        "então": "então",  # "so/then"
        "né": "né",  # informal "right?"
        "tá": "tá",  # informal "ok"
        "pra": "para",  # informal "to/for"
        "pro": "para o",  # informal contraction
        "dum": "de um",  # contraction correction
        "numa": "em uma",  # contraction correction
        "numa": "em uma",
        # Question words
        "que que": "o que",  # "what" correction
        "cadê": "onde está",  # "where is"
        # Common phrases
        "a gente": "a gente",  # "we" (colloquial)
        "tipo assim": "tipo assim",  # "like this"
        "sabe": "sabe"  # "you know"
    }
    
    return apply_text_corrections(segments, pt_corrections)

def apply_english_corrections(segments):
    """Apply English-specific corrections"""
    en_corrections = {
        # Common transcription errors
        "gonna": "going to",
        "wanna": "want to", 
        "gotta": "got to",
        "kinda": "kind of",
        "sorta": "sort of",
        "dunno": "don't know",
        "yeah": "yes",
        "nah": "no",
        "um": "um",  # Keep filler words
        "uh": "uh",
        # Contractions (keep as is for natural speech)
        "can't": "can't",
        "won't": "won't",
        "don't": "don't"
    }
    
    return apply_text_corrections(segments, en_corrections)

def apply_spanish_corrections(segments):
    """Apply Spanish-specific corrections"""
    es_corrections = {
        # Common transcription errors in Spanish
        "pos": "pues",  # "well"
        "ta": "está",  # informal "is"
        "pa": "para",  # informal "for"
        "pal": "para el",  # informal contraction
        "na": "nada",  # informal "nothing"
        # Question words
        "que": "qué",  # add accent for questions
        "como": "cómo",  # add accent for questions
        "cuando": "cuándo",  # add accent for questions
        "donde": "dónde",  # add accent for questions
        # Common phrases
        "o sea": "o sea",  # "I mean"
        "tipo": "tipo",  # "like"
        "sabes": "sabes"  # "you know"
    }
    
    return apply_text_corrections(segments, es_corrections)

def apply_text_corrections(segments, corrections_dict):
    """Apply text corrections to segments"""
    corrected_segments = []
    
    for segment in segments:
        corrected_segment = segment.copy()
        text = segment["text"]
        
        # Apply word-level corrections
        for wrong, correct in corrections_dict.items():
            text = text.replace(f" {wrong} ", f" {correct} ")
            text = text.replace(f" {wrong}.", f" {correct}.")
            text = text.replace(f" {wrong},", f" {correct},")
            text = text.replace(f" {wrong}?", f" {correct}?")
            text = text.replace(f" {wrong}!", f" {correct}!")
            
            # Handle start and end of text
            if text.startswith(wrong + " "):
                text = correct + text[len(wrong):]
            if text.endswith(" " + wrong):
                text = text[:-len(wrong)] + correct
        
        corrected_segment["text"] = text.strip()
        corrected_segments.append(corrected_segment)
    
    return corrected_segments

# Export functions for main usage
def get_transcription_functions():
    """Get transcription functions for lazy import"""
    return transcribe_audio_with_progress