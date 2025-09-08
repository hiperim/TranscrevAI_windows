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
        
        # Step 3: Load model
        yield 30, []
        whisper = get_whisper()
        try:
            model = whisper.load_model(model_name)
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to medium: {e}")
            model = whisper.load_model("medium")
        
        # Step 4: Prepare audio
        yield 40, []
        audio_data = await prepare_audio_for_transcription(audio_file, sample_rate)
        
        # Step 5: Transcribe with adaptive settings
        yield 50, []
        
        # Get the appropriate initial prompt
        # Use configured prompt from whisper_config
        
        transcribe_options = {
            "language": language,
            "word_timestamps": True,
            "initial_prompt": initial_prompt,
            **{k: v for k, v in whisper_config.items() if k != "model"}
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
    CRITICAL FIX: Enhanced audio preparation with noise reduction and normalization
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
        
        # Resample if necessary
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            audio_data = audio_data.astype(np.float32)  # Ensure float32 after resampling
        
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
        
        # Basic noise reduction (high-pass filter)
        from scipy import signal
        sos = signal.butter(4, 80, btype='highpass', fs=target_sr, output='sos')
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