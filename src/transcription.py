# FINALIZED AND CORRECTED - Enhanced Transcription Module with Model Unloading
"""
Enhanced Transcription Module with complete PT-BR corrections, advanced confidence
scoring, automatic model unloading for memory optimization, and production-ready
thread-safety mechanisms.
"""

import asyncio
import logging
import threading
import gc
import time
import re
import unicodedata
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from src.exceptions import TranscriptionError, AudioProcessingError
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Standardized transcription result data structure."""
    text: str
    segments: List[Dict[str, Any]]
    language: str
    confidence: float
    processing_time: float
    word_count: int

class TranscriptionService:
    """Handles the core transcription logic using faster-whisper with automatic model unloading."""

    def __init__(self, model_name: str = "medium", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.compute_type = "int8" # Default compute type
        self.model = None
        self.last_used = time.time()
        self.model_unload_delay = 600  # 10 minutes (optimized from 30min for web usage)
        self._model_lock = threading.Lock()  # Thread-safety for model operations
        self.model_loads_count = 0  # Monitoring metric
        self.model_unloads_count = 0  # Monitoring metric
        self._init_ptbr_corrections()

    async def initialize(self, model_unload_delay: int = 600, compute_type: str = "int8"):
        """Loads the transcription model."""
        self.model_unload_delay = model_unload_delay
        self.compute_type = compute_type
        await self._load_model(compute_type=compute_type)

    def _init_ptbr_corrections(self):
        # OPTIMIZED VERSION: 25 essential PT-BR corrections + improved capitalization
        # Based on: 20251015_expanded_ptbr_rules_25_final.md
        # Result: 86% accuracy with generic capitalization (Oct 23, 2025)
        self.ptbr_corrections = {
            # Original 12 accent corrections (safe, unambiguous)
            "nao": "não",
            "voce": "você",
            "esta": "está",
            "ja": "já",
            "la": "lá",
            "tambem": "também",
            "so": "só",
            "entao": "então",
            "porem": "porém",
            "alem": "além",
            "ate": "até",
            "sao": "são",

            # 11 PT-BR colloquial elisões (safe, Level 1)
            # EXPERIMENTAL: REVERSED normalization (formal→colloquial) based on CORAA analysis
            "para": "pra",       # 32 CORAA occurrences - most common
            "para o": "pro",     # Common colloquial contraction
            "para a": "pra",     # Common colloquial contraction
            "para os": "pros",   # Common colloquial contraction
            "para as": "pras",   # Common colloquial contraction
            "ta": "está",
            "tava": "estava",
            "tao": "tão",
            "ce": "você",
            "ceis": "vocês",
            "ne": "né",
            "num": "não"
        }

        # ⭐ PRÉ-COMPILAR REGEX PATTERNS (CRITICAL OPTIMIZATION)
        # Compiling once at initialization = 5-10x faster than compiling in loop
        # Source: final_optimization_summary.md - "restored 1.64x target"
        self.correction_patterns = [
            (re.compile(rf'\b{re.escape(wrong)}\b', re.IGNORECASE), correct)
            for wrong, correct in self.ptbr_corrections.items()
        ]
        logger.info(f"✅ Pre-compiled {len(self.correction_patterns)} regex patterns for PT-BR corrections")

    def _apply_ptbr_corrections(self, text: str) -> str:
        """
        IMPROVED: Applies PT-BR corrections while preserving proper capitalization.
        - Capitalizes sentence starts (after . ! ?)
        - Preserves proper nouns (mid-sentence capitals from Whisper)
        - Generic logic that works for ANY audio file (no hard-coded words)
        """
        if not text: return ""

        words = text.split()
        corrected_words = []

        for i, word in enumerate(words):
            # Remove punctuation for matching
            word_clean = word.strip('.,;:!?').lower()

            # Apply corrections using PRE-COMPILED PATTERNS (5-10x faster)
            corrected = word_clean
            for pattern, replacement in self.correction_patterns:
                if pattern.search(corrected):
                    corrected = pattern.sub(replacement, corrected)
                    break

            # Capitalize logic (GENERIC - no hard-coded words):
            # 1. First word always capitalize
            if i == 0:
                corrected = corrected.capitalize()
            # 2. After sentence-ending punctuation
            elif i > 0 and any(corrected_words[i-1].endswith(p) for p in '.!?'):
                corrected = corrected.capitalize()
            # 3. Preserve proper nouns (if Whisper capitalized mid-sentence)
            elif word[0].isupper() and i > 0 and not corrected_words[i-1].endswith(('.', '!', '?')):
                # Whisper capitalized mid-sentence = probably proper noun
                corrected = corrected.capitalize()

            # Re-add punctuation
            for punct in '.,;:!?':
                if word.endswith(punct):
                    corrected += punct
                    break

            corrected_words.append(corrected)

        return ' '.join(corrected_words)

    def _calculate_confidence(self, segments: List[Dict[str, Any]]) -> float:
        if not segments: return 0.0
        logprobs = [s.get('avg_logprob', -2.0) for s in segments if s.get('avg_logprob') is not None]
        if not logprobs: return 0.0
        confidences = [np.exp(lp) for lp in logprobs]
        # CORRECTED: Cast numpy float to standard Python float
        return float(np.mean(confidences))

    async def transcribe_with_enhancements(
        self,
        audio_path: str,
        quantization: Optional[str] = None,
        word_timestamps: bool = False,
        whisper_params: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        with self._model_lock:  # CRITICAL: Thread-safety for concurrent requests
            start_time = time.time()
            requested_compute_type = quantization or self.compute_type

            if not self.model or self.compute_type != requested_compute_type:
                await self._load_model(compute_type=requested_compute_type)

            self.last_used = time.time()

            if not self.model:
                raise RuntimeError("Transcription model could not be loaded.")

            # Default VAD parameters (can be overridden)
            vad_parameters = dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=2000
            )

            # Build transcription parameters with defaults
            transcribe_args = {
                "language": "pt",
                "beam_size": 5,
                "best_of": 5,
                "word_timestamps": word_timestamps,
                "vad_filter": True,
                "vad_parameters": vad_parameters
            }

            # Apply custom Whisper parameters if provided
            if whisper_params:
                # Override VAD parameters if provided
                if "vad_parameters" in whisper_params:
                    vad_parameters.update(whisper_params["vad_parameters"])
                    transcribe_args["vad_parameters"] = vad_parameters

                # Apply other Whisper parameters
                for key, value in whisper_params.items():
                    if key != "vad_parameters":  # VAD already handled above
                        transcribe_args[key] = value

            try:
                segments_generator, info = self.model.transcribe(
                    audio_path,
                    **transcribe_args
                )

                raw_segments = []
                full_text = ""
                for seg in segments_generator:
                    segment_dict = {
                        "start": seg.start, 
                        "end": seg.end, 
                        "text": seg.text, 
                        "avg_logprob": seg.avg_logprob
                    }
                    if word_timestamps and hasattr(seg, 'words') and seg.words is not None:
                        segment_dict['words'] = [
                            {'word': w.word, 'start': w.start, 'end': w.end, 'probability': w.probability}
                            for w in seg.words
                        ]
                    
                    raw_segments.append(segment_dict)
                    corrected_text = self._apply_ptbr_corrections(seg.text)
                    full_text += corrected_text + " "

            except FileNotFoundError as e:
                # Audio file doesn't exist
                raise AudioProcessingError(
                    "Audio file not found for transcription",
                    context={"audio_path": audio_path}
                ) from e
            except MemoryError as e:
                # Insufficient RAM for model
                raise TranscriptionError(
                    "Insufficient memory for transcription",
                    context={"audio_path": audio_path}
                ) from e
            except Exception as e:
                # Unexpected error during transcription
                logger.error(
                    "Unexpected transcription error",
                    extra={
                        "audio_path": audio_path,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
                raise TranscriptionError(
                    f"Transcription failed: {str(e)}",
                    context={"audio_path": audio_path}
                ) from e

            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(raw_segments)

            return TranscriptionResult(
                text=full_text.strip(),
                segments=raw_segments, # Now contains word timestamps if requested
                language=info.language,
                confidence=confidence,
                processing_time=processing_time,
                word_count=len(full_text.split())
            )

    async def _load_model(self, compute_type: str = "int8"):
        """Load model with retry logic for production reliability."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from faster_whisper import WhisperModel
                self.compute_type = compute_type
                logger.info(f"Loading Whisper model (attempt {attempt+1}/{max_retries}): {self.model_name} with {self.compute_type} precision...")
                load_start = time.time()

                self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

                load_time = time.time() - load_start
                self.model_loads_count += 1
                logger.info(f"Model loaded successfully in {load_time:.2f}s (total loads: {self.model_loads_count})")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.critical(f"Model reload failed after {max_retries} attempts: {e}")
                    self.model = None
                    raise
                logger.warning(f"Model load attempt {attempt+1} failed, retrying in {2**attempt}s...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def unload_model(self):
        """Unload model from memory with thread-safety to free ~1.5GB RAM."""
        with self._model_lock:  # CRITICAL: Prevent race conditions during unload
            if self.model is not None:
                logger.info("Unloading Whisper model due to inactivity...")
                self.model = None
                self.model_unloads_count += 1
                gc.collect()
                logger.info(f"Model unloaded successfully (total unloads: {self.model_unloads_count})")

    def should_unload(self) -> bool:
        """Check if model should be unloaded based on inactivity period."""
        return (self.model is not None and
                time.time() - self.last_used > self.model_unload_delay)
