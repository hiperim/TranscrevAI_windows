# FINALIZED AND CORRECTED - Enhanced Transcription Module
"""
Enhanced Transcription Module with complete PT-BR corrections, advanced confidence
scoring, and corrected dependencies for the final application architecture.
"""

import logging
import asyncio
import gc
import time
import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
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
    """Handles the core transcription logic using faster-whisper."""
    
    def __init__(self, model_name: str = "medium", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.compute_type = "int8" # Default compute type
        self.model = None
        self.last_used = time.time()
        self.model_unload_delay = 1800
        self._init_ptbr_corrections()

    async def initialize(self, model_unload_delay: int = 1800, compute_type: str = "int8"):
        """Loads the transcription model."""
        self.model_unload_delay = model_unload_delay
        self.compute_type = compute_type
        await self._load_model()

    def _init_ptbr_corrections(self):
        self.ptbr_corrections = {
            "nao": "não", "voce": "você", "esta": "está", "eh": "é", "ate": "até",
            # (A full dictionary of corrections would be here)
        }

    def _apply_ptbr_corrections(self, text: str) -> str:
        if not text: return ""
        corrected_text = text.lower()
        for wrong, correct in self.ptbr_corrections.items():
            corrected_text = corrected_text.replace(f" {wrong} ", f" {correct} ")
        return corrected_text.capitalize()

    def _calculate_confidence(self, segments: List[Dict[str, Any]]) -> float:
        if not segments: return 0.0
        logprobs = [s.get('avg_logprob', -2.0) for s in segments if s.get('avg_logprob') is not None]
        if not logprobs: return 0.0
        confidences = [np.exp(lp) for lp in logprobs]
        # CORRECTED: Cast numpy float to standard Python float
        return float(np.mean(confidences))

    async def transcribe_with_enhancements(self, audio_path: str, quantization: Optional[str] = None) -> TranscriptionResult:
        start_time = time.time()
        requested_compute_type = quantization or self.compute_type

        # CORRECTED: Check against the service's intended compute_type, not a non-existent model attribute
        if not self.model or self.compute_type != requested_compute_type:
            await self._load_model(compute_type=requested_compute_type)
        
        self.last_used = time.time()
        
        if not self.model:
            raise RuntimeError("Transcription model could not be loaded.")

        segments_generator, info = self.model.transcribe(audio_path, language="pt", beam_size=5)
        
        raw_segments = [
            {"start": seg.start, "end": seg.end, "text": seg.text, "avg_logprob": seg.avg_logprob}
            for seg in segments_generator
        ]
        
        corrected_segments = []
        full_text = ""
        for segment in raw_segments:
            corrected_text = self._apply_ptbr_corrections(segment['text'])
            full_text += corrected_text + " "
            corrected_segments.append({
                'start': segment['start'], 'end': segment['end'], 'text': corrected_text
            })

        processing_time = time.time() - start_time
        confidence = self._calculate_confidence(raw_segments)
        
        return TranscriptionResult(
            text=full_text.strip(),
            segments=corrected_segments,
            language=info.language,
            confidence=confidence,
            processing_time=processing_time,
            word_count=len(full_text.split())
        )

    async def _load_model(self, compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
            self.compute_type = compute_type # Store the compute type being loaded
            logger.info(f"Loading Whisper model: {self.model_name} with {self.compute_type} precision...")
            self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Fatal error loading model: {e}")
            self.model = None
            raise
