import asyncio
import logging
import threading
import gc
import time
import re
import functools
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from src.exceptions import TranscriptionError, AudioProcessingError
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Standardized transcription result data structure"""
    text: str
    segments: List[Dict[str, Any]]
    language: str
    confidence: float
    processing_time: float
    word_count: int

class TranscriptionService:
    def __init__(self, model_name: str = "pierreguillou/whisper-medium-portuguese", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.compute_type = "int8"
        self.model = None
        self.last_used = time.time()
        self.model_unload_delay = 600  # 10 minutes
        self._model_lock = threading.Lock()  # Thread-safety for model operations
        self.model_loads_count = 0  # Monitoring metric
        self.model_unloads_count = 0  # Monitoring metric
        self._init_ptbr_corrections()

    async def initialize(self, model_unload_delay: int = 600, compute_type: str = "int8"):
        self.model_unload_delay = model_unload_delay
        self.compute_type = compute_type
        await self._load_model(compute_type=compute_type)

    def _init_ptbr_corrections(self):
        self.ptbr_corrections = {
            # accent corrections (safe, unambiguous)
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

            # PT-BR colloquial elisions
            "para": "pra",       
            "para o": "pro",     
            "para a": "pra",     
            "para os": "pros",   
            "para as": "pras",   
            "ta": "está",
            "tava": "estava",
            "tao": "tão",
            "ce": "você",
            "ceis": "vocês",
            "ne": "né",
            "num": "não"
        }

        # Pre-compile regex patterns at init
        self.correction_patterns = [
            (re.compile(rf'\b{re.escape(wrong)}\b', re.IGNORECASE), correct)
            for wrong, correct in self.ptbr_corrections.items()
        ]
        logger.info(f"✅ Pre-compiled {len(self.correction_patterns)} regex patterns for PT-BR corrections")

    def _apply_ptbr_corrections(self, text: str) -> str:
        """
        PT-BR corrections while preserving proper capitalization
        """
        if not text: return ""

        words = text.split()
        corrected_words = []

        for i, word in enumerate(words):
            # Remove punctuation for matching
            word_clean = word.strip('.,;:!?').lower()

            # Apply corrections
            corrected = word_clean
            for pattern, replacement in self.correction_patterns:
                if pattern.search(corrected):
                    corrected = pattern.sub(replacement, corrected)
                    break

            # Capitalization logic
            if i == 0:
                corrected = corrected.capitalize()
            elif i > 0 and any(corrected_words[i-1].endswith(p) for p in '.!?'):
                corrected = corrected.capitalize()
            elif word[0].isupper() and i > 0 and not corrected_words[i-1].endswith(('.', '!', '?')):
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
        # numpy float to standard python float for .json compatibility
        return float(np.mean(confidences))

    def _transcribe_sync(
        self,
        audio_path: str,
        transcribe_args: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Any, str, float]:
        """Sync transcription - runs in executor thread"""
        if not self.model:
            raise RuntimeError("Model not loaded")

        # Blocking operation
        segments_generator, info = self.model.transcribe(audio_path, **transcribe_args)

        raw_segments = []
        full_text = ""
        for seg in segments_generator:
            segment_dict = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_logprob": seg.avg_logprob
            }
            if transcribe_args.get("word_timestamps") and hasattr(seg, 'words') and seg.words:
                segment_dict['words'] = [
                    {'word': w.word, 'start': w.start, 'end': w.end, 'probability': w.probability}
                    for w in seg.words
                ]
            raw_segments.append(segment_dict)
            corrected_text = self._apply_ptbr_corrections(seg.text)
            full_text += corrected_text + " "

        confidence = self._calculate_confidence(raw_segments)
        return raw_segments, info, full_text, confidence

    async def transcribe_with_enhancements(
        self,
        audio_path: str,
        quantization: Optional[str] = None,
        word_timestamps: bool = False,
        whisper_params: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """Async transcription with executor"""
        start_time = time.time()
        requested_compute_type = quantization or self.compute_type

        # Check and load model outside the transcription lock if necessary
        if not self.model or self.compute_type != requested_compute_type:
            await self._load_model(compute_type=requested_compute_type)

        # Lock for setup only
        with self._model_lock:
            self.last_used = time.time()
            if not self.model:
                raise TranscriptionError("Model not loaded")

            # Prepare args
            transcribe_args = {
                "language": "pt",
                "beam_size": 3,
                "best_of": 3,
                "word_timestamps": word_timestamps,
                "vad_filter": False
            }

            # Execute in separate thread
            loop = asyncio.get_running_loop()
            try:
                sync_func = functools.partial(
                    self._transcribe_sync,
                    audio_path=audio_path,
                    transcribe_args=transcribe_args
                )
                raw_segments, info, full_text, confidence = await loop.run_in_executor(
                    None, sync_func
                )
            except FileNotFoundError as e:
                raise AudioProcessingError("Audio not found", context={"audio_path": audio_path}) from e
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                raise TranscriptionError(f"Failed: {e}", context={"audio_path": audio_path}) from e

        processing_time = time.time() - start_time
        return TranscriptionResult(
            text=full_text.strip(),
            segments=raw_segments,
            language=info.language,
            confidence=confidence,
            processing_time=processing_time,
            word_count=len(full_text.strip().split())
        )

    async def _load_model(self, compute_type: str = "int8"):
        """Load model asynchronously with retry logic"""
        loop = asyncio.get_running_loop()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from faster_whisper import WhisperModel
                self.compute_type = compute_type
                logger.info(f"Loading Whisper model (attempt {attempt+1}/{max_retries}): {self.model_name} with {self.compute_type} precision...")
                load_start = time.time()

                # Create partial function with keyword args to pass to executor
                loader_func = functools.partial(
                    WhisperModel, 
                    model_size_or_path=self.model_name, 
                    device=self.device, 
                    compute_type=self.compute_type
                )
                self.model = await loop.run_in_executor(None, loader_func)

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
                await asyncio.sleep(2 ** attempt)

    def unload_model(self):
        """Unload model from memory with thread-safety"""
        with self._model_lock:  # Prevent race conditions during unload
            if self.model is not None:
                logger.info("Unloading Whisper model due to inactivity...")
                self.model = None
                self.model_unloads_count += 1
                gc.collect()
                logger.info(f"Model unloaded successfully (total unloads: {self.model_unloads_count})")

    def should_unload(self) -> bool:
        """Check if model should be unloaded based on inactivity period"""
        return (self.model is not None and
                time.time() - self.last_used > self.model_unload_delay)
