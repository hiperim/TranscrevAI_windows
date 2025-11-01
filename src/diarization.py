"""
Speaker Diarization using the state-of-the-art pyannote.audio library.

This module implements a robust, 100% offline speaker diarization solution by
leveraging a project-local Hugging Face cache.
"""

import os
from pathlib import Path
import asyncio
import logging
from typing import Dict, Any, List, Optional

# CRITICAL: Set HF_HOME *before* importing pyannote.audio or torch.
# This forces the library to use our local, embedded cache directory.
MODELS_CACHE_DIR = Path(__file__).parent.parent / "models" / ".cache"
os.environ['HF_HOME'] = str(MODELS_CACHE_DIR)

import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class PyannoteDiarizer:
    """Implements diarization using a cached pyannote.audio pipeline."""

    def __init__(self, device: str = "cpu", embedding_batch_size: int = 8):
        logger.info(f"Initializing pyannote.audio pipeline from local cache: {MODELS_CACHE_DIR}")
        self.device = device
        self.embedding_batch_size = embedding_batch_size
        self.pipeline: Optional[Pipeline] = None

        try:
            # 1. Load pipeline from pretrained repo ID.
            #    pyannote automatically uses the cache defined by HF_HOME.
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            logger.info("✓ Pipeline loaded from cache.")

            # 2. Instantiate with custom hyperparameters for accuracy.
            #    The default threshold is too high and merges distinct speakers.
            self.pipeline.instantiate({
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 15,
                    "threshold": 0.35
                }
            })
            logger.info("✓ Pipeline instantiated with custom clustering threshold.")

            # 3. Move to device and set batch size
            self.pipeline.to(torch.device(self.device))
            self.pipeline.embedding_batch_size = self.embedding_batch_size

            logger.info("✓ Diarization pipeline initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}", exc_info=True)
            self.pipeline = None

    async def diarize(self, audio_path: str, transcription_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.pipeline:
            logger.error("Diarization pipeline not available. Falling back to single speaker diarization.")
            for seg in transcription_segments:
                seg['speaker'] = 'SPEAKER_01'
            return {"segments": transcription_segments, "num_speakers": 1}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            self._process_diarization_sync, 
            audio_path, 
            transcription_segments
        )

    def _process_diarization_sync(self, audio_path: str, transcription_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.pipeline:
            raise RuntimeError("Diarization pipeline is not initialized.")

        logger.info(f"Starting pyannote.audio diarization for: {audio_path}")
        try:
            diarization_result = self.pipeline(audio_path)
            num_speakers = len(diarization_result.labels())
            logger.info(f"✓ pyannote.audio detected {num_speakers} speakers")
            aligned_segments = align_speakers_by_word(transcription_segments, diarization_result)
            return {"segments": aligned_segments, "num_speakers": int(num_speakers)}
        except Exception as e:
            logger.error(f"Diarization with pyannote.audio failed: {e}", exc_info=True)
            for seg in transcription_segments:
                seg['speaker'] = 'SPEAKER_01'
            return {"segments": transcription_segments, "num_speakers": 1}

def align_speakers_by_word(transcription_segments: List[Dict[str, Any]], diarization_result) -> List[Dict[str, Any]]:
    logger.info("Aligning speakers to transcription segments...")
    if not diarization_result:
        logger.warning("Diarization result is None/empty, assigning SPEAKER_01 to all")
        for segment in transcription_segments:
            segment['speaker'] = 'SPEAKER_01'
        return transcription_segments

    diarization_segments = []
    speakers_found = set()
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        diarization_segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
        speakers_found.add(speaker)

    logger.info(f"Diarization found {len(speakers_found)} unique speakers: {sorted(speakers_found)}")

    sorted_speakers = sorted(speakers_found)
    speaker_mapping = {old: f"SPEAKER_{str(i+1).zfill(2)}" for i, old in enumerate(sorted_speakers)}
    logger.debug(f"Speaker mapping: {speaker_mapping}")

    def find_speaker_at_timestamp(timestamp: float, margin: float = 0.0) -> Optional[str]:
        overlapping = []
        for seg in diarization_segments:
            if seg['start'] <= timestamp + margin and seg['end'] >= timestamp - margin:
                overlap_start = max(seg['start'], timestamp - margin)
                overlap_end = min(seg['end'], timestamp + margin)
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > 0:
                    overlapping.append((seg['speaker'], overlap_duration))
        if overlapping:
            return max(overlapping, key=lambda x: x[1])[0]
        return None

    for segment in transcription_segments:
        if 'words' not in segment or not segment['words']:
            mid_timestamp = (segment['start'] + segment['end']) / 2
            speaker = find_speaker_at_timestamp(mid_timestamp, margin=0.1)
            if speaker:
                segment['speaker'] = speaker_mapping.get(speaker, speaker)
            else:
                segment['speaker'] = 'SPEAKER_XX'
            continue

        word_speaker_counts: Dict[str, int] = {}
        for word in segment['words']:
            word_mid = (word['start'] + word['end']) / 2
            speaker = find_speaker_at_timestamp(word_mid, margin=0.05)
            if speaker:
                word['speaker'] = speaker_mapping.get(speaker, speaker)
                word_speaker_counts[speaker] = word_speaker_counts.get(speaker, 0) + 1
            else:
                word['speaker'] = 'SPEAKER_XX'

        if word_speaker_counts:
            dominant_speaker_original = max(word_speaker_counts, key=lambda spk: word_speaker_counts[spk])
            segment['speaker'] = speaker_mapping.get(dominant_speaker_original, dominant_speaker_original)
        else:
            segment['speaker'] = 'SPEAKER_XX'

    logger.info("✓ Speaker alignment completed successfully")
    return transcription_segments