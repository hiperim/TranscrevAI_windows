# diarization.py - Implementation with pyannote.audio
"""
Speaker Diarization using the state-of-the-art pyannote.audio library.
This module replaces all previous custom VAD, embedding, and clustering logic.
"""

import logging
import os
import asyncio
from typing import Dict, Any, List, Optional

import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class PyannoteDiarizer:
    """Implements diarization using a pyannote.audio pipeline."""

    def __init__(self, device: str = "cpu"):
        logger.info("Initializing pyannote.audio pipeline...")
        self.device = device
        self.pipeline = None
        try:
            # Load .env file and get token
            load_dotenv()
            auth_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
            if auth_token is None:
                raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable not set. Please get a token from https://hf.co/settings/tokens")

            # Load the pyannote.audio pipeline
            logger.info("Loading pyannote/speaker-diarization-3.1 pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            ).to(torch.device(self.device))

            # Set custom hyperparameters to tune speaker detection
            # A lower threshold encourages the model to find more speakers.
            self.pipeline.instantiate({
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 15,
                    "threshold": 0.35  # ← CUSTOM: Reverted to 0.35 for best overall accuracy
                }
            })
            logger.info("pyannote.audio pipeline loaded with custom clustering threshold=0.4")

        except Exception as e:
            logger.error(f"Failed to load pyannote.audio pipeline: {e}", exc_info=True)
            self.pipeline = None

    async def diarize(self, audio_path: str, transcription_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Asynchronously performs speaker diarization using the pyannote.audio pipeline.
        The pipeline handles chunking and processing internally.
        """
        if not self.pipeline:
            logger.error("Diarization pipeline not available. Falling back to single speaker diarization.")
            for seg in transcription_segments:
                seg['speaker'] = 'SPEAKER_01'
            return {"segments": transcription_segments, "num_speakers": 1}

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._process_diarization_sync, audio_path, transcription_segments
            )
            return result
        except Exception as e:
            logger.error(f"Diarization failed in async wrapper: {e}", exc_info=True)
            for seg in transcription_segments:
                seg['speaker'] = 'SPEAKER_01'
            return {"segments": transcription_segments, "num_speakers": 1}

    def _process_diarization_sync(self, audio_path: str, transcription_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synchronous diarization logic.
        """
        logger.info(f"Starting pyannote.audio diarization for {audio_path}")
        try:
            # The pipeline object handles the entire process: VAD, embedding, clustering.
            # It processes the audio in chunks internally, so we can pass the whole file.
            diarization_result = self.pipeline(audio_path)

            # Convert the pyannote.core.Annotation object to our desired list format
            diarization_segments = []
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                diarization_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            
            num_speakers = len(diarization_result.labels())
            logger.info(f"pyannote.audio found {num_speakers} speakers.")

            # Use the existing alignment function to merge transcription and diarization
            aligned_segments = force_transcription_segmentation(transcription_segments, diarization_segments)

            return {"segments": aligned_segments, "num_speakers": int(num_speakers)}

        except Exception as e:
            logger.error(f"Diarization with pyannote.audio failed: {e}", exc_info=True)
            for seg in transcription_segments:
                seg['speaker'] = 'SPEAKER_01'
            return {"segments": transcription_segments, "num_speakers": 1}

# --- Critical Alignment Function (Preserved) ---
def force_transcription_segmentation(
    transcription_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Aligns transcription segments with speaker labels from diarization.
    """
    if not diarization_segments or not transcription_segments:
        # Return original transcription if either list is empty
        for seg in transcription_segments:
            if 'speaker' not in seg:
                seg['speaker'] = 'SPEAKER_01'
        return transcription_segments

    # Create a timeline mapping tenths of a second to a speaker
    time_to_speaker = {}
    for seg in diarization_segments:
        start = int(seg.get('start', 0) * 10)
        end = int(seg.get('end', 0) * 10)
        speaker = seg.get('speaker', 'Unknown')
        for i in range(start, end):
            time_to_speaker[i] = speaker

    # Assign a speaker to each transcription segment based on temporal overlap
    for trans_seg in transcription_segments:
        trans_start = int(trans_seg.get('start', 0) * 10)
        trans_end = int(trans_seg.get('end', 0) * 10)
        
        speaker_counts = {}
        for i in range(trans_start, trans_end):
            if i in time_to_speaker:
                speaker = time_to_speaker[i]
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        if speaker_counts:
            # Assign the speaker with the most overlap
            dominant_speaker = max(speaker_counts, key=speaker_counts.get)
            trans_seg['speaker'] = dominant_speaker
        else:
            # If no overlap, find the nearest speaker segment
            if time_to_speaker:
                closest_time = min(time_to_speaker.keys(), key=lambda x: abs(x - trans_start))
                trans_seg['speaker'] = time_to_speaker[closest_time]
            else:
                # Fallback if no diarization segments exist at all
                trans_seg['speaker'] = 'SPEAKER_01'

    return transcription_segments
