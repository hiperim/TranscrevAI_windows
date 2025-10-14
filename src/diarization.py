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
            logger.info("pyannote.audio pipeline loaded with custom clustering threshold=0.35")

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
        if not self.pipeline:
            raise RuntimeError("Diarization pipeline is not initialized.")

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

            # Use the new, more accurate word-level alignment function
            aligned_segments = align_speakers_by_word(transcription_segments, diarization_result)

            return {"segments": aligned_segments, "num_speakers": int(num_speakers)}

        except Exception as e:
            logger.error(f"Diarization with pyannote.audio failed: {e}", exc_info=True)
            for seg in transcription_segments:
                seg['speaker'] = 'SPEAKER_01'
            return {"segments": transcription_segments, "num_speakers": 1}

# --- New, Corrected and Robust Alignment Function ---
def align_speakers_by_word(transcription_segments: List[Dict[str, Any]], diarization_result) -> List[Dict[str, Any]]:
    """
    Aligns speaker labels to transcription segments using word-level timestamps for high accuracy.
    This version is robust against words falling into non-speech gaps and is type-safe.
    """
    if not diarization_result:
        # If diarization failed entirely, assign a default speaker to all segments
        for segment in transcription_segments:
            segment['speaker'] = 'SPEAKER_01'
        return transcription_segments

    for segment in transcription_segments:
        if 'words' not in segment or not segment['words']:
            # Fallback for segments without word-level timestamps
            try:
                cropped_annotation = diarization_result.crop(segment['start'], segment['end'])
                if cropped_annotation and not cropped_annotation.is_empty():
                    segment['speaker'] = cropped_annotation.argmax()
                else:
                    segment['speaker'] = 'SPEAKER_XX' # No speaker in this segment
            except Exception:
                segment['speaker'] = 'SPEAKER_XX'
            continue

        word_speaker_counts: Dict[str, int] = {}
        for word in segment['words']:
            # Crop the diarization to the word's timeframe
            cropped_annotation = diarization_result.crop(word['start'], word['end'])
            
            # Check if the cropped annotation is not empty (i.e., a speaker was active)
            if cropped_annotation and not cropped_annotation.is_empty():
                try:
                    speaker = cropped_annotation.argmax()
                    word['speaker'] = speaker
                    word_speaker_counts[speaker] = word_speaker_counts.get(speaker, 0) + 1
                except (IndexError, ValueError):
                    word['speaker'] = 'SPEAKER_XX' # Handle cases where argmax might fail on unusual annotations
            else:
                # This word falls in a gap where no speaker was detected
                word['speaker'] = 'SPEAKER_XX'
        
        # Assign the dominant speaker for the entire segment based on word counts
        if word_speaker_counts:
            # This is now guaranteed to be safe and satisfies Pylance
            dominant_speaker = max(word_speaker_counts, key=lambda spk: word_speaker_counts[spk])
            segment['speaker'] = dominant_speaker
        else:
            # If no words in the segment had a speaker, assign a default
            segment['speaker'] = 'SPEAKER_XX'

    return transcription_segments
