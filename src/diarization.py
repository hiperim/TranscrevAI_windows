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

    def __init__(self, device: str = "cpu", embedding_batch_size: int = 8):
        logger.info("Initializing pyannote.audio pipeline...")
        self.device = device
        self.embedding_batch_size = embedding_batch_size
        self.pipeline = None
        try:
            import yaml
            from huggingface_hub import hf_hub_download
            from pyannote.audio.pipelines import SpeakerDiarization

            logger.info("Starting 100% offline pipeline instantiation with runtime config...")

            # Step 1: Get path to main pipeline config, forcing local-only file resolution
            main_config_path = hf_hub_download(
                repo_id="pyannote/speaker-diarization-3.1",
                filename="config.yaml",
                local_files_only=True
            )
            logger.info(f"Main config found locally at: {main_config_path}")

            # Step 2: Load config into memory
            with open(main_config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Step 3: Get paths to dependency models from local cache only
            logger.info("Resolving dependency model paths from local cache...")
            
            # The previous attempt failed because it was assumed the segmentation model's config was needed.
            # The SpeakerDiarization pipeline actually expects the *model file itself* for its segmentation parameter.
            segmentation_model_path = hf_hub_download(
                repo_id="pyannote/segmentation-3.0",
                filename="pytorch_model.bin",
                local_files_only=True
            )
            logger.info(f"Segmentation model found locally at: {segmentation_model_path}")

            embedding_model_path = hf_hub_download(
                repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
                filename="pytorch_model.bin",
                local_files_only=True
            )
            logger.info(f"Embedding model found locally at: {embedding_model_path}")

            # Step 4: Modify config dictionary in memory to point to local model files
            pipeline_params = config.get('pipeline', {}).get('params', {})
            if not pipeline_params:
                raise ValueError("Could not find 'pipeline.params' in the loaded config.")

            pipeline_params['segmentation'] = segmentation_model_path
            pipeline_params['embedding'] = embedding_model_path

            # Step 5: Instantiate pipeline with modified in-memory config
            self.pipeline = SpeakerDiarization(**pipeline_params)
            self.pipeline.to(torch.device(self.device))
            logger.info("✓ Pipeline instantiated from in-memory config with local paths.")

            # Set custom hyperparameters
            self.pipeline.instantiate({
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 15,
                    "threshold": 0.35
                }
            })

            # Set batch size for embeddings
            self.pipeline.embedding_batch_size = self.embedding_batch_size
            logger.info(f"pyannote.audio pipeline configured with clustering threshold=0.35, embedding_batch_size={self.embedding_batch_size}")

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
    Uses direct segment lookup instead of .crop() which has known issues.
    Renumbers speakers starting from SPEAKER_01 (not SPEAKER_00).
    """
    import logging
    logger = logging.getLogger(__name__)

    if not diarization_result:
        # If diarization failed entirely, assign a default speaker to all segments
        logger.warning("Diarization result is None/empty, assigning default SPEAKER_01 to all segments")
        for segment in transcription_segments:
            segment['speaker'] = 'SPEAKER_01'
        return transcription_segments

    # Convert diarization result to simple list of segments for direct lookup
    diarization_segments = []
    speakers_found = set()
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        diarization_segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
        speakers_found.add(speaker)

    logger.info(f"Diarization found {len(speakers_found)} speakers: {sorted(speakers_found)}")

    # Create speaker mapping: SPEAKER_00 -> SPEAKER_01, SPEAKER_01 -> SPEAKER_02, etc.
    sorted_speakers = sorted(speakers_found)
    speaker_mapping = {old: f"SPEAKER_{str(i+1).zfill(2)}" for i, old in enumerate(sorted_speakers)}
    logger.info(f"Speaker mapping: {speaker_mapping}")

    def find_speaker_at_timestamp(timestamp: float, margin: float = 0.0) -> Optional[str]:
        """
        Find the speaker at a given timestamp using direct segment lookup.
        Returns the speaker with the most overlap with the timestamp range.
        """
        overlapping = []
        for seg in diarization_segments:
            if seg['start'] <= timestamp + margin and seg['end'] >= timestamp - margin:
                # Calculate overlap duration
                overlap_start = max(seg['start'], timestamp - margin)
                overlap_end = min(seg['end'], timestamp + margin)
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > 0:
                    overlapping.append((seg['speaker'], overlap_duration))

        if overlapping:
            # Return speaker with longest overlap
            return max(overlapping, key=lambda x: x[1])[0]
        return None

    # Process each transcription segment
    for segment in transcription_segments:
        if 'words' not in segment or not segment['words']:
            # Fallback for segments without word-level timestamps
            # Use middle of segment timestamp
            mid_timestamp = (segment['start'] + segment['end']) / 2
            speaker = find_speaker_at_timestamp(mid_timestamp, margin=0.1)
            if speaker:
                segment['speaker'] = speaker_mapping.get(speaker, speaker)
            else:
                segment['speaker'] = 'SPEAKER_XX'
            continue

        word_speaker_counts: Dict[str, int] = {}
        for word in segment['words']:
            # Use middle of word timestamp for lookup
            word_mid = (word['start'] + word['end']) / 2
            speaker = find_speaker_at_timestamp(word_mid, margin=0.05)

            if speaker:
                word['speaker'] = speaker_mapping.get(speaker, speaker)
                word_speaker_counts[speaker] = word_speaker_counts.get(speaker, 0) + 1
            else:
                word['speaker'] = 'SPEAKER_XX'

        # Assign the dominant speaker for the entire segment based on word counts
        if word_speaker_counts:
            dominant_speaker_original = max(word_speaker_counts, key=lambda spk: word_speaker_counts[spk])
            segment['speaker'] = speaker_mapping.get(dominant_speaker_original, dominant_speaker_original)
        else:
            # If no words in the segment had a speaker, assign a default
            segment['speaker'] = 'SPEAKER_XX'

    return transcription_segments
