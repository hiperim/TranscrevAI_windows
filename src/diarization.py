"""
Project's Hugging Face cache for local diarization pipeline - run file "'projectRootFolder'/setup_certs_SSL_ModelsCache/download_models.py"
"""

import os
from pathlib import Path
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Set HF_HOME before importing pyannote.audio or torch - forces the library to use local embedded cache directory
MODELS_CACHE_DIR = Path(__file__).parent.parent / "models" / ".cache"
os.environ['HF_HOME'] = str(MODELS_CACHE_DIR)

import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import librosa
import numpy as np
from config.app_config import get_config

logger = logging.getLogger(__name__)
load_dotenv()
config = get_config()


class PyannoteDiarizer:
    def __init__(self, device: str = "cpu", embedding_batch_size: int = 8):
        logger.info(f"Initializing pyannote.audio pipeline from local cache: {MODELS_CACHE_DIR}")
        self.device = device
        self.embedding_batch_size = embedding_batch_size
        self.pipeline: Optional[Pipeline] = None

        try:
            # Load pipeline from pretrained repo ID - pyannote automatically uses the cache defined by HF_HOME
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            logger.info("Pipeline loaded from cache.")

            # Instantiate with custom hyperparameters from config
            instantiate_params = {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": config.diarization_min_cluster_size,
                    "threshold": config.diarization_threshold
                }
            }

            # Add min/max speakers if specified in config
            if config.diarization_min_speakers is not None:
                instantiate_params["clustering"]["min_clusters"] = config.diarization_min_speakers
            if config.diarization_max_speakers is not None:
                instantiate_params["clustering"]["max_clusters"] = config.diarization_max_speakers

            self.pipeline.instantiate(instantiate_params)
            logger.info(f"Pipeline instantiated with threshold={config.diarization_threshold}, min_cluster_size={config.diarization_min_cluster_size}")

            # Move to device and set batch size
            self.pipeline.to(torch.device(self.device))
            self.pipeline.embedding_batch_size = self.embedding_batch_size

            logger.info("Diarization pipeline initialized successfully.")

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
            logger.info(f"pyannote.audio detected {num_speakers} speakers")

            aligned_segments = align_speakers_by_word(transcription_segments, diarization_result, audio_path)
            result = {"segments": aligned_segments, "num_speakers": int(num_speakers)}
            return result
        except Exception as e:
            logger.error(f"Diarization with pyannote.audio failed: {e}", exc_info=True)
            for seg in transcription_segments:
                seg['speaker'] = 'SPEAKER_01'
            return {"segments": transcription_segments, "num_speakers": 1}

def extract_pitch(audio_path: str, start_time: float, end_time: float) -> Optional[float]:
    """Extract mean F0 (pitch) from audio segment"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time-start_time)
        if len(y) == 0:
            return None
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0_valid = f0[f0 > 0]
        return float(np.median(f0_valid)) if len(f0_valid) > 0 else None
    except Exception as e:
        logger.debug(f"Pitch extraction failed: {e}")
        return None

def align_speakers_by_word(transcription_segments: List[Dict[str, Any]], diarization_result, audio_path: Optional[str] = None) -> List[Dict[str, Any]]:
    logger.info("Aligning speakers to transcription segments...")
    if not diarization_result:
        logger.warning("Diarization result is None/empty, assigning SPEAKER_01 to all")
        for segment in transcription_segments:
            segment['speaker'] = 'SPEAKER_01'
        return transcription_segments

    diarization_segments = []
    speakers_found = set()
    speaker_first_appearance = {}  # Track first timestamp for each speaker

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        diarization_segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
        speakers_found.add(speaker)

        # Track first appearance time
        if speaker not in speaker_first_appearance:
            speaker_first_appearance[speaker] = turn.start

    logger.info(f"Diarization found {len(speakers_found)} unique speakers: {sorted(speakers_found)}")

    # Sort speakers by first appearance time (chronological order)
    sorted_speakers = sorted(speakers_found, key=lambda spk: speaker_first_appearance[spk])
    speaker_mapping = {old: f"SPEAKER_{str(i+1).zfill(2)}" for i, old in enumerate(sorted_speakers)}
    logger.info(f"Speaker mapping (chronological order): {speaker_mapping}")

    def find_speaker_at_timestamp(timestamp: float, margin: float = 0.0) -> Optional[str]:
        """Find speaker at timestamp, prioritizing closest segment center when overlapping"""
        candidates = []
        for seg in diarization_segments:
            if seg['start'] <= timestamp + margin and seg['end'] >= timestamp - margin:
                # Calculate distance from timestamp to segment center
                seg_center = (seg['start'] + seg['end']) / 2
                distance_to_center = abs(timestamp - seg_center)
                candidates.append((seg['speaker'], distance_to_center))

        if candidates:
            # Return speaker with closest center (smallest distance)
            return min(candidates, key=lambda x: x[1])[0]
        return None

    for segment in transcription_segments:
        if 'words' not in segment or not segment['words']:
            # Fallback: use segment midpoint if no word timestamps
            mid_timestamp = (segment['start'] + segment['end']) / 2
            speaker = find_speaker_at_timestamp(mid_timestamp, margin=1)
            if speaker:
                segment['speaker'] = speaker_mapping.get(speaker, speaker)
            else:
                segment['speaker'] = 'SPEAKER_XX'
            continue

        # Use word-level timestamps for precise alignment
        # Margin increased to 0.3s to compensate for VAD temporal desynchronization
        word_speaker_counts: Dict[str, int] = {}
        for word in segment['words']:
            word_mid = (word['start'] + word['end']) / 2
            speaker = find_speaker_at_timestamp(word_mid, margin=1)

            # Debug logging for first segment only
            if segment == transcription_segments[0] and len(word_speaker_counts) < 3:
                logger.info(f"  Word '{word.get('word', '?')}' at {word_mid:.2f}s → speaker: {speaker}")

            if speaker:
                word['speaker'] = speaker_mapping.get(speaker, speaker)
                word_speaker_counts[speaker] = word_speaker_counts.get(speaker, 0) + 1
            else:
                word['speaker'] = 'SPEAKER_XX'

        logger.info(f"Segment {segment['start']:.2f}-{segment['end']:.2f}s votes: {word_speaker_counts}")

        if word_speaker_counts:
            dominant_speaker_original = max(word_speaker_counts, key=lambda spk: word_speaker_counts[spk])
            segment['speaker'] = speaker_mapping.get(dominant_speaker_original, dominant_speaker_original)
        else:
            segment['speaker'] = 'SPEAKER_XX'

    # Log which speakers were actually assigned to segments
    assigned_speakers_original = set()
    for seg in transcription_segments:
        mapped_speaker = seg.get('speaker', 'UNKNOWN')
        # Reverse lookup to get original speaker
        for orig, mapped in speaker_mapping.items():
            if mapped == mapped_speaker:
                assigned_speakers_original.add(orig)
                break

    assigned_speakers = set(seg.get('speaker', 'UNKNOWN') for seg in transcription_segments)
    logger.info(f"Speakers assigned to transcription segments: {sorted(assigned_speakers)}")

    # Find speakers detected by pyannote but not assigned to any transcription segment
    unassigned_speakers = speakers_found - assigned_speakers_original

    if unassigned_speakers and audio_path:
        logger.info(f"Found {len(unassigned_speakers)} unassigned speakers: {sorted(unassigned_speakers)}")
        logger.info("Attempting pitch-based re-attribution")

        # Extract pitch for assigned speakers
        speaker_pitches = {}
        for seg in transcription_segments:
            if seg.get('speaker') and seg['speaker'] != 'SPEAKER_XX':
                pitch = extract_pitch(audio_path, seg['start'], seg['end'])
                if pitch:
                    if seg['speaker'] not in speaker_pitches:
                        speaker_pitches[seg['speaker']] = []
                    speaker_pitches[seg['speaker']].append(pitch)

        # Calculate median pitch per speaker
        speaker_median_pitch = {spk: np.median(pitches) for spk, pitches in speaker_pitches.items() if len(pitches) > 0}
        logger.info(f"Speaker pitches: {speaker_median_pitch}")

        for speaker in sorted(unassigned_speakers):
            speaker_segments = [seg for seg in diarization_segments if seg['speaker'] == speaker]
            for dia_seg in speaker_segments:
                # Try pitch matching
                pitch = extract_pitch(audio_path, dia_seg['start'], dia_seg['end'])
                best_match = None

                if pitch and speaker_median_pitch:
                    min_diff = float('inf')
                    for spk, med_pitch in speaker_median_pitch.items():
                        diff = abs(pitch - med_pitch)
                        if diff < min_diff:
                            min_diff = diff
                            best_match = spk

                    # Only re-attribute if difference is reasonable (< 50 Hz)
                    if min_diff < 50:
                        logger.info(f"  Pitch match: {speaker_mapping.get(speaker, speaker)} → {best_match} (diff: {min_diff:.1f}Hz)")
                        # Find and update existing segment or add
                        for seg in transcription_segments:
                            if abs(seg['start'] - dia_seg['start']) < 0.5 and seg.get('speaker') == best_match:
                                logger.info(f"  Merged into existing {best_match} segment")
                                break
                        else:
                            synthetic_segment = {
                                'start': dia_seg['start'],
                                'end': dia_seg['end'],
                                'text': '[inaudível]',
                                'speaker': best_match,
                                'avg_logprob': -1.0,
                                'words': []
                            }
                            transcription_segments.append(synthetic_segment)
                        continue

                # Fallback: add as [inaudível] with original speaker
                synthetic_segment = {
                    'start': dia_seg['start'],
                    'end': dia_seg['end'],
                    'text': '[inaudível]',
                    'speaker': speaker_mapping.get(speaker, speaker),
                    'avg_logprob': -1.0,
                    'words': []
                }
                transcription_segments.append(synthetic_segment)
                logger.info(f"  Added [inaudível] for {speaker_mapping.get(speaker, speaker)} at {dia_seg['start']:.2f}-{dia_seg['end']:.2f}s")

        # Sort segments by start time
        transcription_segments.sort(key=lambda x: x['start'])

    elif unassigned_speakers:
        logger.info(f"Found {len(unassigned_speakers)} unassigned speakers but no audio_path for pitch tracking")
        for speaker in sorted(unassigned_speakers):
            speaker_segments = [seg for seg in diarization_segments if seg['speaker'] == speaker]
            for dia_seg in speaker_segments:
                synthetic_segment = {
                    'start': dia_seg['start'],
                    'end': dia_seg['end'],
                    'text': '[inaudível]',
                    'speaker': speaker_mapping.get(speaker, speaker),
                    'avg_logprob': -1.0,
                    'words': []
                }
                transcription_segments.append(synthetic_segment)
        transcription_segments.sort(key=lambda x: x['start'])

    logger.info("Speaker alignment completed successfully")
    return transcription_segments