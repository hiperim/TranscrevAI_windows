# diarization.py - Final Implementation with Online Diarization Logic
"""
Speaker Diarization using an online, incremental approach to identify speakers.
This implementation finds high-confidence anchor speakers first, then classifies all
speech segments against them, creating new speakers as needed. This is the most
robust method for handling speakers with very few utterances.
"""

import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, cast
from dataclasses import dataclass
import torch
import librosa
import soundfile as sf
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DiarizationSegment:
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float
    text: Optional[str] = None

class TwoPassDiarizer:
    """Implements online diarization using an anchor-and-classify approach."""

    def __init__(self, device: str = "cpu"):
        logger.info("Initializing Silero-VAD based diarizer...")
        self.device = device
        try:
            project_root = Path(__file__).resolve().parent.parent
            local_repo_path = str(project_root / "src" / "silero-vad")
            self.vad_model, self.utils = cast(tuple, torch.hub.load(
                repo_or_dir=local_repo_path,
                model='silero_vad',
                source='local'
            ))
            logger.info("Silero-VAD model loaded successfully from local source.")
        except Exception as e:
            logger.error(f"Failed to load Silero-VAD model from local source: {e}", exc_info=True)
            self.vad_model = None

    def _estimate_dbscan_eps(self, features: np.ndarray, min_samples: int) -> float:
        """
        Estimates the optimal eps value for DBSCAN using the K-distance graph method.
        """
        if len(features) < min_samples:
            return 0.5
        k = min_samples - 1
        neighbors = NearestNeighbors(n_neighbors=k).fit(features)
        distances, _ = neighbors.kneighbors(features)
        distances = np.sort(distances[:, k-1], axis=0)
        try:
            second_derivative = np.diff(distances, 2)
            if len(second_derivative) == 0:
                return float(distances[-1] * 0.5) if len(distances) > 0 else 0.5
            elbow_index = np.argmax(second_derivative) + 1
            optimal_eps = distances[elbow_index]
        except (IndexError, ValueError):
            optimal_eps = np.median(distances)
        return max(0.1, min(float(optimal_eps), 1.0))

    async def diarize(self, audio_path: str, transcription_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Asynchronously performs speaker diarization by running the synchronous
        processing in a separate thread to avoid blocking the event loop.
        """
        if not self.vad_model:
            logger.error("Silero-VAD model not available. Falling back to single speaker diarization.")
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
        Definitive diarization logic using an optimized anchor-and-classify method.
        """
        logger.info(f"Starting definitive online diarization for {audio_path}")
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            (get_speech_timestamps, _, _, _, _) = self.utils
            
            all_speech_timestamps = []
            all_features = []

            num_samples = len(audio)
            for i in range(0, num_samples, 30 * sr):
                chunk = audio[i:i + 30 * sr]
                
                speech_timestamps_chunk = get_speech_timestamps(torch.from_numpy(chunk), self.vad_model, sampling_rate=sr)
                
                for segment in speech_timestamps_chunk:
                    start_sample = segment['start']
                    end_sample = segment['end']
                    segment_audio = chunk[start_sample:end_sample]
                    
                    if len(segment_audio) > 0:
                        mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=20)
                        mfcc_delta = librosa.feature.delta(mfcc)
                        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                        combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
                        all_features.append(np.mean(combined, axis=1))
                        
                        # Adjust segment times to be relative to the whole audio
                        segment['start'] += i
                        segment['end'] += i
                        all_speech_timestamps.append(segment)

            if not all_speech_timestamps:
                raise ValueError("No speech detected.")

            features_normalized = normalize(np.array(all_features))

            # --- Optimized Anchor and Classify Logic ---
            final_labels = np.full(len(features_normalized), -1, dtype=int)
            speaker_profiles = [] # List of centroids
            next_speaker_id = 0

            # Find high-confidence anchors first
            anchor_min_samples = 5
            if len(features_normalized) > anchor_min_samples:
                eps = self._estimate_dbscan_eps(features_normalized, anchor_min_samples)
                clustering = DBSCAN(eps=eps, min_samples=anchor_min_samples).fit(features_normalized)
                unique_anchors = sorted([l for l in np.unique(clustering.labels_) if l != -1])
                
                for label in unique_anchors:
                    anchor_indices = np.where(clustering.labels_ == label)[0]
                    speaker_profiles.append(np.mean(features_normalized[anchor_indices], axis=0))
                    final_labels[anchor_indices] = next_speaker_id
                    next_speaker_id += 1

            # Classify remaining points
            unclassified_indices = np.where(final_labels == -1)[0]
            if len(unclassified_indices) > 0:
                unclassified_features = features_normalized[unclassified_indices]
                
                if not speaker_profiles: # No anchors found, treat first point as first speaker
                    speaker_profiles.append(unclassified_features[0])
                    final_labels[unclassified_indices[0]] = next_speaker_id
                    next_speaker_id += 1
                    # Re-run on remaining points
                    unclassified_indices = np.where(final_labels == -1)[0]
                    unclassified_features = features_normalized[unclassified_indices]

                if len(unclassified_features) > 0:
                    # Vectorized distance calculation
                    distances = cdist(unclassified_features, np.array(speaker_profiles))
                    closest_profile_indices = np.argmin(distances, axis=1)
                    min_distances = distances[np.arange(len(distances)), closest_profile_indices]

                    similarity_threshold = self._estimate_dbscan_eps(features_normalized, min_samples=2) * 1.2

                    for i, original_idx in enumerate(unclassified_indices):
                        if min_distances[i] < similarity_threshold:
                            final_labels[original_idx] = closest_profile_indices[i]
                        else:
                            final_labels[original_idx] = next_speaker_id
                            speaker_profiles.append(unclassified_features[i]) # Simplistic profile creation
                            next_speaker_id += 1
            
            num_speakers = len(speaker_profiles)
            logger.info(f"Optimized online diarization complete. Found {num_speakers} speakers.")

            if num_speakers == 0:
                raise ValueError("Clustering did not find any speakers.")

            diarization_segments = []
            speaker_map = {label: i for i, label in enumerate(sorted(list(set(final_labels))))}

            for i, segment in enumerate(all_speech_timestamps):
                original_label = final_labels[i]
                remapped_label = speaker_map[original_label]
                speaker_id = f'SPEAKER_{remapped_label + 1:02d}'
                diarization_segments.append({
                    'start': segment['start'] / sr,
                    'end': segment['end'] / sr,
                    'speaker': speaker_id
                })

            aligned_segments = force_transcription_segmentation(transcription_segments, diarization_segments)

            return {"segments": aligned_segments, "num_speakers": num_speakers}

        except Exception as e:
            logger.error(f"Diarization failed: {e}", exc_info=True)
            for seg in transcription_segments:
                seg['speaker'] = 'SPEAKER_01'
            return {"segments": transcription_segments, "num_speakers": 1}

# --- Critical Alignment Function (Preserved and Corrected) ---

def force_transcription_segmentation(
    transcription_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Force transcription segmentation based on diarization boundaries.
    """
    
    if not diarization_segments or not transcription_segments:
        return transcription_segments or []

    final_segments = []
    time_to_speaker = {}
    
    for seg in diarization_segments:
        start = int(seg.get('start', 0) * 10)
        end = int(seg.get('end', 0) * 10)
        speaker = seg.get('speaker', 'Unknown')
        
        for i in range(start, end):
            time_to_speaker[i] = speaker

    for trans_seg in transcription_segments:
        trans_start = trans_seg.get('start', 0)
        trans_end = trans_seg.get('end', 0)
        trans_text = trans_seg.get('text', '').strip()
        
        if not trans_text:
            continue

        segment_speakers = []
        for i in range(int(trans_start * 10), int(trans_end * 10)):
            if i in time_to_speaker:
                segment_speakers.append(time_to_speaker[i])

        if segment_speakers:
            speaker_counts = {}
            for speaker in segment_speakers:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            dominant_speaker = max(speaker_counts, key=lambda k: speaker_counts[k])
        else:
            if time_to_speaker:
                closest_time = min(
                    time_to_speaker.keys(), 
                    key=lambda x: abs(x - int(trans_start * 10))
                )
                dominant_speaker = time_to_speaker[closest_time]
            else:
                dominant_speaker = 'SPEAKER_01'

        new_seg = trans_seg.copy()
        new_seg['speaker'] = dominant_speaker
        final_segments.append(new_seg)

    return final_segments