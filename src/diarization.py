# diarization.py - Refactored with DBSCAN for Advanced Accuracy
"""
Speaker Diarization using Silero-VAD, Librosa for MFCCs, and Scikit-learn for clustering.
This implementation uses DBSCAN for robust, adaptive clustering that automatically determines
the number of speakers and identifies noise. It runs the blocking diarization process in a 
separate thread to avoid freezing the WebSocket event loop.
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
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

@dataclass
class DiarizationSegment:
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float
    text: Optional[str] = None

class TwoPassDiarizer:
    """Lightweight diarization system using Silero-VAD, MFCCs, and DBSCAN Clustering."""

    def __init__(self, device: str = "cpu"):
        logger.info("Initializing Silero-VAD based diarizer...")
        self.device = device
        try:
            # Load the Silero VAD model from the local repository to ensure stability.
            local_repo_path = "src/silero-vad"
            self.vad_model, self.utils = cast(tuple, torch.hub.load(
                repo_or_dir=local_repo_path,
                model='silero_vad',
                source='local'
            ))
            logger.info("Silero-VAD model loaded successfully from local source.")
        except Exception as e:
            logger.error(f"Failed to load Silero-VAD model from local source: {e}", exc_info=True)
            self.vad_model = None

    def _estimate_dbscan_eps(self, mfccs_normalized: np.ndarray, min_samples: int) -> float:
        """
        Estimates the optimal eps value for DBSCAN using the K-distance graph method.
        """
        if len(mfccs_normalized) < min_samples:
            return 0.5 # Return a default value if not enough samples

        logger.info(f"Estimating DBSCAN eps with {len(mfccs_normalized)} samples and min_samples={min_samples}")
        
        # Calculate the distance to the k-th nearest neighbor for each point
        k = min_samples - 1
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(mfccs_normalized)
        distances, _ = neighbors_fit.kneighbors(mfccs_normalized)
        
        # Get the k-th distance for each point and sort them
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Find the point of maximum curvature (the "elbow")
        try:
            # This is a simple way to find the elbow of the curve
            # We look for the point with the largest second derivative
            second_derivative = np.diff(distances, 2)
            if len(second_derivative) == 0:
                # Fallback for very few points
                return float(distances[-1] * 0.5) if len(distances) > 0 else 0.5

            elbow_index = np.argmax(second_derivative) + 1 # +1 to adjust for diff index
            optimal_eps = distances[elbow_index]
        except IndexError:
            # Fallback if something goes wrong
            optimal_eps = np.median(distances)

        # Ensure eps is within a reasonable range
        optimal_eps = max(0.1, min(float(optimal_eps), 1.0))
        logger.info(f"Estimated optimal DBSCAN eps: {optimal_eps:.4f}")
        return float(optimal_eps)

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
        Synchronous diarization logic. Processes audio in chunks, performs VAD,
        extracts features, clusters them with DBSCAN, and aligns with transcription.
        """
        logger.info(f"Starting synchronous diarization for {audio_path}")
        
        (get_speech_timestamps, _, _, _, _) = self.utils
        
        all_speech_timestamps = []
        all_features = [] # Changed from all_mfccs to all_features
        
        CHUNK_DURATION = 30  # seconds
        
        try:
            with sf.SoundFile(audio_path, 'r') as audio_file:
                sr = audio_file.samplerate
                if sr != 16000:
                    raise ValueError(f"Audio file has a sample rate of {sr}, but 16000 is required.")

                chunk_size = CHUNK_DURATION * sr
                for i in range(0, audio_file.frames, chunk_size):
                    chunk = audio_file.read(chunk_size, dtype='float32')
                    if len(chunk.shape) > 1:
                        chunk = np.mean(chunk, axis=1)
                    
                    audio_tensor = torch.from_numpy(chunk).float()
                    speech_timestamps_chunk = get_speech_timestamps(audio_tensor, self.vad_model, sampling_rate=sr)
                    
                    for segment in speech_timestamps_chunk:
                        segment['start'] += i
                        segment['end'] += i
                        
                        segment_audio = chunk[segment['start'] - i : segment['end'] - i]
                        if len(segment_audio) > 0:
                            # Enhanced Feature Extraction
                            mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=20)
                            mfcc_delta = librosa.feature.delta(mfcc)
                            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                            
                            # Stack features and take the mean across the time axis
                            combined_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
                            mean_features = np.mean(combined_features, axis=1)
                            
                            all_features.append(mean_features)
                            all_speech_timestamps.append(segment)

            if not all_speech_timestamps:
                logger.warning("No speech detected. Defaulting to a single speaker.")
                for seg in transcription_segments:
                    seg['speaker'] = 'SPEAKER_01'
                return {"segments": transcription_segments, "num_speakers": 1}

            features_normalized = normalize(np.array(all_features))

            # DBSCAN Clustering with tuned min_samples
            min_samples = 3 # Lowered to be more sensitive to speakers with few utterances
            eps = self._estimate_dbscan_eps(features_normalized, min_samples)
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_normalized)
            final_labels = clustering.labels_
            
            # Number of clusters in labels, ignoring noise if present.
            num_speakers = len(set(final_labels)) - (1 if -1 in final_labels else 0)
            logger.info(f"DBSCAN found {num_speakers} speakers and {np.sum(final_labels == -1)} noise points.")

            if num_speakers == 0:
                logger.warning("DBSCAN did not find any speakers. Defaulting to a single speaker.")
                for seg in transcription_segments:
                    seg['speaker'] = 'SPEAKER_01'
                return {"segments": transcription_segments, "num_speakers": 1}

            # Create diarization segments
            diarization_segments = []
            for i, segment in enumerate(all_speech_timestamps):
                speaker_label = final_labels[i]
                if speaker_label == -1:
                    speaker_id = 'SPEAKER_UNKNOWN'
                else:
                    speaker_id = f'SPEAKER_{speaker_label + 1:02d}'
                
                diarization_segments.append({
                    'start': segment['start'] / sr,
                    'end': segment['end'] / sr,
                    'speaker': speaker_id
                })

            # Align transcription segments with diarization results
            aligned_segments = force_transcription_segmentation(transcription_segments, diarization_segments)

            return {
                "segments": aligned_segments,
                "num_speakers": num_speakers
            }

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
    
    # Build time-to-speaker mapping
    for seg in diarization_segments:
        start = int(seg.get('start', 0) * 10)
        end = int(seg.get('end', 0) * 10)
        speaker = seg.get('speaker', 'Unknown')
        
        for i in range(start, end):
            time_to_speaker[i] = speaker

    # Process transcription segments
    for trans_seg in transcription_segments:
        trans_start = trans_seg.get('start', 0)
        trans_end = trans_seg.get('end', 0)
        trans_text = trans_seg.get('text', '').strip()
        
        if not trans_text:
            continue

        # Find dominant speaker for this segment
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
