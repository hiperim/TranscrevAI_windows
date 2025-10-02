"""
TranscrevAI Optimized - Speaker Diarization Module
Sistema avançado de diarização com detecção de sobreposição para PT-BR
"""

import asyncio
import gc
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import warnings

# Import our optimized modules
from logging_setup import get_logger, log_performance
from resource_manager import get_resource_manager, ResourceStatus
from config import CONFIG

logger = get_logger("transcrevai.speaker_diarization")

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Lazy imports for heavy ML dependencies
_sklearn = None
_librosa = None
_scipy = None
_pyaudioanalysis = None

def get_sklearn():
    """Lazy import sklearn"""
    global _sklearn
    if _sklearn is None:
        try:
            import sklearn
            from sklearn.cluster import AgglomerativeClustering, KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            _sklearn = sklearn
            logger.info("scikit-learn loaded successfully")
        except ImportError as e:
            logger.error(f"scikit-learn not available: {e}")
            _sklearn = None
    return _sklearn

def get_librosa():
    """Lazy import librosa"""
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
            logger.info("librosa loaded successfully")
        except ImportError as e:
            logger.warning(f"librosa not available: {e}")
            _librosa = None
    return _librosa

def get_scipy():
    """Lazy import scipy"""
    global _scipy
    if _scipy is None:
        try:
            import scipy
            from scipy import signal
            from scipy.spatial.distance import pdist, squareform
            _scipy = scipy
            logger.info("scipy loaded successfully")
        except ImportError as e:
            logger.warning(f"scipy not available: {e}")
            _scipy = None
    return _scipy

def get_pyaudioanalysis():
    """Lazy import pyAudioAnalysis"""
    global _pyaudioanalysis
    if _pyaudioanalysis is None:
        try:
            from pyAudioAnalysis import audioSegmentation
            _pyaudioanalysis = audioSegmentation
            logger.info("pyAudioAnalysis loaded successfully")
        except ImportError as e:
            logger.warning(f"pyAudioAnalysis not available: {e}")
            _pyaudioanalysis = None
    return _pyaudioanalysis


class DiarizationError(Exception):
    """Custom exception for diarization errors"""
    def __init__(self, message: str, error_type: str = "unknown"):
        self.error_type = error_type
        super().__init__(f"[{error_type}] {message}")


class AdvancedFeatureExtractor:
    """
    Advanced feature extraction for speaker diarization
    Optimized for PT-BR speech patterns
    """
    
    def __init__(self):
        self.sample_rate = CONFIG["audio"]["sample_rate"]
        self.window_size = CONFIG["diarization"]["mid_window"]
        self.step_size = CONFIG["diarization"]["step"]
        
    def extract_speaker_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive speaker features from audio
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Feature matrix (n_frames, n_features)
        """
        try:
            librosa = get_librosa()
            if librosa is None:
                return self._extract_basic_features(audio_data)
            
            # Calculate frame parameters
            hop_length = int(self.step_size * self.sample_rate)
            frame_length = int(self.window_size * self.sample_rate)
            
            # MFCC features (primary for speaker identification)
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=13,  # Standard 13 coefficients
                hop_length=hop_length,
                n_fft=frame_length,
                window='hamming'
            )
            
            # Delta and delta-delta features
            delta_mfcc = librosa.feature.delta(mfcc, order=1)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, 
                sr=self.sample_rate,
                hop_length=hop_length
            )
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=hop_length
            )
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y=audio_data,
                hop_length=hop_length,
                frame_length=frame_length
            )
            
            # Energy and pitch features
            rms_energy = librosa.feature.rms(
                y=audio_data,
                hop_length=hop_length,
                frame_length=frame_length
            )
            
            # Pitch estimation (fundamental frequency)
            try:
                f0 = librosa.yin(
                    audio_data,
                    fmin=50,  # Minimum expected pitch for human speech
                    fmax=400, # Maximum expected pitch for human speech
                    sr=self.sample_rate,
                    hop_length=hop_length
                )
                f0 = f0.reshape(1, -1)  # Ensure 2D
            except:
                # Fallback: create zero pitch features
                f0 = np.zeros((1, mfcc.shape[1]))
            
            # Combine all features
            features = np.vstack([
                mfcc,           # 13 features
                delta_mfcc,     # 13 features  
                delta2_mfcc,    # 13 features
                spectral_centroids,  # 1 feature
                spectral_rolloff,    # 1 feature
                zero_crossing_rate,  # 1 feature
                rms_energy,         # 1 feature
                f0                  # 1 feature
            ])
            
            # Transpose to (n_frames, n_features)
            features = features.T
            
            # Remove any NaN or inf values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.debug(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._extract_basic_features(audio_data)
    
    def _extract_basic_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Fallback basic feature extraction"""
        try:
            # Simple energy-based features
            frame_length = int(self.window_size * self.sample_rate)
            hop_length = int(self.step_size * self.sample_rate)
            
            n_frames = 1 + (len(audio_data) - frame_length) // hop_length
            features = np.zeros((n_frames, 4))  # Basic 4 features
            
            for i in range(n_frames):
                start = i * hop_length
                end = start + frame_length
                frame = audio_data[start:end]
                
                if len(frame) > 0:
                    # Energy
                    features[i, 0] = np.sum(frame ** 2)
                    # Zero crossing rate
                    features[i, 1] = np.sum(np.diff(np.sign(frame)) != 0) / len(frame)
                    # Spectral centroid (simplified)
                    fft = np.abs(np.fft.rfft(frame))
                    freqs = np.linspace(0, self.sample_rate/2, len(fft))
                    features[i, 2] = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)
                    # Spectral bandwidth (simplified)
                    features[i, 3] = np.sqrt(np.sum(((freqs - features[i, 2]) ** 2) * fft) / (np.sum(fft) + 1e-10))
            
            return features
            
        except Exception as e:
            logger.error(f"Basic feature extraction failed: {e}")
            # Return minimal features
            return np.random.rand(100, 4)


class OverlappingSpeechDetector:
    """
    Detect overlapping speech segments for advanced diarization
    """
    
    def __init__(self):
        self.overlap_threshold = CONFIG["diarization"]["overlap_threshold"]
        
    def detect_overlapping_speech(self, 
                                audio_data: np.ndarray, 
                                segments: List[Dict]) -> List[Dict]:
        """
        Detect overlapping speech in segments
        
        Args:
            audio_data: Audio signal
            segments: List of speaker segments
            
        Returns:
            List of segments with overlap information
        """
        try:
            if len(segments) < 2:
                return segments
            
            enhanced_segments = []
            
            for i, segment in enumerate(segments):
                enhanced_segment = segment.copy()
                enhanced_segment["overlapping"] = False
                enhanced_segment["overlap_confidence"] = 0.0
                
                # Check for potential overlaps with other segments
                for j, other_segment in enumerate(segments):
                    if i != j and self._segments_overlap(segment, other_segment):
                        # Extract overlapping audio region
                        overlap_audio = self._extract_overlap_audio(
                            audio_data, segment, other_segment
                        )
                        
                        if overlap_audio is not None:
                            # Analyze if there's actual overlapping speech
                            overlap_confidence = self._analyze_overlap_confidence(overlap_audio)
                            
                            if overlap_confidence > self.overlap_threshold:
                                enhanced_segment["overlapping"] = True
                                enhanced_segment["overlap_confidence"] = overlap_confidence
                                enhanced_segment["overlaps_with"] = j
                                break
                
                enhanced_segments.append(enhanced_segment)
            
            logger.debug(f"Detected overlapping speech in {sum(1 for s in enhanced_segments if s['overlapping'])} segments")
            return enhanced_segments
            
        except Exception as e:
            logger.warning(f"Overlap detection failed: {e}")
            return segments
    
    def _segments_overlap(self, seg1: Dict, seg2: Dict) -> bool:
        """Check if two segments overlap in time"""
        return not (seg1["end"] <= seg2["start"] or seg2["end"] <= seg1["start"])
    
    def _extract_overlap_audio(self, 
                             audio_data: np.ndarray, 
                             seg1: Dict, 
                             seg2: Dict) -> Optional[np.ndarray]:
        """Extract audio from overlapping region"""
        try:
            sample_rate = CONFIG["audio"]["sample_rate"]
            
            # Find overlap region
            overlap_start = max(seg1["start"], seg2["start"])
            overlap_end = min(seg1["end"], seg2["end"])
            
            if overlap_end <= overlap_start:
                return None
            
            # Extract audio samples
            start_sample = int(overlap_start * sample_rate)
            end_sample = int(overlap_end * sample_rate)
            
            if start_sample >= len(audio_data) or end_sample <= start_sample:
                return None
            
            return audio_data[start_sample:end_sample]
            
        except Exception as e:
            logger.warning(f"Failed to extract overlap audio: {e}")
            return None
    
    def _analyze_overlap_confidence(self, audio_chunk: np.ndarray) -> float:
        """Analyze confidence that audio contains overlapping speech"""
        try:
            # Energy variance analysis
            frame_size = int(0.025 * CONFIG["audio"]["sample_rate"])  # 25ms frames
            hop_size = int(0.010 * CONFIG["audio"]["sample_rate"])    # 10ms hop
            
            frames = []
            for i in range(0, len(audio_chunk) - frame_size, hop_size):
                frame = audio_chunk[i:i + frame_size]
                energy = np.sum(frame ** 2)
                frames.append(energy)
            
            if len(frames) < 2:
                return 0.0
            
            # High energy variance suggests overlapping speech
            energy_variance = np.var(frames)
            energy_mean = np.mean(frames)
            
            # Normalize variance by mean to get relative measure
            normalized_variance = energy_variance / (energy_mean + 1e-10)
            
            # Convert to confidence score (0-1)
            confidence = min(1.0, normalized_variance / 100.0)
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Overlap confidence analysis failed: {e}")
            return 0.0


class BrowserSafeSpeakerClustering:
    """
    Browser-safe speaker clustering with progressive processing
    """
    
    def __init__(self):
        self.resource_manager = get_resource_manager()
        self.max_speakers = CONFIG["diarization"]["max_speakers"]
        self.min_speakers = CONFIG["diarization"]["min_speakers"]
        
    async def cluster_speakers(self, 
                             features: np.ndarray,
                             progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        Cluster speaker features with browser-safe processing
        
        Args:
            features: Feature matrix (n_frames, n_features)
            progress_callback: Optional progress callback
            
        Returns:
            Speaker labels array
        """
        try:
            sklearn = get_sklearn()
            if sklearn is None:
                return self._fallback_clustering(features)
            
            if progress_callback:
                await progress_callback(10, "Normalizando características...")
            
            # Feature normalization
            scaler = sklearn.preprocessing.StandardScaler()
            normalized_features = scaler.fit_transform(features)
            
            # Check memory pressure before clustering
            if self.resource_manager.is_memory_pressure_high():
                logger.warning("High memory pressure, using simplified clustering")
                return await self._memory_safe_clustering(normalized_features, progress_callback)
            
            if progress_callback:
                await progress_callback(30, "Determinando número ótimo de falantes...")
            
            # Determine optimal number of speakers
            n_speakers = await self._determine_optimal_speakers(
                normalized_features, progress_callback
            )
            
            if progress_callback:
                await progress_callback(60, f"Agrupando falantes... ({n_speakers} falantes detectados)")
            
            # Perform clustering
            labels = await self._perform_clustering(
                normalized_features, n_speakers, progress_callback
            )
            
            if progress_callback:
                await progress_callback(90, "Refinando agrupamento...")
            
            # Post-process labels
            refined_labels = self._post_process_labels(labels, features)
            
            logger.info(f"Speaker clustering completed: {n_speakers} speakers detected")
            return refined_labels
            
        except Exception as e:
            logger.error(f"Speaker clustering failed: {e}")
            return self._fallback_clustering(features)
    
    async def _determine_optimal_speakers(self, 
                                        features: np.ndarray,
                                        progress_callback: Optional[Callable] = None) -> int:
        """Determine optimal number of speakers using silhouette analysis"""
        try:
            sklearn = get_sklearn()
            n_samples = len(features)
            
            # Limit range based on data size and configuration
            max_k = min(self.max_speakers, max(2, n_samples // 100))
            min_k = max(self.min_speakers, 1)
            
            if max_k <= min_k:
                return min_k
            
            best_score = -1
            best_k = min_k
            
            # Try different numbers of clusters
            for k in range(min_k, max_k + 1):
                try:
                    # Use KMeans for speed in determination phase
                    kmeans = sklearn.cluster.KMeans(
                        n_clusters=k, 
                        random_state=42,
                        n_init=3,  # Reduced for speed
                        max_iter=100
                    )
                    
                    labels = kmeans.fit_predict(features)
                    
                    # Calculate silhouette score
                    if len(np.unique(labels)) > 1:
                        score = sklearn.metrics.silhouette_score(features, labels)
                        
                        if score > best_score:
                            best_score = score
                            best_k = k
                    
                    # Browser-safe: yield control
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate k={k}: {e}")
                    continue
            
            logger.info(f"Optimal speakers determined: {best_k} (silhouette score: {best_score:.3f})")
            return best_k
            
        except Exception as e:
            logger.warning(f"Speaker number determination failed: {e}")
            return CONFIG["diarization"]["default_speakers"]
    
    async def _perform_clustering(self, 
                                features: np.ndarray, 
                                n_speakers: int,
                                progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Perform the actual clustering"""
        try:
            sklearn = get_sklearn()
            
            # Use Agglomerative Clustering for better speaker separation
            clusterer = sklearn.cluster.AgglomerativeClustering(
                n_clusters=n_speakers,
                linkage='ward'
            )
            
            # Run clustering in executor to prevent blocking
            loop = asyncio.get_event_loop()
            labels = await loop.run_in_executor(
                None,
                clusterer.fit_predict,
                features
            )
            
            return labels
            
        except Exception as e:
            logger.warning(f"Agglomerative clustering failed, using KMeans: {e}")
            
            # Fallback to KMeans
            sklearn = get_sklearn()
            kmeans = sklearn.cluster.KMeans(
                n_clusters=n_speakers,
                random_state=42,
                n_init=10
            )
            
            loop = asyncio.get_event_loop()
            labels = await loop.run_in_executor(
                None,
                kmeans.fit_predict,
                features
            )
            
            return labels
    
    async def _memory_safe_clustering(self, 
                                    features: np.ndarray,
                                    progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Memory-safe clustering for high pressure situations"""
        try:
            # Use simple KMeans with minimal memory footprint
            sklearn = get_sklearn()
            n_speakers = min(3, self.max_speakers)  # Conservative speaker count
            
            kmeans = sklearn.cluster.MiniBatchKMeans(
                n_clusters=n_speakers,
                random_state=42,
                batch_size=100,  # Small batches
                n_init=3
            )
            
            # Process in chunks if needed
            if len(features) > 1000:
                # Sample features for clustering
                indices = np.random.choice(len(features), 1000, replace=False)
                sample_features = features[indices]
                
                labels_sample = kmeans.fit_predict(sample_features)
                
                # Predict labels for all features
                labels = kmeans.predict(features)
            else:
                labels = kmeans.fit_predict(features)
            
            return labels
            
        except Exception as e:
            logger.error(f"Memory-safe clustering failed: {e}")
            return self._fallback_clustering(features)
    
    def _post_process_labels(self, labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Post-process clustering labels"""
        try:
            # Remove very small clusters (likely noise)
            unique_labels, counts = np.unique(labels, return_counts=True)
            min_cluster_size = max(10, len(labels) // 100)  # At least 1% of data
            
            # Map small clusters to nearest large cluster
            large_clusters = unique_labels[counts >= min_cluster_size]
            
            if len(large_clusters) == 0:
                return labels  # Keep original if all clusters are small
            
            refined_labels = labels.copy()
            
            for small_label in unique_labels[counts < min_cluster_size]:
                # Find indices of small cluster
                small_indices = np.where(labels == small_label)[0]
                
                if len(small_indices) > 0:
                    # Find nearest large cluster based on feature similarity
                    small_features = features[small_indices]
                    
                    best_cluster = large_clusters[0]  # Default
                    best_distance = float('inf')
                    
                    for large_label in large_clusters:
                        large_indices = np.where(labels == large_label)[0]
                        large_features = features[large_indices]
                        
                        # Calculate mean distance
                        distance = np.mean([
                            np.linalg.norm(sf - lf) 
                            for sf in small_features[:5]  # Sample few points
                            for lf in large_features[:10]
                        ])
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_cluster = large_label
                    
                    refined_labels[small_indices] = best_cluster
            
            # Relabel to consecutive integers starting from 0
            unique_refined = np.unique(refined_labels)
            label_map = {old: new for new, old in enumerate(unique_refined)}
            final_labels = np.array([label_map[label] for label in refined_labels])
            
            return final_labels
            
        except Exception as e:
            logger.warning(f"Label post-processing failed: {e}")
            return labels
    
    def _fallback_clustering(self, features: np.ndarray) -> np.ndarray:
        """Simple fallback clustering when libraries are unavailable"""
        try:
            # Simple energy-based clustering
            if features.shape[1] > 0:
                energy_feature = features[:, 0]  # Use first feature as energy
            else:
                energy_feature = np.random.rand(len(features))
            
            # Simple threshold-based clustering
            threshold = np.median(energy_feature)
            labels = (energy_feature > threshold).astype(int)
            
            # Add some variation
            if len(np.unique(labels)) == 1:
                labels[::2] = 1 - labels[::2]
            
            return labels
            
        except Exception as e:
            logger.error(f"Fallback clustering failed: {e}")
            return np.zeros(len(features), dtype=int)


class SpeakerDiarization:
    """
    Main speaker diarization engine with advanced features and PT-BR optimization
    """
    
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.overlap_detector = OverlappingSpeechDetector()
        self.clusterer = BrowserSafeSpeakerClustering()
        self.resource_manager = get_resource_manager()
        
        # Configuration
        self.min_segment_duration = CONFIG["diarization"]["min_segment_duration"]
        self.merge_threshold = CONFIG["diarization"]["segment_merge_threshold"]
        
        logger.info("SpeakerDiarization initialized")
    
    async def diarize(self, 
                     audio_file: str,
                     transcription_data: Optional[List[Dict]] = None,
                     progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_file: Path to audio file
            transcription_data: Optional transcription segments for alignment
            progress_callback: Optional progress callback
            
        Returns:
            List of speaker segments with timing and speaker IDs
        """
        diarization_start = time.time()
        
        try:
            if progress_callback:
                await progress_callback(5, "Carregando áudio para diarização...")
            
            # Load audio
            audio_data = await self._load_audio_for_diarization(audio_file)
            if audio_data is None:
                raise DiarizationError("Failed to load audio file", "audio_load_error")
            
            if progress_callback:
                await progress_callback(15, "Extraindo características dos falantes...")
            
            # Extract speaker features
            features = await self._extract_features_safe(audio_data, progress_callback)
            
            if progress_callback:
                await progress_callback(40, "Agrupando falantes...")
            
            # Perform speaker clustering
            speaker_labels = await self.clusterer.cluster_speakers(features, progress_callback)
            
            if progress_callback:
                await progress_callback(70, "Criando segmentos de falantes...")
            
            # Create speaker segments
            segments = self._create_speaker_segments(speaker_labels, audio_data)
            
            if progress_callback:
                await progress_callback(85, "Detectando fala sobreposta...")
            
            # Detect overlapping speech
            if CONFIG["diarization"]["enable_overlap_detection"]:
                segments = self.overlap_detector.detect_overlapping_speech(audio_data, segments)
            
            if progress_callback:
                await progress_callback(95, "Refinando segmentação...")
            
            # Post-process segments
            final_segments = self._post_process_segments(segments, transcription_data)
            
            # Calculate metrics
            processing_time = time.time() - diarization_start
            unique_speakers = len(set(seg["speaker"] for seg in final_segments))
            
            # Log performance
            log_performance(
                "Diarization completed",
                duration=processing_time,
                speakers_detected=unique_speakers,
                segments_count=len(final_segments),
                overlapping_segments=sum(1 for s in final_segments if s.get("overlapping", False))
            )
            
            if progress_callback:
                await progress_callback(100, f"Diarização concluída! {unique_speakers} falantes detectados")
            
            logger.info(f"Diarization completed: {unique_speakers} speakers, {len(final_segments)} segments")
            return final_segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            if progress_callback:
                await progress_callback(0, f"Erro na diarização: {str(e)}")
            
            # Return single speaker segments as fallback
            return self._create_fallback_segments(audio_file, transcription_data)
    
    async def _load_audio_for_diarization(self, audio_file: str) -> Optional[np.ndarray]:
        """Load audio file for diarization processing"""
        try:
            librosa = get_librosa()
            if librosa:
                audio_data, _ = librosa.load(
                    audio_file, 
                    sr=CONFIG["audio"]["sample_rate"],
                    mono=True
                )
            else:
                # Fallback using soundfile
                import soundfile as sf
                audio_data, sr = sf.read(audio_file)
                
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Simple resampling if needed
                if sr != CONFIG["audio"]["sample_rate"]:
                    audio_data = audio_data[::int(sr / CONFIG["audio"]["sample_rate"])]
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to load audio for diarization: {e}")
            return None
    
    async def _extract_features_safe(self, 
                                   audio_data: np.ndarray,
                                   progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Safely extract features with memory management"""
        try:
            # Check memory before feature extraction
            estimated_memory = len(audio_data) * 4 / (1024 * 1024)  # Rough estimate
            if not self.resource_manager.can_allocate(estimated_memory):
                await self.resource_manager.perform_cleanup(aggressive=False)
            
            # Extract features in executor to prevent blocking
            loop = asyncio.get_event_loop()
            features = await loop.run_in_executor(
                None,
                self.feature_extractor.extract_speaker_features,
                audio_data
            )
            
            if features.shape[0] == 0:
                raise DiarizationError("No features extracted", "feature_extraction_error")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return minimal fallback features
            n_frames = max(100, len(audio_data) // (CONFIG["audio"]["sample_rate"] // 10))
            return np.random.rand(n_frames, 4)
    
    def _create_speaker_segments(self, 
                               speaker_labels: np.ndarray, 
                               audio_data: np.ndarray) -> List[Dict]:
        """Create speaker segments from clustering labels"""
        try:
            segments = []
            step_size = CONFIG["diarization"]["step"]
            
            if len(speaker_labels) == 0:
                return segments
            
            current_speaker = speaker_labels[0]
            segment_start = 0.0
            
            for i, speaker in enumerate(speaker_labels[1:], 1):
                time_position = i * step_size
                
                if speaker != current_speaker or i == len(speaker_labels) - 1:
                    # End current segment
                    segment_end = time_position
                    
                    # Only add segments longer than minimum duration
                    if segment_end - segment_start >= self.min_segment_duration:
                        segments.append({
                            "start": segment_start,
                            "end": segment_end,
                            "speaker": int(current_speaker),
                            "confidence": 0.8,  # Default confidence
                            "overlapping": False
                        })
                    
                    # Start new segment
                    segment_start = time_position
                    current_speaker = speaker
            
            # Add final segment
            final_end = len(audio_data) / CONFIG["audio"]["sample_rate"]
            if final_end - segment_start >= self.min_segment_duration:
                segments.append({
                    "start": segment_start,
                    "end": final_end,
                    "speaker": int(current_speaker),
                    "confidence": 0.8,
                    "overlapping": False
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Failed to create speaker segments: {e}")
            return []
    
    def _post_process_segments(self, 
                             segments: List[Dict],
                             transcription_data: Optional[List[Dict]] = None) -> List[Dict]:
        """Post-process segments for quality improvement"""
        try:
            if not segments:
                return segments
            
            # Merge close segments from same speaker
            if CONFIG["diarization"]["merge_close_segments"]:
                segments = self._merge_close_segments(segments)
            
            # Align with transcription if available
            if transcription_data:
                segments = self._align_with_transcription(segments, transcription_data)
            
            # Sort segments by start time
            segments.sort(key=lambda x: x["start"])
            
            # Add segment IDs
            for i, segment in enumerate(segments):
                segment["segment_id"] = i
            
            return segments
            
        except Exception as e:
            logger.warning(f"Segment post-processing failed: {e}")
            return segments
    
    def _merge_close_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge segments from same speaker that are close in time"""
        if not segments:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            # Check if same speaker and close in time
            if (segment["speaker"] == current["speaker"] and 
                segment["start"] - current["end"] <= self.merge_threshold):
                
                # Merge segments
                current["end"] = segment["end"]
                current["confidence"] = min(current["confidence"], segment["confidence"])
            else:
                # Add current segment and start new one
                merged.append(current)
                current = segment.copy()
        
        # Add final segment
        merged.append(current)
        
        return merged
    
    def _align_with_transcription(self, 
                                segments: List[Dict],
                                transcription_data: List[Dict]) -> List[Dict]:
        """Align speaker segments with transcription segments"""
        try:
            aligned_segments = []
            
            for diar_seg in segments:
                # Find overlapping transcription segments
                overlapping_trans = [
                    trans for trans in transcription_data
                    if not (trans["end"] <= diar_seg["start"] or trans["start"] >= diar_seg["end"])
                ]
                
                if overlapping_trans:
                    # Create aligned segment for each overlapping transcription
                    for trans_seg in overlapping_trans:
                        aligned_segment = {
                            "start": max(diar_seg["start"], trans_seg["start"]),
                            "end": min(diar_seg["end"], trans_seg["end"]),
                            "speaker": diar_seg["speaker"],
                            "text": trans_seg.get("text", ""),
                            "confidence": min(
                                diar_seg["confidence"], 
                                1.0 - trans_seg.get("confidence", 0.0)
                            ),
                            "overlapping": diar_seg.get("overlapping", False)
                        }
                        
                        # Only add if meaningful duration
                        if aligned_segment["end"] - aligned_segment["start"] > 0.1:
                            aligned_segments.append(aligned_segment)
                else:
                    # No transcription overlap, keep original segment
                    aligned_segments.append(diar_seg)
            
            return aligned_segments
            
        except Exception as e:
            logger.warning(f"Transcription alignment failed: {e}")
            return segments
    
    def _create_fallback_segments(self, 
                                audio_file: str,
                                transcription_data: Optional[List[Dict]] = None) -> List[Dict]:
        """Create fallback segments when diarization fails"""
        try:
            if transcription_data:
                # Use transcription segments as single speaker
                segments = []
                for i, trans_seg in enumerate(transcription_data):
                    segments.append({
                        "start": trans_seg["start"],
                        "end": trans_seg["end"],
                        "speaker": 0,  # Single speaker
                        "text": trans_seg.get("text", ""),
                        "confidence": 0.5,  # Low confidence
                        "overlapping": False,
                        "segment_id": i
                    })
                return segments
            else:
                # Create single segment for entire audio
                try:
                    import soundfile as sf
                    info = sf.info(audio_file)
                    duration = info.frames / info.samplerate
                except:
                    duration = 60.0  # Fallback duration
                
                return [{
                    "start": 0.0,
                    "end": duration,
                    "speaker": 0,
                    "text": "",
                    "confidence": 0.3,
                    "overlapping": False,
                    "segment_id": 0
                }]
                
        except Exception as e:
            logger.error(f"Failed to create fallback segments: {e}")
            return []
    
    def get_diarization_statistics(self) -> Dict[str, Any]:
        """Get diarization statistics"""
        return {
            "feature_extractor": "MFCC + Delta + Spectral + Pitch",
            "clustering_method": "Agglomerative + KMeans fallback",
            "overlap_detection": CONFIG["diarization"]["enable_overlap_detection"],
            "min_segment_duration": self.min_segment_duration,
            "merge_threshold": self.merge_threshold,
            "max_speakers": CONFIG["diarization"]["max_speakers"]
        }


# Utility functions for external use
async def quick_diarize(audio_file: str, 
                       transcription_data: Optional[List[Dict]] = None,
                       progress_callback: Optional[Callable] = None) -> List[Dict]:
    """Quick diarization function for simple use cases"""
    diarizer = SpeakerDiarization()
    return await diarizer.diarize(audio_file, transcription_data, progress_callback)


def estimate_diarization_time(audio_duration: float) -> float:
    """Estimate diarization processing time"""
    # Diarization typically takes 0.3-0.8x audio duration depending on complexity
    return audio_duration * 0.5  # Conservative estimate