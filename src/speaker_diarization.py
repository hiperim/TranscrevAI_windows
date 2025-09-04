import asyncio
import logging
import os
import static_ffmpeg
import numpy as np
import librosa
import shutil
import time
from scipy.fftpack import fft
from scipy import signal
from scipy.ndimage import median_filter
from enum import Enum
from unittest.mock import patch
from scipy.io import wavfile
import warnings

# Import scikit-learn components
try:
    # Suppress sklearn version warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    StandardScaler = None
    PCA = None
    KMeans = None
    AgglomerativeClustering = None
    silhouette_score = None

# Import pyAudioAnalysis
try:
    # Suppress sklearn warnings from PyAudioAnalysis
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        from pyAudioAnalysis import audioSegmentation as aS
        # Fix: Handle the specific import that was causing issues
        try:
            from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction
        except ImportError:
            mid_feature_extraction = None
    PYAUDIO_ANALYSIS_AVAILABLE = True
except ImportError as e:
    PYAUDIO_ANALYSIS_AVAILABLE = False
    aS = None
    mid_feature_extraction = None

from src.file_manager import FileManager
from src.logging_setup import setup_app_logging

# Use proper logging setup first
logger = setup_app_logging(logger_name="transcrevai.speaker_diarization")

# Lazy import for PyAnnote.Audio neural diarization
PYANNOTE_AVAILABLE = False
Pipeline = None
VoiceActivityDetection = None
torch = None
_pyannote_imports_attempted = False


def _ensure_pyannote_imports():
    """Lazy import of PyAnnote.Audio dependencies"""
    global PYANNOTE_AVAILABLE, Pipeline, VoiceActivityDetection, torch, _pyannote_imports_attempted
    
    if _pyannote_imports_attempted:
        return PYANNOTE_AVAILABLE
    
    _pyannote_imports_attempted = True
    
    try:
        from pyannote.audio import Pipeline as _Pipeline
        from pyannote.audio.pipelines import VoiceActivityDetection as _VAD
        import torch as _torch
        
        Pipeline = _Pipeline
        VoiceActivityDetection = _VAD
        torch = _torch
        PYANNOTE_AVAILABLE = True
        logger.info("PyAnnote.Audio dependencies loaded successfully")
    except ImportError as e:
        logger.warning(f"PyAnnote.Audio not available: {e}")
        PYANNOTE_AVAILABLE = False
        Pipeline = None
        VoiceActivityDetection = None
        torch = None
    
    return PYANNOTE_AVAILABLE

def load_audio_librosa(audio_file, sr=None, mono=True):
    """
    Load audio using librosa
    
    Args:
        audio_file: Path to audio file
        sr: Target sample rate (None to keep original)
        mono: Convert to mono if True
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    return librosa.load(audio_file, sr=sr, mono=mono)

from config.app_config import PYAUDIOANALYSIS_CONFIG

class DiarizationError(Enum):
    FILE_NOT_FOUND = 1
    INVALID_FORMAT = 2
    EMPTY_AUDIO = 3
    INSUFFICIENT_DATA = 4

class SpeakerDiarization:
    """PyAnnote.Audio neural speaker diarization with TorchCodec support"""

    def __init__(self):
        self.pipeline = None
        self._device = None # Lazy device detection
        self._ffmpeg_path = None # Lazy FFmpeg setup

    @property
    def device(self):
        """Lazy device detection only when needed"""
        if self._device is None:
            if _ensure_pyannote_imports() and torch:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"PyAnnote device: {self._device}")
            else:
                self._device = "cpu"
        return self._device

    @property
    def ffmpeg_path(self):
        """Lazy FFmpeg path detection"""
        if self._ffmpeg_path is None:
            try:
                if shutil.which('ffmpeg') is None:
                    # Lazy load static_ffmpeg
                    try:
                        import static_ffmpeg
                        static_ffmpeg.add_paths()
                    except ImportError:
                        logger.warning("static_ffmpeg not available")
                self._ffmpeg_path = shutil.which('ffmpeg')
            except Exception as e:
                logger.warning(f"FFmpeg setup failed: {e}")
                self._ffmpeg_path = None
        return self._ffmpeg_path

    async def _load_pyannote_pipeline(self):
        """Load PyAnnote.Audio pipeline"""
        if self.pipeline is not None:
            return self.pipeline

        # Ensure PyAnnote dependencies are loaded
        if not _ensure_pyannote_imports():
            raise RuntimeError("PyAnnote.Audio not available")

        try:
            loop = asyncio.get_event_loop()
            
            # Fix: Handle the case where Pipeline.from_pretrained might not have these parameters
            if Pipeline is None:
                raise RuntimeError("Pipeline not available - PyAnnote.Audio not properly imported")
                
            try:
                self.pipeline = await loop.run_in_executor(
                    None,
                    Pipeline.from_pretrained,
                    PYANNOTE_CONFIG["pipeline"],
                    PYANNOTE_CONFIG["cache_dir"], # Fix: Remove parameter name
                    PYANNOTE_CONFIG.get("use_auth_token") # Fix: Remove parameter name
                )
            except TypeError:
                # Fallback if parameters don't exist
                if Pipeline is None:
                    raise RuntimeError("Pipeline not available - PyAnnote.Audio not properly imported")
                self.pipeline = await loop.run_in_executor(
                    None,
                    Pipeline.from_pretrained,
                    PYANNOTE_CONFIG["pipeline"]
                )

            # Configure pipeline
            if hasattr(self.pipeline, "to") and self.device and torch is not None: # Fix: Add torch None check
                self.pipeline = self.pipeline.to(torch.device(self.device))

            logger.info(f"PyAnnote pipeline loaded: {PYANNOTE_CONFIG['pipeline']}")
            return self.pipeline

        except Exception as e:
            logger.error(f"Failed to load PyAnnote pipeline: {e}")
            raise RuntimeError(f"PyAnnote pipeline loading failed: {str(e)}")

    @staticmethod
    def safe_fft(x, n=None):
        """Enhanced FFT wrapper with comprehensive empty data handling"""
        if x is None or len(x) == 0:
            logger.warning("Empty input to FFT, returning zero array")
            return np.zeros(n if n and n > 0 else 1, dtype=complex)

        # Ensure minimum length for FFT
        if len(x) < 2:
            logger.warning(f"Input too short for FFT: {len(x)} samples, padding to minimum")
            x = np.pad(x, (0, max(0, 2 - len(x))), mode='constant')

        try:
            return fft(x, n=n)
        except Exception as e:
            logger.error(f"FFT operation failed: {e}")
            return np.zeros(n if n and n > 0 else len(x), dtype=complex)

    def preprocess_audio_with_vad(self, audio_file):
        """Enhanced Voice Activity Detection with better error handling"""
        try:
            # Read audio file
            Fs, x = wavfile.read(audio_file)
            if len(x) == 0:
                raise ValueError("Empty audio file")

            # Convert to mono if stereo
            if x.ndim > 1:
                x = x.mean(axis=1)

            # Normalize audio
            if np.max(np.abs(x)) == 0:
                logger.warning("Silent audio detected, creating minimal segments")
                return [(0.0, min(1.0, len(x) / Fs))]

            # Ensure minimum duration for VAD
            min_duration = 0.5 # 500ms minimum
            if len(x) / Fs < min_duration:
                logger.warning(f"Audio too short for VAD: {len(x)/Fs:.2f}s, returning full duration")
                return [(0.0, len(x) / Fs)]

            # Perform voice activity detection with error handling
            try:
                if aS is not None:
                    vad_segments = aS.silence_removal(
                        x, Fs,
                        st_win=0.1, # Smaller window for short audio
                        st_step=0.05,
                        smooth_window=0.2,
                        weight=0.3 # Lower threshold for detection
                    )
                else:
                    # Fallback if aS is not available
                    logger.warning("aS not available, using full audio duration")
                    return [(0.0, len(x) / Fs)]

                # Validate VAD results
                if not vad_segments or len(vad_segments) == 0:
                    logger.warning("VAD found no speech segments, using full audio")
                    return [(0.0, len(x) / Fs)]

                # Filter out very short segments
                valid_segments = []
                for start, end in vad_segments:
                    if end - start >= 0.1: # Minimum 100ms segments
                        valid_segments.append((start, end))

                if not valid_segments:
                    logger.warning("All VAD segments too short, using full audio")
                    return [(0.0, len(x) / Fs)]

                return valid_segments

            except Exception as vad_error:
                logger.warning(f"VAD failed: {vad_error}, using full audio duration")
                return [(0.0, len(x) / Fs)]

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise ValueError(f"Cannot preprocess audio: {str(e)}")

    def _extract_frequency_features(self, audio_segment, sample_rate):
        """
        Extract frequency-based features for speaker identification

        Args:
            audio_segment: numpy array of audio samples
            sample_rate: sample rate of the audio

        Returns:
            list: [mean_f0, freq_ratio, mean_centroid, formant_f1, formant_f2]
        """
        try:
            # Ensure audio has minimum length
            if len(audio_segment) < sample_rate * 0.1: # 100ms minimum
                logger.warning("Audio segment too short for frequency analysis")
                return [0, 0, 0, 0, 0]

            # Normalize audio to prevent overflow
            if np.max(np.abs(audio_segment)) > 0:
                audio_segment = audio_segment / np.max(np.abs(audio_segment))

            # 1. Fundamental Frequency (F0) extraction
            try:
                # Use librosa.yin for pitch detection
                f0 = librosa.yin(audio_segment, fmin=50, fmax=500, sr=sample_rate)
                # Filter out unvoiced segments (f0 = 0)
                voiced_f0 = f0[f0 > 0]
                mean_f0 = np.mean(voiced_f0) if len(voiced_f0) > 0 else 150
                logger.debug(f"Mean F0: {mean_f0:.2f} Hz")
            except Exception as e:
                logger.warning(f"F0 extraction failed: {e}")
                mean_f0 = 150 # Default middle value

            # 2. Spectral analysis
            try:
                # Short-time Fourier Transform
                stft = librosa.stft(audio_segment, n_fft=2048, hop_length=512)
                magnitude = np.abs(stft)

                # Frequency bins
                freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

                # Low/High frequency energy ratio (key differentiator)
                low_freq_mask = freqs < 1000 # Below 1kHz
                high_freq_mask = freqs > 1000 # Above 1kHz

                low_energy = np.sum(magnitude[low_freq_mask])
                high_energy = np.sum(magnitude[high_freq_mask])

                # Frequency ratio (higher for male voices, lower for female)
                freq_ratio = low_energy / (high_energy + 1e-8)
                logger.debug(f"Frequency ratio (low/high): {freq_ratio:.3f}")

            except Exception as e:
                logger.warning(f"Spectral analysis failed: {e}")
                freq_ratio = 1.0 # Default neutral ratio

            # 3. Spectral centroid (brightness of sound)
            try:
                centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)[0]
                mean_centroid = np.mean(centroid)
                logger.debug(f"Mean spectral centroid: {mean_centroid:.2f} Hz")
            except Exception as e:
                logger.warning(f"Spectral centroid calculation failed: {e}")
                mean_centroid = 2000 # Default value

            # 4. Formant analysis (F1 and F2)
            try:
                # Use LPC (Linear Predictive Coding) for formant estimation
                # Get power spectral density
                freqs_psd, psd = signal.welch(audio_segment, fs=sample_rate, nperseg=1024)

                # Find peaks in the spectrum (formants)
                peaks, _ = signal.find_peaks(psd, height=np.max(psd) * 0.1, distance=20)

                if len(peaks) >= 2:
                    formant_f1 = freqs_psd[peaks[0]] # First formant
                    formant_f2 = freqs_psd[peaks[1]] # Second formant
                else:
                    # Default formant values
                    formant_f1 = 500 # Typical F1 for /a/ vowel
                    formant_f2 = 1500 # Typical F2 for /a/ vowel

                logger.debug(f"Formants - F1: {formant_f1:.2f} Hz, F2: {formant_f2:.2f} Hz")

            except Exception as e:
                logger.warning(f"Formant analysis failed: {e}")
                formant_f1, formant_f2 = 500, 1500 # Default values

            features = [mean_f0, freq_ratio, mean_centroid, formant_f1, formant_f2]
            logger.debug(f"Extracted features: {features}")
            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return [150, 1.0, 2000, 500, 1500] # Default feature values

    def _frequency_based_diarization(self, audio_file, min_speakers=2, max_speakers=5):
        """
        Speaker diarization based on frequency characteristics

        Args:
            audio_file: path to audio file
            min_speakers: minimum number of speakers to detect
            max_speakers: maximum number of speakers to detect

        Returns:
            list: diarization segments with speaker labels
        """
        try:
            logger.info(f"Starting frequency-based diarization with {min_speakers}-{max_speakers} speakers")

            # Read audio file using librosa
            try:
                audio_data, sample_rate = load_audio_librosa(audio_file, sr=None, mono=True)
                logger.info(f"Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")
            except Exception as e:
                logger.error(f"Failed to load audio: {e}")
                return []

            if len(audio_data) < sample_rate * 0.5: # Less than 500ms
                logger.warning("Audio too short for reliable diarization")
                return [{
                    "start": 0.0,
                    "end": len(audio_data) / sample_rate,
                    "speaker": "Speaker_1"
                }]

            # Segment audio into analysis windows
            window_duration = 1.0 # 1 second windows
            hop_duration = 0.5 # 500ms hop (50% overlap)
            window_samples = int(window_duration * sample_rate)
            hop_samples = int(hop_duration * sample_rate)

            segments = []
            features = []

            # Extract features for each window
            for start_sample in range(0, len(audio_data) - window_samples, hop_samples):
                end_sample = start_sample + window_samples
                segment_audio = audio_data[start_sample:end_sample]

                # Extract frequency features
                segment_features = self._extract_frequency_features(segment_audio, sample_rate)

                # Create segment info
                segment = {
                    "start": start_sample / sample_rate,
                    "end": end_sample / sample_rate,
                    "audio": segment_audio,
                    "features": segment_features
                }

                segments.append(segment)
                features.append(segment_features)

            if not features:
                logger.error("No features extracted from audio")
                return []

            # Normalize features for clustering
            features_array = np.array(features)

            # Handle edge case where all features are identical
            if np.std(features_array) == 0:
                logger.warning("All audio segments have identical features")
                return [{
                    "start": 0.0,
                    "end": len(audio_data) / sample_rate,
                    "speaker": "Speaker_1"
                }]

            # Check if scikit-learn is available
            if not SKLEARN_AVAILABLE or StandardScaler is None:
                logger.warning("scikit-learn not available, using simple energy-based diarization")
                return self._simple_energy_diarization(audio_file, max_speakers)

            # Normalize features
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_array)

            # Apply PCA for dimensionality reduction and noise reduction
            if PCA is not None:
                pca = PCA(n_components=min(3, features_normalized.shape[1]))
                features_pca = pca.fit_transform(features_normalized)
            else:
                # Fallback: use raw normalized features if PCA is not available
                features_pca = features_normalized

            logger.info(f"Feature extraction complete: {len(features)} segments, {features_pca.shape[1]} PCA components")

            # Try different numbers of clusters and select best
            best_score = -1
            best_labels = None
            best_n_clusters = min_speakers

            for n_clusters in range(min_speakers, min(max_speakers + 1, len(features) + 1)):
                try:
                    # Fix: Check if KMeans is not None before calling
                    if KMeans is None:
                        logger.warning("KMeans not available, skipping clustering")
                        break

                    # K-means clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features_pca)

                    # Calculate silhouette score for cluster quality
                    if len(set(labels)) > 1 and silhouette_score is not None: # Fix: Check if silhouette_score is not None
                        score = silhouette_score(features_pca, labels)
                        logger.debug(f"K-means with {n_clusters} clusters: silhouette score = {score:.3f}")

                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_n_clusters = n_clusters

                except Exception as e:
                    logger.warning(f"Clustering with {n_clusters} clusters failed: {e}")
                    continue

            if best_labels is None:
                logger.warning("All clustering attempts failed, using single speaker")
                best_labels = np.zeros(len(segments), dtype=int)
                best_n_clusters = 1

            logger.info(f"Best clustering: {best_n_clusters} speakers, silhouette score: {best_score:.3f}")

            # Apply temporal smoothing to reduce rapid speaker changes
            smoothed_labels = self._temporal_smoothing(best_labels, segments)

            # Convert segments to diarization format
            diarization_segments = []
            for i, (segment, speaker_id) in enumerate(zip(segments, smoothed_labels)):
                speaker_label = f"Speaker_{speaker_id + 1}"
                diarization_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": speaker_label,
                    "confidence": 0.8, # Fixed confidence for frequency-based method
                    "method": "frequency_analysis"
                }

                diarization_segments.append(diarization_segment)

            # Merge consecutive segments from same speaker
            merged_segments = self._merge_consecutive_segments(diarization_segments)

            logger.info(f"Frequency-based diarization complete: {len(merged_segments)} segments, {len(set(smoothed_labels))} unique speakers")
            return merged_segments

        except Exception as e:
            logger.error(f"Frequency-based diarization failed: {e}")
            return []

    def _temporal_smoothing(self, labels, segments, min_segment_duration=0.5):
        """Apply temporal smoothing to reduce rapid speaker changes"""
        try:
            smoothed = labels.copy()

            # Remove very short segments
            for i in range(1, len(labels) - 1):
                segment_duration = segments[i]["end"] - segments[i]["start"]
                if segment_duration < min_segment_duration:
                    # If segment is too short, assign to majority neighbor
                    if labels[i-1] == labels[i+1]:
                        smoothed[i] = labels[i-1]

            # Apply median filter for additional smoothing
            smoothed = median_filter(smoothed.astype(float), size=3).astype(int)

            return smoothed

        except Exception as e:
            logger.warning(f"Temporal smoothing failed: {e}")
            return labels

    def _merge_consecutive_segments(self, segments, max_gap=0.1):
        """Merge consecutive segments from the same speaker"""
        if not segments:
            return segments

        merged = []
        current = segments[0].copy()

        for next_segment in segments[1:]:
            # If same speaker and small gap, merge
            if (next_segment["speaker"] == current["speaker"] and
                next_segment["start"] - current["end"] <= max_gap):
                current["end"] = next_segment["end"]
            else:
                merged.append(current)
                current = next_segment.copy()

        merged.append(current) # Don't forget the last segment
        return merged

    async def diarize_audio(self, audio_file, number_speakers=0):
        """Speaker diarization using PyAudioAnalysis (free alternative to PyAnnote)"""
        try:
            logger.info(f"Starting PyAudioAnalysis diarization for {audio_file}")

            # Validate input file
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            if os.path.getsize(audio_file) == 0:
                raise ValueError("Audio file is empty")

            # Check if PyAudioAnalysis is available
            if not PYAUDIO_ANALYSIS_AVAILABLE or aS is None:
                raise RuntimeError("PyAudioAnalysis not available")

            # Run diarization using PyAudioAnalysis
            loop = asyncio.get_event_loop()
            
            # Set number of speakers (default to 2 if not specified)
            n_speakers = number_speakers if number_speakers > 0 else 2
            
            # Run speaker diarization in executor to avoid blocking
            cls = await loop.run_in_executor(
                None,
                aS.speaker_diarization,
                audio_file,
                n_speakers,
                'svm',  # Use SVM for classification
                True,   # Apply VAD preprocessing
                0.5     # LDA dimensionality reduction
            )

            # Convert PyAudioAnalysis format to TranscrevAI format
            segments = []
            if cls is not None and len(cls) > 0:
                # Get audio duration to calculate segment timings
                duration = self.get_audio_duration(audio_file)
                segment_duration = duration / len(cls)
                
                current_speaker = None
                segment_start = 0.0
                
                for i, speaker_id in enumerate(cls):
                    segment_time = i * segment_duration
                    segment_end = (i + 1) * segment_duration
                    
                    # Group consecutive segments with same speaker
                    if current_speaker != speaker_id:
                        # Close previous segment
                        if current_speaker is not None:
                            segments.append({
                                "start": segment_start,
                                "end": segment_time,
                                "speaker": f"Speaker_{current_speaker + 1}",
                                "confidence": 0.85,
                                "method": "pyaudioanalysis"
                            })
                        
                        # Start new segment
                        current_speaker = speaker_id
                        segment_start = segment_time
                
                # Close final segment
                if current_speaker is not None:
                    segments.append({
                        "start": segment_start,
                        "end": duration,
                        "speaker": f"Speaker_{current_speaker + 1}",
                        "confidence": 0.85,
                        "method": "pyaudioanalysis"
                    })
            
            if not segments:
                # Fallback to single speaker if no segments found
                duration = self.get_audio_duration(audio_file)
                segments = [{
                    "start": 0.0,
                    "end": duration,
                    "speaker": "Speaker_1",
                    "confidence": 0.5,
                    "method": "fallback_single"
                }]

            # Sort segments by start time
            segments.sort(key=lambda x: x["start"])

            unique_speakers = len(set(seg["speaker"] for seg in segments))
            logger.info(f"PyAudioAnalysis diarization completed: {len(segments)} segments, {unique_speakers} speakers")

            return segments

        except Exception as e:
            logger.error(f"PyAudioAnalysis diarization failed: {e}")
            # Fallback to simple single speaker
            try:
                duration = self.get_audio_duration(audio_file)
                return [{
                    "start": 0.0,
                    "end": duration,
                    "speaker": "Speaker_1",
                    "confidence": 0.5,
                    "method": "fallback"
                }]
            except:
                return [{
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "Speaker_1",
                    "confidence": 0.5,
                    "method": "fallback"
                }]

    def diarize(self, audio_file, number_speakers=0, vad_segments=None):
        """Enhanced internal diarization with FFT error fixes"""
        # Fix: Initialize variables at the start to avoid unbound issues
        x = None
        Fs = None

        try:
            logger.info(f"Processing diarization on {audio_file}")

            # Read and validate audio
            try:
                Fs, x = wavfile.read(audio_file)
                if len(x) == 0:
                    raise ValueError("Empty audio data")
            except Exception as e:
                logger.error(f"Failed to read audio file: {e}")
                return [{
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "Speaker_1"
                }]

            # Convert to mono
            if x.ndim > 1:
                x = x.mean(axis=1)

            # Extract speech segments based on VAD
            speech_samples = []
            if vad_segments is None:
                logger.warning("vad_segments is None, using full audio as a single segment")
                vad_segments = [(0.0, len(x) / Fs)]

            for start, end in vad_segments:
                start_sample = max(0, int(start * Fs))
                end_sample = min(len(x), int(end * Fs))
                if end_sample > start_sample:
                    segment = x[start_sample:end_sample]
                    if len(segment) > 0:
                        speech_samples.append(segment)

            if not speech_samples:
                logger.warning("No valid speech segments found, using full audio")
                speech_samples = [x]

            # Concatenate speech samples
            x_filt = np.concatenate(speech_samples)

            # Ensure minimum length
            min_samples = int(Fs * 0.5) # 500ms minimum
            if len(x_filt) < min_samples:
                logger.warning(f"Audio too short: {len(x_filt)/Fs:.2f}s, padding to minimum")
                x_filt = np.pad(x_filt, (0, min_samples - len(x_filt)), mode='constant')

            # Resample if necessary
            if Fs != 16000:
                x_filt = librosa.resample(x_filt.astype(np.float32), orig_sr=Fs, target_sr=16000)
                Fs = 16000

            # Normalize
            if np.max(np.abs(x_filt)) > 0:
                if np.issubdtype(x_filt.dtype, np.integer):
                    x_filt = x_filt.astype(np.float32) / 32768
                else:
                    x_filt = x_filt.astype(np.float32)
                if np.max(np.abs(x_filt)) > 1.0:
                    x_filt = x_filt / np.max(np.abs(x_filt))
            else:
                logger.warning("Silent audio after filtering, creating minimal noise")
                x_filt = np.random.normal(0, 0.001, len(x_filt)).astype(np.float32)

            # Save processed audio with unique filename to avoid conflicts
            processed_dir = FileManager.get_data_path("processed")
            unique_filename = f"vad_processed_{int(time.time()*1000)}_{os.getpid()}.wav"
            vad_proc_wav = os.path.join(processed_dir, unique_filename)
            wavfile.write(vad_proc_wav, Fs, (x_filt * 32768).astype(np.int16))

            # Extract features with safe FFT and better validation
            try:
                # Ensure we have sufficient audio length for feature extraction
                min_audio_length = 2.0 # 2 seconds minimum
                if len(x_filt) / Fs < min_audio_length:
                    logger.warning(f"Audio too short for proper diarization: {len(x_filt)/Fs:.2f}s")
                    # Return single speaker for short audio
                    return [{
                        "start": 0.0,
                        "end": len(x_filt) / Fs,
                        "speaker": "Speaker_1"
                    }]

                if mid_feature_extraction is not None:
                    with patch("scipy.fftpack.fft", new=self.safe_fft):
                        features = mid_feature_extraction(
                            signal=x_filt,
                            sampling_rate=Fs,
                            mid_window=min(1.0, len(x_filt) / Fs / 4), # Adaptive window
                            mid_step=0.5,
                            short_window=0.05,
                            short_step=0.05
                        )
                else:
                    logger.warning("mid_feature_extraction not available, using fallback")
                    raise ValueError("Feature extraction not available")

                if not features or len(features) < 2:
                    raise ValueError("Feature extraction failed - insufficient data")

                mid_term_features = features[0]
                if mid_term_features.shape[1] < 2:
                    raise ValueError("Insufficient features for diarization")

            except Exception as feature_error:
                logger.warning(f"Feature extraction failed ({feature_error}), falling back to single speaker")
                # Return single speaker fallback
                return [{
                    "start": 0.0,
                    "end": len(x_filt) / Fs,
                    "speaker": "Speaker_1"
                }]

            # Perform speaker diarization with enhanced error handling and fallbacks
            try:
                if not PYAUDIO_ANALYSIS_AVAILABLE or aS is None:
                    raise ValueError("pyAudioAnalysis not available")

                result = aS.speaker_diarization(
                    vad_proc_wav,
                    n_speakers=max(1, number_speakers),
                    mid_window=2.0,
                    mid_step=0.1,
                    short_window=0.05,
                    lda_dim=min(35, mid_term_features.shape[0] - 1)
                )

                if not isinstance(result, tuple) or len(result) < 3:
                    raise ValueError("Invalid diarization result")

                flags, classes, _ = result

                if not isinstance(flags, np.ndarray) or len(flags) < 1:
                    raise ValueError("Invalid speaker flags")

                logger.info("pyAudioAnalysis diarization successful")

            except Exception as diar_error:
                logger.warning(f"pyAudioAnalysis diarization failed: {diar_error}, trying alternative methods")

                # Try alternative diarization methods
                try:
                    # Try scikit-learn based diarization
                    alt_segments = self._alternative_diarization_sklearn(vad_proc_wav, max(2, number_speakers))
                    if alt_segments and len(alt_segments) > 0:
                        logger.info("Alternative scikit-learn diarization successful")
                        # Clean up temporary file
                        if os.path.exists(vad_proc_wav):
                            try:
                                os.remove(vad_proc_wav)
                            except Exception as cleanup_error:
                                logger.warning(f"Failed to clean up temporary file: {cleanup_error}")
                        return alt_segments
                    else:
                        raise ValueError("Alternative diarization returned no segments")

                except Exception as alt_error:
                    logger.warning(f"Alternative diarization failed: {alt_error}, using simple energy-based method")

                    # Final fallback: simple energy-based diarization
                    try:
                        energy_segments = self._simple_energy_diarization(vad_proc_wav, max(2, number_speakers))
                        if energy_segments and len(energy_segments) > 0:
                            logger.info("Simple energy diarization successful")
                            # Clean up temporary file
                            if os.path.exists(vad_proc_wav):
                                try:
                                    os.remove(vad_proc_wav)
                                except Exception as cleanup_error:
                                    logger.warning(f"Failed to clean up temporary file: {cleanup_error}")
                            return energy_segments

                    except Exception as energy_error:
                        logger.error(f"All diarization methods failed: {energy_error}")

                    # Ultimate fallback: single speaker
                    logger.warning("All diarization methods failed, returning single speaker")
                    return [{
                        "start": 0.0,
                        "end": len(x_filt) / Fs,
                        "speaker": "Speaker_1"
                    }]

            # Clean up temporary file
            if os.path.exists(vad_proc_wav):
                try:
                    os.remove(vad_proc_wav)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file: {cleanup_error}")

            # Convert results to segments
            return self._convert_flags_to_segments(flags, len(x_filt) / Fs)

        except Exception as e:
            logger.error(f"Diarization processing failed: {e}")
            # Return fallback result with proper handling of unbound variables
            try:
                # Fix: Check if variables are properly initialized
                if x is not None and Fs is not None:
                    duration = len(x) / Fs
                else:
                    duration = 1.0
            except:
                duration = 1.0

            return [{
                "start": 0.0,
                "end": duration,
                "speaker": "Speaker_1"
            }]

    def _convert_flags_to_segments(self, flags, total_duration):
        """Convert speaker flags to time segments with validation"""
        try:
            segments = []
            if len(flags) == 0:
                return [{
                    "start": 0.0,
                    "end": total_duration,
                    "speaker": "Speaker_1"
                }]

            current_speaker = int(flags[0])
            start_time = 0.0
            min_duration = 0.3 # Minimum segment duration

            for i in range(1, len(flags)):
                if flags[i] != current_speaker:
                    end_time = (i / len(flags)) * total_duration
                    duration = end_time - start_time

                    if duration >= min_duration:
                        segments.append({
                            "start": float(start_time),
                            "end": float(end_time),
                            "speaker": f"Speaker_{current_speaker + 1}"
                        })

                    start_time = end_time
                    current_speaker = int(flags[i])

            # Add final segment
            final_duration = total_duration - start_time
            if final_duration >= min_duration:
                segments.append({
                    "start": float(start_time),
                    "end": float(total_duration),
                    "speaker": f"Speaker_{current_speaker + 1}"
                })

            # Ensure at least one segment exists
            if not segments:
                segments = [{
                    "start": 0.0,
                    "end": total_duration,
                    "speaker": "Speaker_1"
                }]

            return segments

        except Exception as e:
            logger.error(f"Segment conversion failed: {e}")
            return [{
                "start": 0.0,
                "end": total_duration,
                "speaker": "Speaker_1"
            }]

    def get_audio_duration(self, file_path):
        """Get audio file duration with error handling"""
        try:
            Fs, x = wavfile.read(file_path)
            return len(x) / Fs
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return 1.0 # Default duration

    def _alternative_diarization_sklearn(self, audio_file, n_speakers=2):
        """Alternative diarization using scikit-learn and spectral features"""
        try:
            logger.info(f"Starting alternative diarization with scikit-learn for {audio_file}")

            if not SKLEARN_AVAILABLE:
                logger.warning("scikit-learn not available for alternative diarization")
                return self._simple_energy_diarization(audio_file, n_speakers)

            # Read audio
            Fs, x = wavfile.read(audio_file)
            if len(x) == 0:
                raise ValueError("Empty audio data")

            # Convert to mono
            if x.ndim > 1:
                x = x.mean(axis=1)

            # Normalize
            if np.max(np.abs(x)) > 0:
                x = x.astype(np.float32) / np.max(np.abs(x))

            # Extract MFCC-like features using librosa
            duration = len(x) / Fs
            window_size = 2.0 # 2 second windows
            hop_size = 0.5 # 0.5 second hop

            features = []
            timestamps = []

            window_samples = int(window_size * Fs)
            hop_samples = int(hop_size * Fs)

            for start_sample in range(0, len(x) - window_samples + 1, hop_samples):
                end_sample = start_sample + window_samples
                segment = x[start_sample:end_sample]

                if len(segment) < window_samples:
                    segment = np.pad(segment, (0, window_samples - len(segment)), mode='constant')

                # Extract spectral features
                feature_vector = self._extract_spectral_features(segment, Fs)
                features.append(feature_vector)
                timestamps.append(start_sample / Fs)

            if len(features) < 2:
                logger.warning("Not enough features for clustering, using single speaker")
                return [{
                    "start": 0.0,
                    "end": duration,
                    "speaker": "Speaker_1"
                }]

            features = np.array(features)

            # Normalize features
            if StandardScaler is not None:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
            else:
                features_scaled = features

            # Determine optimal number of speakers
            if n_speakers <= 0:
                # Use simple heuristic: try 2-4 speakers and pick based on duration
                n_speakers = min(4, max(2, int(duration / 30))) # 1 speaker per 30 seconds

            n_speakers = min(n_speakers, len(features))

            # Perform clustering
            try:
                # Try KMeans first
                if KMeans is not None:
                    kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features_scaled)
                else:
                    raise Exception("KMeans not available")

            except Exception as e:
                logger.warning(f"KMeans failed: {e}, trying AgglomerativeClustering")
                try:
                    if AgglomerativeClustering is not None:
                        clustering = AgglomerativeClustering(n_clusters=n_speakers)
                        labels = clustering.fit_predict(features_scaled)
                    else:
                        raise Exception("AgglomerativeClustering not available")

                except Exception as e2:
                    logger.warning(f"AgglomerativeClustering failed: {e2}, using simple energy-based diarization")
                    return self._simple_energy_diarization(audio_file, n_speakers)

            # Convert labels to segments
            segments = []
            current_speaker = labels[0]
            start_time = timestamps[0]

            for i in range(1, len(labels)):
                if labels[i] != current_speaker or i == len(labels) - 1:
                    end_time = timestamps[i] if i < len(timestamps) else duration
                    segments.append({
                        "start": float(start_time),
                        "end": float(end_time),
                        "speaker": f"Speaker_{current_speaker + 1}"
                    })
                    start_time = timestamps[i] if i < len(timestamps) else duration
                    current_speaker = labels[i]

            # Add final segment if needed
            if segments and segments[-1]["end"] < duration:
                segments.append({
                    "start": float(segments[-1]["end"]),
                    "end": float(duration),
                    "speaker": f"Speaker_{current_speaker + 1}"
                })

            # Merge short segments
            segments = self._merge_short_segments(segments, min_duration=1.0)

            logger.info(f"Alternative diarization completed: {len(segments)} segments, {len(set(s['speaker'] for s in segments))} speakers")
            return segments

        except Exception as e:
            logger.error(f"Alternative diarization failed: {e}")
            return self._simple_energy_diarization(audio_file, n_speakers)

    def _extract_spectral_features(self, audio_segment, sample_rate):
        """Extract spectral features from audio segment"""
        try:
            # Compute FFT
            fft_result = np.fft.fft(audio_segment)
            magnitude = np.abs(fft_result[:len(fft_result)//2])

            # Spectral centroid
            freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)[:len(fft_result)//2]
            spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)

            # Spectral rolloff
            magnitude_sum = np.sum(magnitude)
            rolloff_threshold = 0.85 * magnitude_sum
            cumsum_magnitude = np.cumsum(magnitude)
            rolloff_idx = np.where(cumsum_magnitude >= rolloff_threshold)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

            # Spectral flux
            spectral_flux = np.sum(np.diff(magnitude))

            # MFCC-like features (simplified)
            mel_filters = self._create_mel_filterbank(sample_rate, len(magnitude))
            mfcc_features = np.dot(mel_filters, magnitude)
            mfcc_features = np.log(mfcc_features + 1e-10)

            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(audio_segment)) != 0)
            zcr = zero_crossings / len(audio_segment)

            # RMS energy
            rms_energy = np.sqrt(np.mean(audio_segment**2))

            # Combine features
            features = np.concatenate([
                [spectral_centroid, spectral_rolloff, spectral_flux, zcr, rms_energy],
                mfcc_features[:12] # First 12 MFCC coefficients
            ])

            return features

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}, using basic features")
            # Basic fallback features
            return np.array([
                np.mean(audio_segment),
                np.std(audio_segment),
                np.max(audio_segment),
                np.min(audio_segment),
                np.sqrt(np.mean(audio_segment**2)) # RMS
            ])

    def _create_mel_filterbank(self, sample_rate, n_fft_bins, n_filters=13):
        """Create a simple mel filterbank"""
        try:
            # Mel scale conversion
            def hz_to_mel(hz):
                return 2595 * np.log10(1 + hz / 700)

            def mel_to_hz(mel):
                return 700 * (10**(mel / 2595) - 1)

            # Frequency range
            low_freq_mel = hz_to_mel(0)
            high_freq_mel = hz_to_mel(sample_rate / 2)

            # Mel points
            mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
            hz_points = mel_to_hz(mel_points)

            # Bin points
            bin_points = np.floor((n_fft_bins + 1) * hz_points / sample_rate).astype(int)

            # Create filterbank
            filterbank = np.zeros((n_filters, n_fft_bins))

            for i in range(1, n_filters + 1):
                left = bin_points[i - 1]
                center = bin_points[i]
                right = bin_points[i + 1]

                for j in range(left, center):
                    filterbank[i - 1, j] = (j - left) / (center - left)
                for j in range(center, right):
                    filterbank[i - 1, j] = (right - j) / (right - center)

            return filterbank

        except Exception as e:
            logger.warning(f"Mel filterbank creation failed: {e}")
            # Return identity-like matrix as fallback
            return np.eye(min(n_filters, n_fft_bins))[:n_filters]

    def _simple_energy_diarization(self, audio_file, n_speakers=2):
        """Simple energy-based diarization as final fallback"""
        try:
            logger.info(f"Using simple energy-based diarization for {audio_file}")

            # Read audio
            Fs, x = wavfile.read(audio_file)
            if len(x) == 0:
                raise ValueError("Empty audio data")

            # Convert to mono
            if x.ndim > 1:
                x = x.mean(axis=1)

            duration = len(x) / Fs
            window_size = 1.0 # 1 second windows
            window_samples = int(window_size * Fs)

            # Calculate energy for each window
            energies = []
            timestamps = []

            for start in range(0, len(x), window_samples):
                end = min(start + window_samples, len(x))
                segment = x[start:end]
                energy = np.mean(segment**2)
                energies.append(energy)
                timestamps.append(start / Fs)

            if len(energies) < 2:
                return [{
                    "start": 0.0,
                    "end": duration,
                    "speaker": "Speaker_1"
                }]

            # Simple threshold-based speaker change detection
            energies = np.array(energies)
            mean_energy = np.mean(energies)

            # Detect speaker changes based on energy variations
            segments = []
            current_speaker = 0
            start_time = 0.0

            for i, timestamp in enumerate(timestamps):
                # Simple heuristic: high energy = speaker 1, low energy = speaker 2
                speaker = 0 if energies[i] > mean_energy else 1

                if speaker != current_speaker or i == len(timestamps) - 1:
                    end_time = timestamp if i < len(timestamps) - 1 else duration
                    segments.append({
                        "start": float(start_time),
                        "end": float(end_time),
                        "speaker": f"Speaker_{current_speaker + 1}"
                    })
                    start_time = timestamp
                    current_speaker = speaker

            # Ensure we have segments
            if not segments:
                segments = [{
                    "start": 0.0,
                    "end": duration,
                    "speaker": "Speaker_1"
                }]

            logger.info(f"Simple energy diarization completed: {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Simple energy diarization failed: {e}")
            # Ultimate fallback
            duration = self.get_audio_duration(audio_file)
            return [{
                "start": 0.0,
                "end": duration,
                "speaker": "Speaker_1"
            }]

    def _merge_short_segments(self, segments, min_duration=1.0):
        """Merge segments that are too short"""
        if not segments:
            return segments

        merged = []
        i = 0

        while i < len(segments):
            current = segments[i]
            duration = current["end"] - current["start"]

            if duration < min_duration and i < len(segments) - 1:
                # Merge with next segment
                next_segment = segments[i + 1]
                merged_segment = {
                    "start": current["start"],
                    "end": next_segment["end"],
                    "speaker": current["speaker"] # Keep first speaker
                }

                merged.append(merged_segment)
                i += 2 # Skip next segment as it's been merged
            else:
                merged.append(current)
                i += 1

        return merged