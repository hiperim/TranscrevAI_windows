import asyncio
import numpy as np
import shutil
import warnings
from enum import Enum

# Suppress sklearn version compatibility warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# Suppress PyAudioAnalysis runtime warnings (divide by zero in audioSegmentation)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyAudioAnalysis')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='divide by zero encountered')
# Specifically suppress InconsistentVersionWarning from sklearn
try:
    from sklearn.exceptions import InconsistentVersionWarning  # type: ignore
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except (ImportError, AttributeError):
    # Older sklearn versions don't have this warning, or it might be in a different module
    try:
        # Try alternative import location
        from sklearn.utils import InconsistentVersionWarning  # type: ignore
        warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
    except (ImportError, AttributeError):
        pass  # Warning class not available in this sklearn version

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

# Lazy import globals for heavy dependencies
from typing import Any, Optional
_scipy_imports = {}
_scipy_fft: Optional[Any] = None
_scipy_signal: Optional[Any] = None
_scipy_median_filter: Optional[Any] = None
_scipy_wavfile: Optional[Any] = None
_librosa: Optional[Any] = None
_static_ffmpeg: Optional[Any] = None
_scipy_imports_attempted = False
_librosa_imports_attempted = False
_static_ffmpeg_attempted = False

def _ensure_scipy_imports():
    """Lazy import of scipy dependencies"""
    global _scipy_fft, _scipy_signal, _scipy_median_filter, _scipy_wavfile, _scipy_imports_attempted
    
    if _scipy_imports_attempted:
        return all(v is not None for v in [_scipy_fft, _scipy_signal, _scipy_median_filter, _scipy_wavfile])
    
    _scipy_imports_attempted = True
    
    try:
        from scipy.fftpack import fft
        from scipy import signal
        from scipy.ndimage import median_filter
        from scipy.io import wavfile
        
        _scipy_fft = fft
        _scipy_signal = signal
        _scipy_median_filter = median_filter
        _scipy_wavfile = wavfile
        
        # Store in global dict for legacy compatibility
        _scipy_imports['wavfile'] = wavfile
        _scipy_imports['signal'] = signal
        _scipy_imports['median_filter'] = median_filter
        
        logger.info("Scipy dependencies loaded successfully")
        return True
    except ImportError as e:
        logger.warning(f"Scipy dependencies not available: {e}")
        return False

def _ensure_librosa_imports():
    """Lazy import of librosa"""
    global _librosa, _librosa_imports_attempted
    
    if _librosa_imports_attempted:
        return _librosa is not None
    
    _librosa_imports_attempted = True
    
    try:
        import librosa as _lib
        _librosa = _lib
        logger.info("Librosa loaded successfully")
        return True
    except ImportError as e:
        logger.warning(f"Librosa not available: {e}")
        return False

def _ensure_static_ffmpeg():
    """Simplified FFmpeg setup for Docker environment"""
    global _static_ffmpeg, _static_ffmpeg_attempted
    
    if _static_ffmpeg_attempted:
        return _static_ffmpeg is not None
    
    _static_ffmpeg_attempted = True
    
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        _static_ffmpeg = True
        logger.info("static_ffmpeg configured successfully")
        return True
    except ImportError:
        # FFmpeg should be available system-wide in Docker
        _static_ffmpeg = True  # Assume available
        logger.info("Using system FFmpeg")
        return True

def safe_speaker_id_conversion(speaker_id):
    """
    CRITICAL FIX: Safely convert speaker ID to integer for arithmetic operations
    
    This function handles all the edge cases where PyAudioAnalysis or other libraries
    return speaker IDs as strings, floats, or other types that cause arithmetic errors.
    
    Args:
        speaker_id: Speaker ID (can be string, int, float, numpy types, etc.)
        
    Returns:
        int: Speaker ID as integer, defaults to 0 if conversion fails
    """
    try:
        if speaker_id is None:
            return 0
        elif isinstance(speaker_id, str):
            # Handle common string patterns like "Speaker_1", "1", "speaker_0", etc.
            if speaker_id.lower().startswith("speaker_"):
                # Extract number from "Speaker_1" format
                return int(speaker_id.split("_")[-1])
            else:
                # Try direct conversion
                return int(float(speaker_id))  # float first handles "1.0" strings
        elif isinstance(speaker_id, (int, np.integer)):
            return int(speaker_id)
        elif isinstance(speaker_id, (float, np.floating)):
            return int(round(speaker_id))
        elif isinstance(speaker_id, np.ndarray):
            # CRITICAL FIX: Handle numpy arrays properly - OPTIMIZED FOR HOT PATH
            if speaker_id.size == 1:
                return int(speaker_id.item())  # Extract scalar value
            else:
                # Remove logging from hot path for performance (as per fixes.txt)
                return int(speaker_id.flatten()[0])
        elif isinstance(speaker_id, (list, tuple)):
            # CRITICAL FIX: Handle lists and tuples - OPTIMIZED FOR HOT PATH
            if len(speaker_id) == 1:
                return safe_speaker_id_conversion(speaker_id[0])  # Recursive call
            else:
                # Remove logging from hot path for performance
                return safe_speaker_id_conversion(speaker_id[0])
        else:
            logger.warning(f"Unexpected speaker_id type: {type(speaker_id)}, value: {speaker_id}")
            try:
                return int(speaker_id)  # Last resort conversion
            except (ValueError, TypeError, AttributeError):
                logger.error(f"Failed final conversion of speaker_id '{speaker_id}', defaulting to 0")
                return 0
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Failed to convert speaker_id '{speaker_id}' to int: {e}, using default 0")
        return 0

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
    if not _ensure_librosa_imports() or _librosa is None:
        raise RuntimeError("Librosa not available for audio loading")
    return _librosa.load(audio_file, sr=sr, mono=mono)

class DiarizationError(Enum):
    FILE_NOT_FOUND = 1
    INVALID_FORMAT = 2
    EMPTY_AUDIO = 3
    INSUFFICIENT_DATA = 4

class SpeakerDiarization:
    """PyAudioAnalysis-based speaker diarization with comprehensive type safety and overlapping audio enhancement"""

    def __init__(self):
        self._device = None # Lazy device detection
        self._ffmpeg_path = None # Lazy FFmpeg setup

    @property
    def device(self):
        """Lazy device detection only when needed"""
        if self._device is None:
            self._device = "cpu"  # Default to CPU since we're not using PyAnnote
            logger.info(f"Diarization device: {self._device}")
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
            if not _ensure_scipy_imports() or _scipy_fft is None:
                raise RuntimeError("Scipy FFT not available")
            return _scipy_fft(x, n=n)
        except Exception as e:
            logger.error(f"FFT operation failed: {e}")
            return np.zeros(n if n and n > 0 else len(x), dtype=complex)

    def analyze_audio_for_speaker_count(self, audio_file, transcription_data=None):
        """
        Analyze audio to intelligently determine the likely number of speakers
        
        Args:
            audio_file: Path to audio file
            transcription_data: Optional transcription data for hint analysis
            
        Returns:
            int: Estimated number of speakers (1 if mono-speaker detected, 2+ otherwise)
        """
        try:
            logger.info(f"Analyzing audio characteristics to estimate speaker count: {audio_file}")
            
            # Check transcription hints first (fast path)
            hint_result = self.analyze_with_transcription_hints(audio_file, transcription_data)
            if hint_result > 0:
                return hint_result
            
            # Read audio file
            if not _ensure_scipy_imports():
                logger.warning("Scipy not available, defaulting to 1 speaker")
                return 1
            if _scipy_wavfile is None:
                logger.warning("Scipy wavfile not available, defaulting to 1 speaker")
                return 1
                
            Fs, x = _scipy_wavfile.read(audio_file)
            if len(x) == 0:
                logger.warning("Empty audio file, defaulting to 1 speaker")
                return 1

            # Convert to mono if stereo
            if x.ndim > 1:
                x = x.mean(axis=1)

            # Normalize audio
            if np.max(np.abs(x)) == 0:
                logger.warning("Silent audio detected, defaulting to 1 speaker")
                return 1
            
            x = x.astype(np.float32) / np.max(np.abs(x))
            
            duration = len(x) / Fs
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Short audio is likely single speaker  
            if duration < 10.0:  # Less than 10 seconds
                logger.info("Short audio detected (<10s), likely single speaker")
                return 1
            
            # Analyze audio in segments to detect speaker changes
            window_size = int(2.0 * Fs)  # 2-second windows
            hop_size = int(1.0 * Fs)     # 1-second hop
            
            segment_features = []
            
            for start in range(0, len(x) - window_size, hop_size):
                segment = x[start:start + window_size]
                
                # Calculate features for this segment
                features = self._calculate_segment_features(segment, Fs)
                segment_features.append(features)
                
                if len(segment_features) >= 10:  # Limit analysis for performance
                    break
            
            if len(segment_features) < 2:
                logger.info("Insufficient segments for analysis, defaulting to 1 speaker")
                return 1
            
            # Analyze feature variation to detect speaker changes
            features_array = np.array(segment_features)
            
            # Calculate coefficient of variation for each feature
            feature_variations = []
            for i in range(features_array.shape[1]):
                feature_col = features_array[:, i]
                if np.std(feature_col) == 0:
                    cv = 0
                else:
                    cv = np.std(feature_col) / (np.mean(np.abs(feature_col)) + 1e-10)
                feature_variations.append(cv)
            
            # Average coefficient of variation across all features
            avg_variation = np.mean(feature_variations)
            
            logger.info(f"Average feature variation: {avg_variation:.3f}")
            
            # Thresholds for speaker detection - adjusted to be less conservative
            SINGLE_SPEAKER_THRESHOLD = 0.15   # Below this = likely single speaker (lowered from 0.4)
            CLEAR_MULTI_SPEAKER_THRESHOLD = 0.3  # Above this = likely multiple speakers (lowered from 0.6)
            
            if avg_variation < SINGLE_SPEAKER_THRESHOLD:
                logger.info(f"Very low variation ({avg_variation:.3f}) detected - likely single speaker")
                return 1
            elif avg_variation > CLEAR_MULTI_SPEAKER_THRESHOLD:
                logger.info(f"High variation ({avg_variation:.3f}) detected - likely multiple speakers")
                # Use duration to estimate number of speakers
                estimated_speakers = min(4, max(2, int(duration / 30) + 2))  # 1 speaker per 30 seconds + base 2
                return estimated_speakers
            else:
                logger.info(f"Moderate variation ({avg_variation:.3f}) detected - defaulting to 2 speakers for conversation")
                return 2
                
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}, defaulting to 1 speaker")
            return 1

    def analyze_with_transcription_hints(self, audio_file, transcription_data=None):
        """Use transcription data to inform speaker detection"""
        try:
            if transcription_data:
                # Count sentences/phrases
                total_text = " ".join([seg.get('text', '') for seg in transcription_data])
                sentence_count = len([s for s in total_text.split('.') if s.strip()])
                
                if sentence_count <= 2:  # Very brief speech
                    logger.info("Brief transcription detected, likely single speaker")
                    return 1
                    
                # Check for conversation markers
                conversation_markers = ['hello', 'hi', 'yes', 'no', 'okay', 'right', 'sure', 'well']
                if not any(marker in total_text.lower() for marker in conversation_markers):
                    logger.info("No conversation markers, likely monologue")
                    return 1
            
            return 0  # Continue with normal analysis
        except Exception as e:
            logger.warning(f"Transcription hint analysis failed: {e}")
            return 0
    
    def _calculate_segment_features(self, segment, sample_rate):
        """Calculate acoustic features for a segment to help detect speaker changes"""
        try:
            # 1. RMS Energy
            rms_energy = np.sqrt(np.mean(segment**2))
            
            # 2. Zero Crossing Rate
            zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)
            zcr = zero_crossings / len(segment)
            
            # 3. Spectral Centroid (simplified)
            fft_result = np.fft.fft(segment)
            magnitude = np.abs(fft_result[:len(fft_result)//2])
            freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)[:len(fft_result)//2]
            
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                spectral_centroid = 0
            
            # 4. Fundamental Frequency Estimation (simplified)
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation for pitch
            if len(autocorr) > 100:
                peak_idx = np.argmax(autocorr[50:min(400, len(autocorr))]) + 50
                fundamental_freq = sample_rate / peak_idx if peak_idx > 0 else 0
            else:
                fundamental_freq = 0
            
            return [rms_energy, zcr, spectral_centroid, fundamental_freq]
            
        except Exception as e:
            logger.warning(f"Feature calculation failed: {e}")
            return [0, 0, 0, 0]  # Return default features

    def detect_overlapping_speakers(self, audio_file, segments):
        """
        Enhanced method to detect and handle overlapping speech from multiple speakers
        """
        try:
            if not _ensure_scipy_imports() or _scipy_wavfile is None:
                logger.warning("Scipy not available for overlap detection")
                return segments
                
            logger.info("Analyzing for overlapping speaker segments...")
            
            # Load audio for analysis
            Fs, x = _scipy_wavfile.read(audio_file)
            if x.ndim > 1:
                x = x.mean(axis=1)
            x = x.astype(np.float32) / (np.max(np.abs(x)) + 1e-10)
            
            enhanced_segments = []
            
            for i, segment in enumerate(segments):
                start_sample = int(segment['start'] * Fs)
                end_sample = int(segment['end'] * Fs)
                
                if end_sample > len(x):
                    end_sample = len(x)
                if start_sample >= end_sample:
                    continue
                    
                audio_segment = x[start_sample:end_sample]
                
                # Detect overlapping speech using energy distribution analysis
                overlap_detected = self._detect_energy_overlap(audio_segment, Fs)
                
                if overlap_detected:
                    # Split segment if overlap detected
                    sub_segments = self._split_overlapping_segment(
                        segment, audio_segment, Fs, start_sample, end_sample
                    )
                    enhanced_segments.extend(sub_segments)
                else:
                    enhanced_segments.append(segment)
            
            logger.info(f"Overlap detection: {len(segments)} -> {len(enhanced_segments)} segments")
            return enhanced_segments
            
        except Exception as e:
            logger.error(f"Overlapping speaker detection failed: {e}")
            return segments

    def _detect_energy_overlap(self, audio_segment, sample_rate):
        """Detect if audio segment contains overlapping speakers"""
        try:
            # Analyze energy in different frequency bands
            if len(audio_segment) < sample_rate * 0.5:  # Less than 500ms
                return False
                
            # Split into overlapping windows
            window_size = int(0.25 * sample_rate)  # 250ms windows
            hop_size = int(0.1 * sample_rate)      # 100ms hop
            
            energies_low = []   # 0-1kHz
            energies_mid = []   # 1-3kHz  
            energies_high = []  # 3-8kHz
            
            for start in range(0, len(audio_segment) - window_size, hop_size):
                window = audio_segment[start:start + window_size]
                
                # Apply window function
                windowed = window * np.hanning(len(window))
                
                # FFT analysis
                fft_result = np.fft.fft(windowed)
                magnitude = np.abs(fft_result[:len(fft_result)//2])
                
                # Frequency bins
                freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)[:len(magnitude)]
                
                # Energy in frequency bands
                low_band = magnitude[(freqs >= 0) & (freqs < 1000)]
                mid_band = magnitude[(freqs >= 1000) & (freqs < 3000)]
                high_band = magnitude[(freqs >= 3000) & (freqs < 8000)]
                
                energies_low.append(np.sum(low_band**2))
                energies_mid.append(np.sum(mid_band**2))
                energies_high.append(np.sum(high_band**2))
            
            if len(energies_low) < 3:
                return False
                
            # Analyze energy distribution patterns
            low_var = np.var(energies_low) if len(energies_low) > 1 else 0
            mid_var = np.var(energies_mid) if len(energies_mid) > 1 else 0
            high_var = np.var(energies_high) if len(energies_high) > 1 else 0
            
            # High variance in multiple bands suggests overlapping speakers
            variance_threshold = 0.1
            high_variance_bands = sum([
                low_var > variance_threshold,
                mid_var > variance_threshold, 
                high_var > variance_threshold
            ])
            
            # Also check for sudden energy spikes (characteristic of overlaps)
            total_energies = np.array(energies_low) + np.array(energies_mid) + np.array(energies_high)
            if len(total_energies) > 2:
                energy_spikes = np.diff(total_energies)
                spike_count = np.sum(np.abs(energy_spikes) > 2 * np.std(energy_spikes))
                
                # Overlap detected if multiple criteria met
                return high_variance_bands >= 2 or spike_count > len(total_energies) * 0.3
            
            return False
            
        except Exception as e:
            logger.warning(f"Energy overlap detection failed: {e}")
            return False

    def _split_overlapping_segment(self, original_segment, audio_segment, sample_rate, start_sample, end_sample):
        """Split a segment that contains overlapping speakers"""
        try:
            # Use energy-based splitting for overlapping segments
            window_size = int(0.5 * sample_rate)  # 500ms windows
            hop_size = int(0.25 * sample_rate)    # 250ms hop
            
            segment_energies = []
            segment_positions = []
            
            for start in range(0, len(audio_segment) - window_size, hop_size):
                window = audio_segment[start:start + window_size]
                energy = np.mean(window**2)
                segment_energies.append(energy)
                segment_positions.append(start)
            
            if len(segment_energies) < 2:
                return [original_segment]
                
            # Find energy valleys (potential speaker boundaries)
            energy_array = np.array(segment_energies)
            median_energy = np.median(energy_array)
            
            # Find positions where energy drops below median (likely speaker changes)
            valley_positions = []
            for i in range(1, len(segment_energies) - 1):
                if (segment_energies[i] < median_energy * 0.7 and 
                    segment_energies[i] < segment_energies[i-1] and 
                    segment_energies[i] < segment_energies[i+1]):
                    valley_positions.append(segment_positions[i])
            
            if not valley_positions:
                return [original_segment]
            
            # Create sub-segments
            sub_segments = []
            current_start = original_segment['start']
            
            for valley_pos in valley_positions:
                valley_time = original_segment['start'] + (valley_pos / sample_rate)
                
                # Ensure minimum segment duration
                if valley_time - current_start >= 0.5:  # At least 500ms
                    sub_segment = {
                        'start': current_start,
                        'end': valley_time,
                        'speaker': original_segment['speaker'],
                        'confidence': original_segment['confidence'] * 0.9  # Slightly lower confidence
                    }
                    sub_segments.append(sub_segment)
                    current_start = valley_time
            
            # Add final segment
            if original_segment['end'] - current_start >= 0.5:
                sub_segment = {
                    'start': current_start,
                    'end': original_segment['end'],
                    'speaker': original_segment['speaker'],
                    'confidence': original_segment['confidence'] * 0.9
                }
                sub_segments.append(sub_segment)
            
            return sub_segments if sub_segments else [original_segment]
            
        except Exception as e:
            logger.warning(f"Segment splitting failed: {e}")
            return [original_segment]

    async def diarize_audio(self, audio_file, transcription_data=None):
        """
        Enhanced diarization with overlapping speaker detection
        
        Args:
            audio_file: Path to the audio file
            transcription_data: Optional transcription segments for guidance
            
        Returns:
            List of diarization segments with speaker labels
        """
        try:
            logger.info(f"Starting enhanced diarization: {audio_file}")
            
            # Step 1: Intelligent speaker count estimation
            estimated_speakers = self.analyze_audio_for_speaker_count(audio_file, transcription_data)
            logger.info(f"Estimated speakers: {estimated_speakers}")
            
            # Step 2: PyAudioAnalysis-based segmentation
            segments = await self._perform_pyaudio_analysis_diarization(audio_file, estimated_speakers)
            
            # Step 3: Enhanced overlapping speaker detection and handling
            segments = self.detect_overlapping_speakers(audio_file, segments)
            
            # Step 4: Post-process segments
            segments = self._post_process_segments(segments)
            
            logger.info(f"Enhanced diarization completed: {len(segments)} segments, {len(set(seg['speaker'] for seg in segments))} unique speakers")
            return segments
            
        except Exception as e:
            logger.error(f"Enhanced diarization failed: {e}")
            return self._create_fallback_segments(audio_file, 1)

    async def _perform_pyaudio_analysis_diarization(self, audio_file, num_speakers):
        """Perform diarization using PyAudioAnalysis"""
        try:
            if not PYAUDIO_ANALYSIS_AVAILABLE or aS is None:
                logger.warning("PyAudioAnalysis not available, using simple fallback")
                return self._create_fallback_segments(audio_file, num_speakers)
            
            logger.info(f"Running PyAudioAnalysis diarization with {num_speakers} speakers")
            
            # Run speaker segmentation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._run_pyaudio_analysis_segmentation, audio_file, num_speakers
            )
            
            return result
            
        except Exception as e:
            logger.error(f"PyAudioAnalysis diarization failed: {e}")
            return self._create_fallback_segments(audio_file, num_speakers)

    def _run_pyaudio_analysis_segmentation(self, audio_file, num_speakers):
        """Run PyAudioAnalysis segmentation in executor"""
        try:
            # Use speaker diarization from PyAudioAnalysis
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if aS is not None:
                    speaker_labels, _, _ = aS.speaker_diarization(
                        audio_file, 
                        num_speakers, 
                        mid_window=0.5,  # 500ms step (correct parameter name)
                        short_window=0.1,  # 100ms short-term window (correct parameter name)
                        plot_res=False
                    )
                else:
                    speaker_labels = None
            
            if speaker_labels is None or len(speaker_labels) == 0:
                logger.warning("PyAudioAnalysis returned empty results")
                return self._create_fallback_segments(audio_file, num_speakers)
            
            # Convert results to segments
            segments = self._convert_pyaudio_results_to_segments(speaker_labels, audio_file)
            return segments
            
        except Exception as e:
            logger.error(f"PyAudioAnalysis execution failed: {e}")
            return self._create_fallback_segments(audio_file, num_speakers)

    def _convert_pyaudio_results_to_segments(self, speaker_labels, audio_file):
        """Convert PyAudioAnalysis results to segment format"""
        try:
            # Get audio duration
            if not _ensure_scipy_imports() or _scipy_wavfile is None:
                logger.warning("Cannot determine audio duration")
                duration = 10.0  # Default fallback
            else:
                try:
                    Fs, x = _scipy_wavfile.read(audio_file)
                    duration = len(x) / Fs if x.ndim == 1 else len(x) / Fs
                except Exception:
                    duration = 10.0
            
            segments = []
            current_speaker = None
            current_start = 0.0
            step_size = 0.5  # 500ms steps (matching mid_step from segmentation)
            
            for i, speaker_label in enumerate(speaker_labels):
                speaker_id = safe_speaker_id_conversion(speaker_label)
                current_time = i * step_size
                
                if current_speaker is None:
                    current_speaker = speaker_id
                    current_start = current_time
                elif speaker_id != current_speaker:
                    # Speaker change detected
                    if current_time > current_start:  # Ensure positive duration
                        segment = {
                            'start': current_start,
                            'end': current_time,
                            'speaker': current_speaker,
                            'confidence': 0.8  # Default confidence for PyAudioAnalysis
                        }
                        segments.append(segment)
                    
                    current_speaker = speaker_id
                    current_start = current_time
            
            # Add final segment
            if current_speaker is not None and duration > current_start:
                segment = {
                    'start': current_start,
                    'end': duration,
                    'speaker': current_speaker,
                    'confidence': 0.8
                }
                segments.append(segment)
            
            # Filter out very short segments (< 500ms)
            segments = [seg for seg in segments if seg['end'] - seg['start'] >= 0.5]
            
            logger.info(f"Converted PyAudioAnalysis results: {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to convert PyAudioAnalysis results: {e}")
            return []

    def _post_process_segments(self, segments):
        """Post-process segments to improve quality"""
        try:
            if not segments:
                return segments
            
            # 1. Merge very short segments with adjacent ones
            processed_segments = []
            for segment in segments:
                duration = segment['end'] - segment['start']
                if duration < 0.3:  # Very short segment (< 300ms)
                    # Try to merge with previous segment
                    if processed_segments:
                        last_segment = processed_segments[-1]
                        if last_segment['speaker'] == segment['speaker']:
                            # Same speaker, merge
                            last_segment['end'] = segment['end']
                            continue
                        elif segment['start'] - last_segment['end'] < 0.2:  # Small gap
                            # Close gap, merge
                            last_segment['end'] = segment['end']
                            continue
                
                processed_segments.append(segment)
            
            # 2. Smooth speaker transitions (reduce ping-pong effect)
            if len(processed_segments) >= 3:
                smoothed_segments = [processed_segments[0]]
                
                for i in range(1, len(processed_segments) - 1):
                    prev_seg = processed_segments[i-1]
                    curr_seg = processed_segments[i]
                    next_seg = processed_segments[i+1]
                    
                    # If current segment is very short and sandwiched between same speaker
                    curr_duration = curr_seg['end'] - curr_seg['start']
                    if (curr_duration < 1.0 and 
                        prev_seg['speaker'] == next_seg['speaker'] and
                        curr_seg['speaker'] != prev_seg['speaker']):
                        # Skip this segment (merge prev with next)
                        continue
                    
                    smoothed_segments.append(curr_seg)
                
                # Add last segment
                if len(processed_segments) > 1:
                    smoothed_segments.append(processed_segments[-1])
                
                processed_segments = smoothed_segments
            
            # 3. Ensure no overlapping segments
            for i in range(len(processed_segments) - 1):
                if processed_segments[i]['end'] > processed_segments[i+1]['start']:
                    # Adjust boundary to midpoint
                    midpoint = (processed_segments[i]['end'] + processed_segments[i+1]['start']) / 2
                    processed_segments[i]['end'] = midpoint
                    processed_segments[i+1]['start'] = midpoint
            
            logger.info(f"Post-processing: {len(segments)} -> {len(processed_segments)} segments")
            return processed_segments
            
        except Exception as e:
            logger.error(f"Segment post-processing failed: {e}")
            return segments

    def _create_fallback_segments(self, audio_file, num_speakers):
        """Create fallback segments when diarization fails"""
        try:
            # Get audio duration
            if not _ensure_scipy_imports() or _scipy_wavfile is None:
                duration = 10.0  # Default fallback
            else:
                try:
                    Fs, x = _scipy_wavfile.read(audio_file)
                    duration = len(x) / Fs if x.ndim == 1 else len(x) / Fs
                except Exception:
                    duration = 10.0
            
            # Create simple segments alternating between speakers
            segments = []
            if num_speakers == 1:
                segments.append({
                    'start': 0.0,
                    'end': duration,
                    'speaker': 0,
                    'confidence': 0.5
                })
            else:
                # Divide duration among speakers
                segment_duration = duration / max(2, num_speakers)
                for i in range(num_speakers):
                    start_time = i * segment_duration
                    end_time = min((i + 1) * segment_duration, duration)
                    if end_time > start_time:
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'speaker': i,
                            'confidence': 0.3  # Low confidence for fallback
                        })
            
            logger.info(f"Created {len(segments)} fallback segments")
            return segments
            
        except Exception as e:
            logger.error(f"Fallback segment creation failed: {e}")
            return [{
                'start': 0.0,
                'end': 10.0,
                'speaker': 0,
                'confidence': 0.1
            }]

    def cleanup_temp_files(self):
        """Clean up temporary files created during processing"""
        try:
            from pathlib import Path
            # Use the data path with temp subdirectory instead of get_temp_dir
            temp_dir = Path(FileManager.get_data_path("temp"))
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*diarization*"):
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
        except Exception as e:
            logger.warning(f"Temp file cleanup failed: {e}")

# Cleanup temp files on module exit
import atexit
_diarization_instance = None

def _cleanup_on_exit():
    global _diarization_instance
    if _diarization_instance:
        _diarization_instance.cleanup_temp_files()

atexit.register(_cleanup_on_exit)

def detect_speaker_changes_from_content(transcription_segments, language="pt"):
    """Detect potential speaker changes from transcription content"""
    try:
        if not transcription_segments:
            return []
        
        content_hints = []
        
        # Portuguese conversation markers that suggest speaker changes
        pt_markers = [
            "então", "bem", "olha", "veja", "mas", "porém", "contudo",
            "sim", "não", "claro", "certo", "ok", "tá", "né", "sabe",
            "desculpa", "desculpe", "perdão", "obrigado", "obrigada"
        ]
        
        # English conversation markers
        en_markers = [
            "well", "okay", "right", "yes", "no", "sure", "actually",
            "but", "however", "sorry", "excuse me", "thank you", "thanks"
        ]
        
        # Spanish conversation markers  
        es_markers = [
            "bueno", "bien", "mira", "pero", "sin embargo", "sí", "no",
            "claro", "vale", "perdón", "disculpa", "gracias"
        ]
        
        markers = pt_markers if language == "pt" else en_markers if language == "en" else es_markers
        
        for i, segment in enumerate(transcription_segments):
            text = segment.get('text', '').lower().strip()
            
            # Check for conversation markers at segment start
            words = text.split()
            if words and words[0] in markers:
                content_hints.append({
                    'time': segment.get('start', 0),
                    'confidence': 0.6,
                    'reason': f'conversation_marker_{words[0]}'
                })
            
            # Check for question-answer patterns
            if text.endswith('?') and i < len(transcription_segments) - 1:
                next_segment = transcription_segments[i + 1]
                next_text = next_segment.get('text', '').lower().strip()
                if any(next_text.startswith(marker) for marker in markers[:5]):
                    content_hints.append({
                        'time': next_segment.get('start', 0),
                        'confidence': 0.7,
                        'reason': 'question_answer_pattern'
                    })
        
        logger.info(f"Content-based speaker change hints: {len(content_hints)}")
        return content_hints
        
    except Exception as e:
        logger.error(f"Content-based speaker change detection failed: {e}")
        return []

def align_transcription_with_diarization(transcription_segments, diarization_segments):
    """Align transcription with diarization segments"""
    try:
        if not transcription_segments or not diarization_segments:
            logger.warning("Empty segments provided for alignment")
            return transcription_segments
        
        aligned_segments = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg.get('start', 0)
            trans_end = trans_seg.get('end', 0)
            trans_mid = (trans_start + trans_end) / 2
            
            # Find the diarization segment that best overlaps with this transcription
            best_overlap = 0
            best_speaker = 0
            
            for diar_seg in diarization_segments:
                diar_start = diar_seg.get('start', 0)
                diar_end = diar_seg.get('end', 0)
                
                # Calculate overlap
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    trans_duration = trans_end - trans_start
                    
                    if trans_duration > 0:
                        overlap_ratio = overlap_duration / trans_duration
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_speaker = diar_seg.get('speaker', 0)
            
            # Create aligned segment
            aligned_segment = trans_seg.copy()
            aligned_segment['speaker'] = best_speaker
            aligned_segment['diarization_confidence'] = best_overlap
            
            aligned_segments.append(aligned_segment)
        
        logger.info(f"Alignment completed: {len(transcription_segments)} -> {len(aligned_segments)} aligned segments")
        return aligned_segments
        
    except Exception as e:
        logger.error(f"Transcription-diarization alignment failed: {e}")
        return transcription_segments