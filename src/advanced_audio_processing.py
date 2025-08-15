import numpy as np
import scipy.signal
import logging
from typing import Tuple, Optional
from scipy import ndimage
from scipy.signal import butter, filtfilt, medfilt
import librosa

from src.logging_setup import setup_app_logging

logger = setup_app_logging(logger_name="transcrevai.advanced_audio")

class AdvancedAudioProcessor:
    """
    Advanced audio processing with noise reduction, echo cancellation, 
    and audio enhancement features for Phase 2.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize advanced audio processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.noise_profile = None
        self.prev_audio = None
        
        # Audio processing parameters
        self.noise_reduction_factor = 0.8
        self.echo_delay_samples = int(0.1 * sample_rate)  # 100ms echo delay
        self.echo_decay = 0.3
        
        # Spectral subtraction parameters
        self.alpha = 2.0  # Over-subtraction factor
        self.beta = 0.01  # Spectral floor
        
        # High-pass filter for removing low frequency noise
        self.highpass_cutoff = 80  # Hz
        
        logger.info(f"Advanced audio processor initialized (sample_rate: {sample_rate})")
    
    def create_noise_profile(self, noise_audio: np.ndarray, duration: float = 1.0) -> bool:
        """
        Create a noise profile from a sample of background noise.
        
        Args:
            noise_audio: Audio containing only background noise
            duration: Duration of noise sample to analyze
            
        Returns:
            True if noise profile created successfully
        """
        try:
            if len(noise_audio) == 0:
                logger.warning("Empty noise audio provided")
                return False
            
            # Take first 'duration' seconds of audio
            max_samples = int(duration * self.sample_rate)
            noise_sample = noise_audio[:min(len(noise_audio), max_samples)]
            
            # Compute noise spectrum using STFT
            stft_noise = librosa.stft(noise_sample.astype(np.float32), hop_length=512)
            noise_magnitude = np.abs(stft_noise)
            
            # Average noise spectrum across time
            self.noise_profile = np.mean(noise_magnitude, axis=1, keepdims=True)
            
            logger.info(f"Noise profile created from {len(noise_sample)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create noise profile: {e}")
            return False
    
    def reduce_noise_spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Reduce noise using spectral subtraction method.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Noise-reduced audio signal
        """
        try:
            if self.noise_profile is None:
                logger.warning("No noise profile available, skipping noise reduction")
                return audio
            
            if len(audio) == 0:
                return audio
            
            # Convert to float32 for processing
            audio_float = audio.astype(np.float32)
            
            # Compute STFT
            stft_audio = librosa.stft(audio_float, hop_length=512)
            magnitude = np.abs(stft_audio)
            phase = np.angle(stft_audio)
            
            # Ensure noise profile matches frequency bins
            if self.noise_profile.shape[0] != magnitude.shape[0]:
                # Resize noise profile to match
                from scipy import interpolate
                old_freqs = np.linspace(0, 1, self.noise_profile.shape[0])
                new_freqs = np.linspace(0, 1, magnitude.shape[0])
                interp_func = interpolate.interp1d(old_freqs, self.noise_profile[:, 0], 
                                                 kind='linear', fill_value='extrapolate')
                self.noise_profile = interp_func(new_freqs).reshape(-1, 1)
            
            # Spectral subtraction
            noise_estimate = self.noise_profile * self.alpha
            
            # Subtract noise estimate from magnitude
            enhanced_magnitude = magnitude - noise_estimate
            
            # Apply spectral floor to prevent over-subtraction
            spectral_floor = self.beta * magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, spectral_floor)
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio.astype(audio.dtype)
            
        except Exception as e:
            logger.error(f"Spectral subtraction failed: {e}")
            return audio
    
    def reduce_noise_wiener_filter(self, audio: np.ndarray, noise_power: float = 0.01) -> np.ndarray:
        """
        Reduce noise using Wiener filtering.
        
        Args:
            audio: Input audio signal
            noise_power: Estimated noise power
            
        Returns:
            Filtered audio signal
        """
        try:
            if len(audio) == 0:
                return audio
            
            # Convert to frequency domain
            audio_fft = np.fft.fft(audio.astype(np.float32))
            power_spectrum = np.abs(audio_fft) ** 2
            
            # Wiener filter
            wiener_filter = power_spectrum / (power_spectrum + noise_power)
            
            # Apply filter
            filtered_fft = audio_fft * wiener_filter
            filtered_audio = np.fft.ifft(filtered_fft).real
            
            return filtered_audio.astype(audio.dtype)
            
        except Exception as e:
            logger.error(f"Wiener filtering failed: {e}")
            return audio
    
    def cancel_echo(self, audio: np.ndarray, reference_audio: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cancel echo using adaptive filtering.
        
        Args:
            audio: Input audio with echo
            reference_audio: Reference signal for echo cancellation
            
        Returns:
            Echo-cancelled audio
        """
        try:
            if len(audio) == 0:
                return audio
            
            audio_float = audio.astype(np.float32)
            
            # If no reference audio, use simple echo suppression
            if reference_audio is None:
                return self._simple_echo_suppression(audio_float).astype(audio.dtype)
            
            # LMS adaptive filter for echo cancellation
            return self._adaptive_echo_cancellation(audio_float, reference_audio.astype(np.float32)).astype(audio.dtype)
            
        except Exception as e:
            logger.error(f"Echo cancellation failed: {e}")
            return audio
    
    def _simple_echo_suppression(self, audio: np.ndarray) -> np.ndarray:
        """Simple echo suppression by removing delayed signals."""
        try:
            if len(audio) <= self.echo_delay_samples:
                return audio
            
            # Create delayed version of audio
            delayed_audio = np.zeros_like(audio)
            delayed_audio[self.echo_delay_samples:] = audio[:-self.echo_delay_samples] * self.echo_decay
            
            # Subtract estimated echo
            echo_cancelled = audio - delayed_audio
            
            return echo_cancelled
            
        except Exception as e:
            logger.error(f"Simple echo suppression failed: {e}")
            return audio
    
    def _adaptive_echo_cancellation(self, audio: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Adaptive echo cancellation using LMS algorithm."""
        try:
            # Simple LMS implementation
            filter_length = min(256, len(audio) // 4)
            step_size = 0.01
            
            # Initialize adaptive filter
            adaptive_filter = np.zeros(filter_length)
            output = np.zeros_like(audio)
            
            # Process in chunks
            for i in range(filter_length, len(audio)):
                # Get reference window
                ref_window = reference[i-filter_length:i]
                
                # Predict echo
                echo_estimate = np.dot(adaptive_filter, ref_window)
                
                # Error signal (echo-cancelled output)
                error = audio[i] - echo_estimate
                output[i] = error
                
                # Update filter coefficients
                adaptive_filter += step_size * error * ref_window
            
            return output
            
        except Exception as e:
            logger.error(f"Adaptive echo cancellation failed: {e}")
            return audio
    
    def apply_high_pass_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.
        
        Args:
            audio: Input audio signal
            
        Returns:
            High-pass filtered audio
        """
        try:
            if len(audio) == 0:
                return audio
            
            # Design high-pass Butterworth filter
            nyquist = self.sample_rate / 2
            normalized_cutoff = self.highpass_cutoff / nyquist
            
            if normalized_cutoff >= 1.0:
                logger.warning("High-pass cutoff frequency too high, skipping filter")
                return audio
            
            b, a = butter(4, normalized_cutoff, btype='high')
            
            # Apply filter
            filtered_audio = filtfilt(b, a, audio.astype(np.float32))
            
            return filtered_audio.astype(audio.dtype)
            
        except Exception as e:
            logger.error(f"High-pass filtering failed: {e}")
            return audio
    
    def enhance_speech(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance speech quality using multiple techniques.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Enhanced audio signal
        """
        try:
            if len(audio) == 0:
                return audio
            
            enhanced = audio.astype(np.float32)
            
            # Apply high-pass filter to remove low-frequency noise
            enhanced = self.apply_high_pass_filter(enhanced)
            
            # Apply median filter to remove impulse noise
            enhanced = medfilt(enhanced, kernel_size=3)
            
            # Dynamic range compression
            enhanced = self._dynamic_range_compression(enhanced)
            
            # Spectral enhancement
            enhanced = self._spectral_enhancement(enhanced)
            
            return enhanced.astype(audio.dtype)
            
        except Exception as e:
            logger.error(f"Speech enhancement failed: {e}")
            return audio
    
    def _dynamic_range_compression(self, audio: np.ndarray, 
                                 threshold: float = 0.5, 
                                 ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression."""
        try:
            # Compute RMS energy in overlapping windows
            window_size = int(0.01 * self.sample_rate)  # 10ms windows
            hop_size = window_size // 2
            
            compressed = np.copy(audio)
            
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                
                if rms > threshold:
                    # Apply compression
                    gain_reduction = 1.0 / ratio
                    gain = threshold + (rms - threshold) * gain_reduction
                    compression_factor = gain / rms if rms > 0 else 1.0
                    compressed[i:i + window_size] *= compression_factor
            
            return compressed
            
        except Exception as e:
            logger.error(f"Dynamic range compression failed: {e}")
            return audio
    
    def _spectral_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Enhance spectral clarity."""
        try:
            # Apply spectral subtraction if noise profile available
            if self.noise_profile is not None:
                audio = self.reduce_noise_spectral_subtraction(audio)
            
            # Spectral sharpening using cepstral processing
            stft = librosa.stft(audio, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Enhance formants by sharpening spectral peaks
            enhanced_magnitude = np.power(magnitude, 1.2)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Spectral enhancement failed: {e}")
            return audio
    
    def normalize_audio(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """
        Normalize audio to target RMS level.
        
        Args:
            audio: Input audio signal
            target_rms: Target RMS level
            
        Returns:
            Normalized audio signal
        """
        try:
            if len(audio) == 0:
                return audio
            
            audio_float = audio.astype(np.float32)
            
            # Calculate current RMS
            current_rms = np.sqrt(np.mean(audio_float ** 2))
            
            if current_rms > 0:
                # Calculate gain factor
                gain = target_rms / current_rms
                
                # Limit gain to prevent clipping
                max_gain = 0.95 / np.max(np.abs(audio_float)) if np.max(np.abs(audio_float)) > 0 else 1.0
                gain = min(gain, max_gain)
                
                # Apply gain
                normalized = audio_float * gain
            else:
                normalized = audio_float
            
            return normalized.astype(audio.dtype)
            
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            return audio
    
    def process_audio_chunk(self, audio: np.ndarray, 
                          enable_noise_reduction: bool = True,
                          enable_echo_cancellation: bool = True,
                          enable_enhancement: bool = True) -> np.ndarray:
        """
        Process audio chunk with all enabled enhancements.
        
        Args:
            audio: Input audio chunk
            enable_noise_reduction: Enable noise reduction
            enable_echo_cancellation: Enable echo cancellation
            enable_enhancement: Enable speech enhancement
            
        Returns:
            Processed audio chunk
        """
        # Input validation
        if audio is None:
            logger.warning("Null audio input provided")
            return np.array([], dtype=np.float32)
        
        if not isinstance(audio, np.ndarray):
            try:
                audio = np.array(audio, dtype=np.float32)
            except Exception as e:
                logger.error(f"Failed to convert audio to numpy array: {e}")
                return np.array([], dtype=np.float32)
        
        if len(audio) == 0:
            return audio
        
        # Check for valid audio range
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            logger.warning("Invalid audio data (NaN or Inf), cleaning...")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            processed = audio.astype(np.float32)
            original_dtype = audio.dtype
            
            # Noise reduction with error handling
            if enable_noise_reduction:
                try:
                    if self.noise_profile is not None:
                        processed = self.reduce_noise_spectral_subtraction(processed)
                    else:
                        # Use Wiener filter as fallback
                        processed = self.reduce_noise_wiener_filter(processed)
                except Exception as nr_error:
                    logger.warning(f"Noise reduction failed: {nr_error}")
            
            # Echo cancellation with error handling
            if enable_echo_cancellation:
                try:
                    processed = self.cancel_echo(processed)
                except Exception as ec_error:
                    logger.warning(f"Echo cancellation failed: {ec_error}")
            
            # Speech enhancement with error handling
            if enable_enhancement:
                try:
                    processed = self.enhance_speech(processed)
                except Exception as enh_error:
                    logger.warning(f"Speech enhancement failed: {enh_error}")
            
            # Final normalization with error handling
            try:
                processed = self.normalize_audio(processed)
            except Exception as norm_error:
                logger.warning(f"Audio normalization failed: {norm_error}")
            
            # Ensure output is valid
            if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
                logger.warning("Processing resulted in invalid data, using original")
                return audio
            
            return processed.astype(original_dtype)
            
        except Exception as e:
            logger.error(f"Audio chunk processing failed: {e}")
            return audio
    
    def get_processing_stats(self) -> dict:
        """Get statistics about audio processing."""
        return {
            "sample_rate": self.sample_rate,
            "has_noise_profile": self.noise_profile is not None,
            "noise_reduction_factor": self.noise_reduction_factor,
            "echo_delay_ms": self.echo_delay_samples / self.sample_rate * 1000,
            "highpass_cutoff_hz": self.highpass_cutoff
        }