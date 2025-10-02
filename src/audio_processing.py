"""
Enhanced Audio Processing Module - Optimized Chunking for ≤0.5:1 ratio
Includes adaptive chunking system merged from FASE 2 optimizations
CPU-only audio processing optimized
"""
import logging
import os
import asyncio
import numpy as np
from typing import Dict, Any, List, Tuple, Union

# Lazy imports for performance
_torch = None
_librosa = None
_soundfile = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

def _get_librosa():
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa

def _get_soundfile():
    global _soundfile
    if _soundfile is None:
        import soundfile as sf
        _soundfile = sf
    return _soundfile

logger = logging.getLogger(__name__)



class OptimizedAudioProcessor:
    """Enhanced audio utilities with VAD and optimized parallel processing for ≤0.5:1 ratio"""

    @staticmethod
    def torchaudio_get_duration(audio_path: str) -> float:
        """FASE 1 OPT 1: Ultra-fast duration extraction using torchaudio"""
        try:
            import torchaudio
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception as e:
            logger.warning(f"FASE 1: torchaudio duration failed: {e}, using soundfile fallback")
            try:
                with _get_soundfile().SoundFile(audio_path) as f:
                    return len(f) / f.samplerate
            except Exception as e2:
                logger.error(f"FASE 1: All duration methods failed: {e2}")
                return 30.0  # Safe fallback

    @staticmethod
    def torchaudio_optimized_resample(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """FASE 1 OPT 1: Optimized resampling using torchaudio tensors"""
        try:
            import torchaudio
            import torch

            # Convert to tensor efficiently
            audio_tensor = _get_torch().from_numpy(audio_data).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Use torchaudio's optimized resampler
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            resampled = resampler(audio_tensor)

            return resampled.squeeze().numpy()

        except Exception as e:
            logger.warning(f"FASE 1: torchaudio resample failed: {e}, using basic fallback")
            # Simple fallback using basic interpolation
            from scipy import signal
            import numpy as np
            return np.array(signal.resample(audio_data, int(len(audio_data) * target_sr / orig_sr)))

    @staticmethod
    def optimized_audio_load(audio_path: str, target_sr: int = 16000) -> Tuple[Any, int]:
        """Optimized audio loading with smart resampling - FASE 2.2 optimization"""
        try:
            import torchaudio
            import torch

            # First, check the original sample rate without loading the full audio
            info = _get_soundfile().info(audio_path)
            original_sr = info.samplerate

            logger.info(f"FASE 2.2: Audio SR {original_sr}Hz → target {target_sr}Hz")

            # Load audio with torchaudio (more stable)
            audio_tensor, sr = torchaudio.load(audio_path)

            # Convert to mono if needed
            if audio_tensor.shape[0] > 1:
                audio_tensor = _get_torch().mean(audio_tensor, dim=0, keepdim=True)

            # Convert to numpy
            audio_data = audio_tensor.squeeze().numpy()

            # If already at target sample rate, no resampling needed
            if sr == target_sr:
                logger.info("FASE 2.2: No resampling needed - optimal loading")
                return audio_data, sr

            # If close to target (±10%), use original to avoid quality loss
            elif abs(sr - target_sr) / target_sr <= 0.1:
                logger.info(f"FASE 2.2: SR close enough ({sr}Hz), using original")
                return audio_data, sr

            # Otherwise, resample efficiently using torchaudio
            else:
                logger.info(f"FASE 2.2: Resampling {sr}Hz → {target_sr}Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                resampled = resampler(audio_tensor)
                audio_data = resampled.squeeze().numpy()
                return audio_data, target_sr

        except Exception as e:
            logger.warning(f"FASE 1: Optimized torchaudio load failed: {e}, using soundfile fallback")
            # FASE 1 OPT 1: Remove librosa dependency, use soundfile
            try:
                audio_data, sr = _get_soundfile().read(audio_path)
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)  # Convert to mono

                # Resample if needed using optimized torchaudio
                if sr != target_sr:
                    audio_data = OptimizedAudioProcessor.torchaudio_optimized_resample(audio_data, sr, target_sr)
                    sr = target_sr

                return audio_data, sr
            except Exception as e2:
                logger.error(f"FASE 1: All audio loading methods failed: {e2}")
                raise e2








    






# ==========================================
# ROBUST AUDIO LOADER IMPLEMENTATION
# ==========================================
# Merged from robust_audio_loader.py for file consolidation

class RobustAudioLoader:
    """
    Robust audio loading with fallback strategies for Windows compatibility
    Handles various audio formats and loading libraries gracefully
    """

    def __init__(self):
        self.fallback_strategies = [
            self._load_with_soundfile,
            self._load_with_librosa_safe
        ]

    def load_audio(self, audio_path: str, target_sr: int = 16000, duration: Union[float, None] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio with multiple fallback strategies and comprehensive validation
        """
        # Pre-processing validation
        logger.info(f"Loading audio: {audio_path}")

        # Validate input parameters
        if not audio_path:
            raise ValueError("Audio path cannot be empty")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        if file_size < 100:  # Less than 100 bytes
            raise ValueError(f"Audio file too small ({file_size} bytes): {audio_path}")

        logger.info(f"✓ Pre-validation passed: {file_size} bytes")

        # Try each loading strategy
        last_exception = None
        for i, strategy in enumerate(self.fallback_strategies):
            try:
                logger.info(f"Trying audio loading strategy {i+1}/{len(self.fallback_strategies)}")
                data, sr = strategy(audio_path, target_sr, duration)

                # Comprehensive validation of loaded data
                if data is None:
                    raise ValueError("Strategy returned None data")

                if not isinstance(data, np.ndarray):
                    raise ValueError(f"Strategy returned invalid data type: {type(data)}")

                if len(data) == 0:
                    raise ValueError("Strategy returned empty audio data")

                # Check for silent audio
                max_amplitude = np.max(np.abs(data))
                if max_amplitude < 1e-6:
                    logger.warning(f"Audio appears to be very quiet (max amplitude: {max_amplitude:.8f})")

                # Check sample rate
                if sr <= 0:
                    raise ValueError(f"Invalid sample rate: {sr}")

                # Calculate duration
                duration_seconds = len(data) / sr
                logger.info(f"✅ Audio loading successful with strategy {i+1}:")
                logger.info(f"  Shape: {data.shape}")
                logger.info(f"  Sample rate: {sr} Hz")
                logger.info(f"  Duration: {duration_seconds:.2f} seconds")
                logger.info(f"  Max amplitude: {max_amplitude:.6f}")

                return data, sr

            except Exception as e:
                last_exception = e
                logger.error(f"Audio loading strategy {i+1} failed: {e}")
                continue

        # If all strategies fail, raise comprehensive error
        logger.error(f"CRITICAL: All audio loading strategies failed for: {audio_path}")
        logger.error(f"File size: {file_size} bytes")
        logger.error(f"Last exception: {last_exception}")
        raise RuntimeError(f"All audio loading strategies failed for: {audio_path}. Last error: {last_exception}")

    def _load_with_soundfile(self, audio_path: str, target_sr: int, duration: Union[float, None]) -> Tuple[np.ndarray, int]:
        """
        Primary loading strategy using soundfile (most reliable)
        """
        try:
            # Load with soundfile (handles most formats well)
            data, sr = _get_soundfile().read(audio_path, dtype='float32')

            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Handle duration limiting
            if duration is not None:
                max_samples = int(duration * sr)
                if len(data) > max_samples:
                    data = data[:max_samples]

            # Resample if needed
            if sr != target_sr:
                data = self._simple_resample(data, sr, target_sr)
                sr = target_sr

            logger.debug(f"✅ Soundfile loading success: shape={data.shape}, sr={sr}")
            return data, int(sr)

        except Exception as e:
            logger.debug(f"Soundfile loading failed: {e}")
            raise

    def _load_with_librosa_safe(self, audio_path: str, target_sr: int, duration: Union[float, None]) -> Tuple[np.ndarray, int]:
        """
        Fallback loading strategy using librosa with conservative settings
        """
        try:
            # First attempt with librosa
            try:
                data, sr = _get_librosa().load(
                    audio_path,
                    sr=target_sr,
                    duration=duration,
                    mono=True,
                    res_type='kaiser_fast'  # Resampling mais rápido e estável
                )

                logger.debug(f"✅ Librosa loading success: shape={data.shape}, sr={sr}")
                return data, int(sr)

            except Exception as e:
                # Se falhar, tentar com configurações ainda mais conservadoras
                logger.warning(f"Librosa first attempt failed: {e}, trying conservative mode")

                data, sr = _get_librosa().load(
                    audio_path,
                    sr=None,  # Manter sample rate original
                    duration=duration,
                    mono=True
                )

                # Resample manually se necessário
                if sr != target_sr:
                    data = self._simple_resample(data, int(sr), target_sr)
                    sr = target_sr

                logger.debug(f"✅ Librosa conservative loading success: shape={data.shape}, sr={sr}")
                return data, int(sr)

        except Exception as e:
            logger.error(f"All audio loading attempts failed: {e}")
            raise

    def _simple_resample(self, data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """Resampling simples usando interpolação linear"""
        if original_sr == target_sr:
            return data

        # Calculate new length
        new_length = int(len(data) * target_sr / original_sr)

        # Create indices for interpolation
        old_indices = np.linspace(0, len(data) - 1, len(data))
        new_indices = np.linspace(0, len(data) - 1, new_length)

        # Interpolate
        resampled = np.interp(new_indices, old_indices, data)

        return resampled.astype(np.float32)








