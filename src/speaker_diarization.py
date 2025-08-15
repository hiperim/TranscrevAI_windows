import asyncio
import logging
import os
import subprocess
import static_ffmpeg
import sys
import numpy as np
import librosa
import shutil
import warnings
import time
from scipy.fftpack import fft
from enum import Enum
from unittest.mock import patch
from scipy.io import wavfile
from src.file_manager import FileManager
from src.logging_setup import setup_app_logging

# Use proper logging setup first
logger = setup_app_logging(logger_name="transcrevai.speaker_diarization")

# Import pyAudioAnalysis with graceful fallback
try:
    from pyAudioAnalysis import audioSegmentation as aS
    from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction
    PYAUDIO_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"pyAudioAnalysis not available: {e}")
    PYAUDIO_ANALYSIS_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class DummyAudioSegmentation:
        @staticmethod
        def silence_removal(*args, **kwargs):
            return [(0.0, 1.0)]  # Return single segment
        
        @staticmethod
        def speaker_diarization(*args, **kwargs):
            import numpy as np
            return (np.array([0]), np.array([0]), None)
    
    class DummyMidTermFeatures:
        @staticmethod
        def mid_feature_extraction(*args, **kwargs):
            import numpy as np
            return (np.random.rand(50, 10), None)  # Dummy features
    
    aS = DummyAudioSegmentation()
    mid_feature_extraction = DummyMidTermFeatures.mid_feature_extraction

class DiarizationError(Enum):
    FILE_NOT_FOUND = 1
    INVALID_FORMAT = 2
    EMPTY_AUDIO = 3
    INSUFFICIENT_DATA = 4

class SpeakerDiarization:
    """Enhanced speaker diarization with comprehensive error handling"""
    
    def __init__(self):
        try:
            if shutil.which('ffmpeg') is None:
                static_ffmpeg.add_paths()
            
            self.ffmpeg_path = shutil.which('ffmpeg')
            if not self.ffmpeg_path:
                logger.warning("FFmpeg not found - some features may be limited")
        except Exception as e:
            logger.warning(f"FFmpeg setup failed: {str(e)}")
    
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
            min_duration = 0.5  # 500ms minimum
            if len(x) / Fs < min_duration:
                logger.warning(f"Audio too short for VAD: {len(x)/Fs:.2f}s, returning full duration")
                return [(0.0, len(x) / Fs)]
            
            # Perform voice activity detection with error handling
            try:
                vad_segments = aS.silence_removal(
                    x, Fs,
                    st_win=0.1,     # Smaller window for short audio
                    st_step=0.05,
                    smooth_window=0.2,
                    weight=0.3      # Lower threshold for detection
                )
                
                # Validate VAD results
                if not vad_segments or len(vad_segments) == 0:
                    logger.warning("VAD found no speech segments, using full audio")
                    return [(0.0, len(x) / Fs)]
                
                # Filter out very short segments
                valid_segments = []
                for start, end in vad_segments:
                    if end - start >= 0.1:  # Minimum 100ms segments
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
    
    async def diarize_audio(self, audio_file, number_speakers=0):
        """Enhanced diarization with comprehensive error handling"""
        try:
            logger.info(f"Starting diarization process for {audio_file}")
            
            # Validate input file
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
            if os.path.getsize(audio_file) == 0:
                raise ValueError("Audio file is empty")
            
            # Perform VAD preprocessing with enhanced error handling
            try:
                vad_segments = self.preprocess_audio_with_vad(audio_file)
            except Exception as vad_error:
                logger.error(f"VAD preprocessing failed: {vad_error}")
                # Create fallback segments
                duration = self.get_audio_duration(audio_file)
                vad_segments = [(0.0, duration)]
            
            # Run diarization in thread pool
            return await asyncio.to_thread(self.diarize, audio_file, number_speakers, vad_segments)
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            # Return fallback single speaker segment
            try:
                duration = self.get_audio_duration(audio_file)
                return [{
                    "start": 0.0,
                    "end": duration,
                    "speaker": "Speaker_1"
                }]
            except:
                return [{
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "Speaker_1"
                }]
    
    def diarize(self, audio_file, number_speakers=0, vad_segments=None):
        """Enhanced internal diarization with FFT error fixes"""
        try:
            logger.info(f"Processing diarization on {audio_file}")
            
            # Read and validate audio
            Fs, x = wavfile.read(audio_file)
            if len(x) == 0:
                raise ValueError("Empty audio data")
            
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
            min_samples = int(Fs * 0.5)  # 500ms minimum
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
                min_audio_length = 2.0  # 2 seconds minimum
                if len(x_filt) / Fs < min_audio_length:
                    logger.warning(f"Audio too short for proper diarization: {len(x_filt)/Fs:.2f}s")
                    # Return single speaker for short audio
                    return [{
                        "start": 0.0,
                        "end": len(x_filt) / Fs,
                        "speaker": "Speaker_1"
                    }]
                
                with patch("scipy.fftpack.fft", new=self.safe_fft):
                    features = mid_feature_extraction(
                        signal=x_filt,
                        sampling_rate=Fs,
                        mid_window=min(1.0, len(x_filt) / Fs / 4),  # Adaptive window
                        mid_step=0.5,
                        short_window=0.05,
                        short_step=0.05
                    )
                
                if not features or len(features) < 2:
                    raise ValueError("Feature extraction failed - insufficient data")
                
                mid_term_features = features[0]
                if mid_term_features.shape[1] < 2:
                    raise ValueError("Insufficient features for diarization")
                    
            except Exception as feature_error:
                logger.error(f"Feature extraction failed: {feature_error}")
                # Return single speaker fallback
                return [{
                    "start": 0.0,
                    "end": len(x_filt) / Fs,
                    "speaker": "Speaker_1"
                }]
            
            # Perform speaker diarization with error handling
            try:
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
                    
            except Exception as diar_error:
                logger.error(f"Diarization algorithm failed: {diar_error}")
                # Return single speaker fallback
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
            # Return fallback result
            try:
                duration = len(x) / Fs if 'x' in locals() and 'Fs' in locals() else 1.0
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
            min_duration = 0.3  # Minimum segment duration
            
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
            return 1.0  # Default duration