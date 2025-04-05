import asyncio
import logging
import os
import subprocess
import static_ffmpeg
import sys
import numpy as np
import librosa
import regex as re
import shutil
import warnings
from scipy.fftpack import fft
from enum import Enum
from unittest.mock import patch
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction
from scipy.io import wavfile
from src.file_manager import FileManager

logger = logging.getLogger(__name__)

class DiarizationError(Enum):
    FILE_NOT_FOUND = 1
    INVALID_FORMAT = 2

class SpeakerDiarization:
    def __init__(self):
        try:
            if shutil.which('ffmpeg') is None:
                raise RuntimeError("FFmpeg not found in PATH")
            self.ffmpeg_path = shutil.which('ffmpeg')
            if not os.access(self.ffmpeg_path, os.X_OK):
                raise PermissionError(f"FFmpeg not executable: {self.ffmpeg_path}")
        except Exception as e:
            logger.critical(f"FFmpeg setup failed: {str(e)}")
            raise RuntimeError("FFmpeg initialization failed") from e
            
    def _get_version(self):
        try:
            result = subprocess.run([self.ffmpeg_path, "-version"], capture_output=True, text=True)
            version_line = result.stdout.split('\n')[0]
            version_match = re.search(r'ffmpeg version (\S+)', version_line)
            if version_match:
                return version_match.group(1)
            return "unknown version"
        except Exception:
            return "unknown"
    
    @staticmethod
    # Defensive wrapper around Scipy FFT - prevent crahes when processing data
    def safe_fft(x, n=None):
        if len(x) == 0:
            warnings.warn("Empty input to FFT, returning zero array")
            return np.zeros(n) if n else np.array([])
        return fft(x, n=n)
    
    def _verify_ffmpeg(self):
        try:
            if self.ffmpeg_path and os.access(self.ffmpeg_path, os.X_OK):
                result = subprocess.run([self.ffmpeg_path, "-version"], capture_output=True, text=True, check=True)
                return "ffmpeg version 4" in result.stdout or "ffmpeg version 5" in result.stdout
            else:
                import static_ffmpeg
                static_ffmpeg.add_paths()
                return self.ffmpeg_path
        except Exception as e:
            logger.error(f"FFmpeg verification failed: {e}")
            return False
        
    def preprocess_audio_with_vad(self, audio_file):
        # Voice activity detection
        Fs, x = wavfile.read(audio_file)
        if len(x) == 0:
            raise ValueError("Empty audio file")
        vad_segments = aS.silence_removal(x, Fs, st_win=0.05, st_step=0.05, smooth_window=1.0, weight=0.3)
        return vad_segments
    
    def save_processed_audio(self, x, Fs, output_file=None):
        try:
            if output_file is None:
                output_dir = FileManager.get_data_path("processed")
                base_filename = "output"
                extension = ".wav"
                counter = 1
                output_file = os.path.join(output_dir, f"{base_filename}{extension}")
                while os.path.exists(output_file):
                    counter += 1
                    output_file = os.path.join(output_dir, f"{base_filename}_{counter}{extension}")
                wavfile.write(output_file, Fs, (x * 32768).astype(np.int16))
                return output_file
        except Exception as e:
            raise RuntimeError(f"Failed to save processed audio: {e}")

    async def diarize_audio(self, audio_file, number_speakers=0):
        try:
            logger.info(f"Starting diarization process for {audio_file}")
            vad_segments = self.preprocess_audio_with_vad(audio_file)
            return await asyncio.to_thread(self._diarize, audio_file, number_speakers, vad_segments)
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise

    def _diarize(self, audio_file, number_speakers=0, vad_segments=None):
        try:
            logger.info(f"Performing diarization on {audio_file} with {len(vad_segments) if vad_segments else 0} VAD segments")
            if not vad_segments:
                raise ValueError("No VAD segments provided for diarization")
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
            if os.path.getsize(audio_file) == 0:
                raise ValueError("Audio file is empty")
            # x = array of audio data in numerical format (int or float)
            Fs, x = wavfile.read(audio_file)
            min_samples = int(Fs * 0.05)
            if len(x) < min_samples:
                pad_size = min_samples - len(x)
                x = np.pad(x, (0, pad_size), mode='constant')
            if x.ndim > 1:
                x = x.mean(axis=1)
            speech_samples = []
            for start, end in vad_segments:
                start_sample = int(start * Fs)
                end_sample = int(end * Fs)
                speech_samples.append(x[start_sample:end_sample])
            if not speech_samples:
                raise ValueError("VAD segments contain no audio data")
            # Parameter representing speech samples after VAD (filtered out non-speech segments)
            x_filt = np.concatenate(speech_samples)          
            if Fs != 16000:
                logger.warning(f"Resampling from {Fs}Hz to 16kHz")
                x_filt = librosa.resample(x_filt.astype(np.float32), orig_sr=Fs, target_sr=16000)
                Fs = 16000
            if x_filt.size == 0:
                raise ValueError(f"VAD processed .wav contains no audio data")
            if np.issubdtype(x_filt.dtype, np.integer) or np.max(np.abs(x_filt)) > 1.0:
                x_filt = x_filt.astype(np.float32) / 32.768
            else:
                x_filt = x_filt.astype(np.float32)
            if np.max(np.abs(x_filt)) == 0:
                raise ValueError("Silent audio file after VAD filtering")
            vad_proc_wav = os.path.join(FileManager.get_data_path("processed"), "vad_processed.wav")
            self.save_processed_audio(x_filt, Fs, vad_proc_wav)
            logger.info(f"Audio file read successfully: {len(x_filt)} samples at {Fs}Hz")
            min_samples = int(Fs * 0.05)
            if len(x_filt) < min_samples:
                raise ValueError(f"Audio too short for windowing: {len(x_filt)/Fs:.2f}s < {min_samples/Fs:.2f}s")

            if len(x_filt) == 0:
                logger.warning("Empty filtered audio data, adding padding")
                x_filt = np.zeros(int(Fs * 0.1), dtype=np.float32)  # Add 100ms of silence as padding
            with patch("scipy.fftpack.fft", new=self.safe_fft):
                features = mid_feature_extraction(signal=x_filt,
                                                  sampling_rate=Fs,
                                                  mid_window=1.0,
                                                  mid_step=0.5,
                                                  short_window=0.05,
                                                  short_step=0.05)
            if not features or not isinstance(features, tuple) or len(features) < 2:
                raise ValueError("Feature extraction failed")
            mid_term_features = features[0]
            if mid_term_features.shape[1] < 2:
                raise ValueError(f"Insufficient features for diarization: {mid_term_features.shape}")
            if len(x_filt) < Fs * 2:
                raise ValueError("Audio file too short for diarization")
            result = aS.speaker_diarization(vad_proc_wav, 
                                            n_speakers=number_speakers,
                                            mid_window=2.0,
                                            mid_step=0.1,
                                            short_window=0.05,
                                            short_step=0.05,
                                            lda_dim=35)
            logger.info(f"Result from aS.speaker_diarization: {result}")
            if not isinstance(result, tuple) or len(result) < 3:
                raise ValueError("Diarization failed - invalid results structure")
            flags, classes, _ = result
            if not isinstance(flags, np.ndarray) or len(flags) < 2:
                raise ValueError("Invalid diarization result - speaker flags array")
            if not isinstance(classes, np.ndarray):
                raise ValueError("Invalid diarization result - speaker classes array")
            if os.path.exists(vad_proc_wav):
                os.remove(vad_proc_wav)
            time_segments = []
            current_speaker = int(flags[0])
            start_time = 0
            audio_length = len(x_filt)/Fs
            min_duration = max(0.3, audio_length * 0.05)
            for i in range(1, len(flags)):
                if flags[i] != current_speaker:
                    end_time = (i * 0.1) * (len(x_filt)/Fs) / len(flags) # Conversion to secs
                    duration = end_time - start_time
                    if duration >= min_duration:
                        time_segments.append({"start": float(start_time), "end": float(end_time), "speaker": f"Speaker_{current_speaker + 1}"})
                    start_time = i
                    current_speaker = int(flags[i])
            final_end = len(flags) * 0.1
            final_duration = final_end - start_time
            if final_duration >= min_duration:
                time_segments.append({"start": float(start_time), "end": float(final_end), "speaker": f"Speaker_{current_speaker + 1}"}) 
            logger.info(f"Diarization completed: {len(time_segments)} segments identified")
            return time_segments
        except FileNotFoundError as e:
            logger.error(f"{DiarizationError.FILE_NOT_FOUND.name}: {str(e)}")
            raise RuntimeError("Audio file missing for diarization") from e
        except ValueError as e:
            logger.error(f"{DiarizationError.INVALID_FORMAT.name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during diarization: {str(e)}")
            raise