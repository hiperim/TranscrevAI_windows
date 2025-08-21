import asyncio
import wave
import json
import logging
import os
import tempfile
import soundfile as sf
import numpy as np
from scipy import signal

# Enhanced audio preprocessing imports
try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
from typing import AsyncGenerator, Tuple, List, Dict, Any
from pathlib import Path
from src.file_manager import FileManager
from config.app_config import MODEL_DIR, LANGUAGE_MODELS
from src.logging_setup import setup_app_logging

# Use proper logging setup first
logger = setup_app_logging(logger_name="transcrevai.transcription")

# Log library availability after logger is set up
if not PYLOUDNORM_AVAILABLE:
    logger.warning("pyloudnorm not available - LUFS normalization disabled")
if not NOISEREDUCE_AVAILABLE:
    logger.warning("noisereduce not available - noise reduction disabled")

# Import Vosk with graceful fallback
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vosk not available: {e}")
    VOSK_AVAILABLE = False
    
    # Create dummy classes for graceful degradation
    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass
    
    class DummyKaldiRecognizer:
        def __init__(self, *args, **kwargs):
            pass
        
        def AcceptWaveform(self, data):
            return True
        
        def Result(self):
            return '{"text": "Vosk not available"}'
        
        def FinalResult(self):
            return '{"text": "Vosk not available"}'
    
    Model = DummyModel
    KaldiRecognizer = DummyKaldiRecognizer

class TranscriptionError(Exception):
    pass

# Model management removed - handled by main.py

class AsyncTranscriptionService:
    """Enhanced transcription service with automatic model management"""
    
    def __init__(self):
        self._models = {}
        self._model_lock = asyncio.Lock()
    
    async def load_language_model(self, language_code: str) -> Any:
        """Load cached language model (assumes model already exists)"""
        if not VOSK_AVAILABLE:
            logger.warning("Vosk not available, returning dummy model")
            return Model()  # Return dummy model
        
        async with self._model_lock:
            if language_code not in self._models:
                try:
                    # Get model path (assumes model already downloaded by main.py)
                    model_path = os.path.join(MODEL_DIR, language_code)
                    
                    # Load model in thread pool
                    loop = asyncio.get_event_loop()
                    model = await loop.run_in_executor(None, Model, model_path)
                    
                    self._models[language_code] = model
                    logger.info(f"Model loaded successfully for {language_code}")
                    
                except Exception as e:
                    logger.error(f"Failed to load model for {language_code}: {e}")
                    raise TranscriptionError(f"Model loading failed for {language_code}: {str(e)}")
            
            return self._models[language_code]

# Global service instance
transcription_service = AsyncTranscriptionService()

async def transcribe_audio_with_progress(
    wav_file: str,
    model_path: str,
    language_code: str,
    sample_rate: int = 16000
) -> AsyncGenerator[Tuple[int, List[Dict]], None]:
    """Enhanced transcription (assumes model already available)"""
    try:
        logger.info(f"Starting transcription for {wav_file} with language {language_code}")
        
        # Check if Vosk is available
        if not VOSK_AVAILABLE:
            logger.warning("Vosk not available, returning dummy transcription")
            yield 100, [{"start": 0.0, "end": 1.0, "text": "Vosk speech recognition not available"}]
            return
        
        # Load model (assumes already downloaded)
        model = await transcription_service.load_language_model(language_code)
        
        # Validate audio file
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"Audio file not found: {wav_file}")
        
        if os.path.getsize(wav_file) == 0:
            raise ValueError("Audio file is empty")
        
        # Process audio with enhanced error handling and preprocessing
        loop = asyncio.get_event_loop()
        
        def get_wave_info_and_preprocess():
            try:
                with wave.open(wav_file, "rb") as wf:
                    wave_info = {
                        "channels": wf.getnchannels(),
                        "sample_width": wf.getsampwidth(),
                        "framerate": wf.getframerate(),
                        "total_frames": wf.getnframes()
                    }
                
                # ENHANCED: Preprocess audio if needed for better quality
                needs_preprocessing = (
                    wave_info["channels"] != 1 or 
                    wave_info["sample_width"] != 2 or 
                    wave_info["framerate"] != sample_rate
                )
                
                if needs_preprocessing:
                    logger.info(f"Audio preprocessing needed: channels={wave_info['channels']}, "
                              f"sample_width={wave_info['sample_width']}, framerate={wave_info['framerate']}")
                    
                    # Read audio data for preprocessing
                    audio_data, original_sr = sf.read(wav_file)
                    
                    # ENHANCED: Stereo channel separation for better speaker detection
                    if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
                        logger.info("Stereo audio detected - analyzing channels for speaker separation")
                        
                        # Calculate energy difference between channels
                        left_channel = audio_data[:, 0]
                        right_channel = audio_data[:, 1]
                        
                        left_energy = np.sum(left_channel ** 2)
                        right_energy = np.sum(right_channel ** 2)
                        
                        # If significant energy difference, preserve stronger channel
                        if abs(left_energy - right_energy) > 0.3 * max(left_energy, right_energy):
                            audio_data = left_channel if left_energy > right_energy else right_channel
                            logger.info(f"Using {'left' if left_energy > right_energy else 'right'} channel (stronger signal)")
                        else:
                            # Balanced stereo - convert to mono with weighted mixing
                            audio_data = np.mean(audio_data, axis=1)
                            logger.info("Converted balanced stereo to mono")
                    elif len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)
                        logger.info("Converted multi-channel audio to mono")
                    
                    # ENHANCED: LUFS-based loudness normalization (professional standard)
                    if PYLOUDNORM_AVAILABLE:
                        try:
                            meter = pyln.Meter(original_sr)  # Create loudness meter
                            loudness = meter.integrated_loudness(audio_data)
                            
                            if loudness != -np.inf:  # Valid loudness measurement
                                # Normalize to -23 LUFS (broadcast standard)
                                target_lufs = -23.0
                                audio_data = pyln.normalize.loudness(audio_data, loudness, target_lufs)
                                logger.info(f"LUFS normalization: {loudness:.1f} LUFS → {target_lufs:.1f} LUFS")
                            else:
                                # Fallback to peak normalization
                                if np.max(np.abs(audio_data)) > 0:
                                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
                                    logger.info("Applied peak normalization (fallback)")
                        except Exception as lufs_error:
                            logger.warning(f"LUFS normalization failed: {lufs_error}, using peak normalization")
                            if np.max(np.abs(audio_data)) > 0:
                                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
                                logger.info("Applied peak normalization")
                    else:
                        # Standard peak normalization
                        if np.max(np.abs(audio_data)) > 0:
                            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
                            logger.info("Applied peak normalization")
                    
                    # Resample if needed
                    target_sr = sample_rate
                    if original_sr != target_sr:
                        try:
                            import librosa
                            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
                            logger.info(f"Resampled audio from {original_sr}Hz to {target_sr}Hz")
                        except ImportError:
                            logger.warning("librosa not available, skipping resampling")
                            target_sr = original_sr
                            logger.info(f"Using original sample rate: {target_sr}Hz")
                    
                    # ENHANCED: Advanced noise reduction
                    if NOISEREDUCE_AVAILABLE:
                        try:
                            # Apply spectral subtraction noise reduction
                            reduced_noise = nr.reduce_noise(y=audio_data, sr=target_sr, stationary=True)
                            audio_data = reduced_noise
                            logger.info("Applied advanced spectral noise reduction")
                        except Exception as nr_error:
                            logger.warning(f"Advanced noise reduction failed: {nr_error}, using basic filter")
                    
                    # Apply additional filtering
                    try:
                        # High-pass filter to remove low-frequency noise (rumble, hum)
                        sos_hp = signal.butter(4, 80, btype='high', fs=target_sr, output='sos')
                        audio_data = signal.sosfilt(sos_hp, audio_data)
                        
                        # Low-pass filter to remove high-frequency noise (hiss)
                        sos_lp = signal.butter(6, 8000, btype='low', fs=target_sr, output='sos')
                        audio_data = signal.sosfilt(sos_lp, audio_data)
                        
                        logger.info("Applied frequency band filtering (80Hz-8kHz)")
                        
                        # Gentle compressor to reduce dynamic range for better recognition
                        threshold = 0.3
                        ratio = 3.0
                        compressed = np.where(np.abs(audio_data) > threshold, 
                                            np.sign(audio_data) * (threshold + (np.abs(audio_data) - threshold) / ratio),
                                            audio_data)
                        audio_data = compressed
                        logger.info("Applied gentle audio compression")
                        
                    except Exception as filter_error:
                        logger.warning(f"Audio filtering failed: {filter_error}")
                    
                    # Final amplitude check and normalization
                    max_amplitude = np.max(np.abs(audio_data))
                    if max_amplitude > 0.95:
                        audio_data = audio_data / max_amplitude * 0.9
                        logger.info("Applied final amplitude limiting")
                    
                    # Save preprocessed audio to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        preprocessed_file = tmp_file.name
                    
                    # Convert to 16-bit PCM
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    sf.write(preprocessed_file, audio_int16, target_sr)
                    
                    # Update wave_info with preprocessed file info
                    with wave.open(preprocessed_file, "rb") as wf:
                        wave_info = {
                            "channels": wf.getnchannels(),
                            "sample_width": wf.getsampwidth(),
                            "framerate": wf.getframerate(),
                            "total_frames": wf.getnframes(),
                            "preprocessed_file": preprocessed_file
                        }
                    
                    logger.info(f"Enhanced audio preprocessing complete: {preprocessed_file}")
                
                return wave_info
            except Exception as e:
                raise ValueError(f"Invalid audio file format: {str(e)}")
        
        wave_info = await loop.run_in_executor(None, get_wave_info_and_preprocess)
        
        # Use preprocessed file if available
        actual_wav_file = wave_info.get("preprocessed_file", wav_file)
        
        if wave_info["total_frames"] == 0:
            raise ValueError("Audio file contains no audio data")
        
        logger.info(f"Audio info - Channels: {wave_info['channels']}, Sample rate: {wave_info['framerate']}, "
                   f"Total frames: {wave_info['total_frames']}, Duration: {wave_info['total_frames']/wave_info['framerate']:.2f}s")
        
        # Validate audio format for Vosk (mono, 16-bit PCM)
        if wave_info["channels"] != 1:
            logger.warning(f"Audio has {wave_info['channels']} channels, Vosk expects mono. This may affect transcription quality.")
        
        if wave_info["sample_width"] != 2:
            logger.warning(f"Audio has {wave_info['sample_width']} bytes per sample, Vosk expects 16-bit (2 bytes). This may affect transcription quality.")
        
        # Create recognizer with enhanced configuration
        recognizer = KaldiRecognizer(model, wave_info["framerate"])
        
        # ENHANCED: Configure recognizer for better accuracy and word-level timestamps
        try:
            # Enable word-level timestamps for better synchronization with diarization
            recognizer.SetWords(True)
            logger.info("Vosk recognizer configured with word-level timestamps")
        except Exception as e:
            logger.warning(f"Could not enable word-level timestamps: {e}")
        
        try:
            # Enable partial results for better real-time feedback
            recognizer.SetPartialWords(True)
            logger.info("Vosk recognizer configured with partial word results")
        except Exception as e:
            logger.warning(f"Could not enable partial word results: {e}")
        
        # ENHANCED: Additional Vosk optimization (maintaining confidence threshold)
        try:
            # Enable multiple alternatives for better accuracy
            recognizer.SetMaxAlternatives(3)
            logger.info("Vosk recognizer configured with multiple alternatives (3)")
        except Exception as e:
            logger.warning(f"Could not enable multiple alternatives: {e}")
        
        try:
            # Enable NLSML for better structured output
            recognizer.SetNLSML(True)
            logger.info("Vosk recognizer configured with NLSML structured output")
        except Exception as e:
            logger.warning(f"Could not enable NLSML: {e}")
        
        try:
            # Configure speaker adaptation (helps with multiple speakers)
            recognizer.SetSpkModel(None)  # Use default speaker model
            logger.info("Vosk recognizer configured with speaker adaptation")
        except Exception as e:
            logger.debug(f"Speaker adaptation not available: {e}")  # Not critical
        
        # Process audio in chunks with optimized size for better accuracy
        chunk_size = 16384  # Increased from 8192 for better processing efficiency
        transcription_data = []
        processed_frames = 0
        
        def process_audio():
            nonlocal processed_frames
            chunk_results = []
            
            try:
                with wave.open(actual_wav_file, "rb") as wf:
                    while True:
                        data = wf.readframes(chunk_size)
                        if len(data) == 0:
                            break
                        
                        processed_frames += len(data) // wave_info["sample_width"]
                        
                        # Process chunk
                        if recognizer.AcceptWaveform(data):
                            result = json.loads(recognizer.Result())
                            text = result.get("text", "").strip()
                            if text:
                                # ENHANCED: Extract word-level information if available
                                words = result.get("result", [])
                                if words and isinstance(words, list):
                                    # Process individual words with timestamps
                                    for word_info in words:
                                        if isinstance(word_info, dict) and "word" in word_info:
                                            word_segment = {
                                                "start": word_info.get("start", processed_frames / wave_info["framerate"] - 
                                                        len(data) / (wave_info["sample_width"] * wave_info["framerate"])),
                                                "end": word_info.get("end", processed_frames / wave_info["framerate"]),
                                                "text": word_info["word"],
                                                "confidence": word_info.get("conf", 1.0)
                                            }
                                            chunk_results.append(word_segment)
                                else:
                                    # Fallback to sentence-level timing
                                    segment = {
                                        "start": processed_frames / wave_info["framerate"] - 
                                                len(data) / (wave_info["sample_width"] * wave_info["framerate"]),
                                        "end": processed_frames / wave_info["framerate"],
                                        "text": text,
                                        "confidence": result.get("confidence", 1.0)
                                    }
                                    chunk_results.append(segment)
                                logger.debug(f"Interim result: '{text}' at {(processed_frames / wave_info['framerate'] - len(data) / (wave_info['sample_width'] * wave_info['framerate'])):.2f}-{processed_frames / wave_info['framerate']:.2f}s")
                        
                        # Yield progress
                        progress = min(100, int((processed_frames / wave_info["total_frames"]) * 100))
                        chunk_results.append(("progress", progress))
                
                # Get final result with enhanced processing
                final_result = json.loads(recognizer.FinalResult())
                final_text = final_result.get("text", "").strip()
                logger.info(f"Final result: '{final_text}' (length: {len(final_text)})")
                
                if final_text:
                    # ENHANCED: Process final result with word-level information if available
                    final_words = final_result.get("result", [])
                    if final_words and isinstance(final_words, list):
                        # Process individual words from final result
                        for word_info in final_words:
                            if isinstance(word_info, dict) and "word" in word_info:
                                word_segment = {
                                    "start": word_info.get("start", max(0, processed_frames / wave_info["framerate"] - 1)),
                                    "end": word_info.get("end", processed_frames / wave_info["framerate"]),
                                    "text": word_info["word"],
                                    "confidence": word_info.get("conf", 1.0)
                                }
                                chunk_results.append(word_segment)
                    else:
                        # Fallback to sentence-level final result
                        final_segment = {
                            "start": max(0, processed_frames / wave_info["framerate"] - 1),
                            "end": processed_frames / wave_info["framerate"],
                            "text": final_text,
                            "confidence": final_result.get("confidence", 1.0)
                        }
                        chunk_results.append(final_segment)
                    logger.info(f"Added final segment(s): '{final_text}' at {max(0, processed_frames / wave_info['framerate'] - 1):.2f}-{processed_frames / wave_info['framerate']:.2f}s")
                else:
                    logger.warning("No final transcription result detected")
                
                return chunk_results
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                return [("error", str(e))]
        
        # Process in executor
        chunk_results = await loop.run_in_executor(None, process_audio)
        
        # Yield results
        for item in chunk_results:
            if isinstance(item, tuple):
                if item[0] == "progress":
                    yield int(item[1]), transcription_data
                elif item[0] == "error":
                    raise TranscriptionError(f"Audio processing failed: {item[1]}")
            else:
                transcription_data.append(item)
                progress = min(100, len(transcription_data) * 10)
                # Yield without filtering during processing (filtering happens at end)
                yield progress, transcription_data
        
        # Enhanced filtering and post-processing
        def filter_transcription_duplicates(segments):
            """Filter segments with identical text and overlapping timestamps"""
            if not segments:
                return segments
            
            filtered = []
            for current in segments:
                # Skip empty transcriptions but be less aggressive with short ones
                text = current.get('text', '').strip()
                if not text:
                    continue
                
                should_add = True
                current_text = current.get('text', '').strip().lower()
                current_start = current.get('start', 0)
                current_end = current.get('end', 0)
                
                # Check against previous segments for duplicates
                for prev in filtered[-3:]:  # Only check last 3 segments for efficiency
                    prev_text = prev.get('text', '').strip().lower()
                    prev_start = prev.get('start', 0)
                    prev_end = prev.get('end', 0)
                    
                    # Check if texts are identical and timestamps overlap
                    if current_text == prev_text:
                        # Check for timestamp overlap
                        overlap = min(current_end, prev_end) - max(current_start, prev_start)
                        if overlap > 0:  # Timestamps overlap
                            should_add = False
                            break
                
                if should_add:
                    filtered.append(current)
            
            return filtered
        
        def enhance_transcription_text(segments):
            """Apply text post-processing and corrections"""
            enhanced = []
            
            for segment in segments:
                if not isinstance(segment, dict) or 'text' not in segment:
                    continue
                
                text = segment['text'].strip()
                if not text:
                    continue
                
                # Text enhancement techniques
                enhanced_text = text
                
                # 1. Capitalize first letter of sentences
                if enhanced_text and not enhanced_text[0].isupper():
                    enhanced_text = enhanced_text[0].upper() + enhanced_text[1:]
                
                # 2. Fix common transcription errors (context-dependent)
                common_fixes = {
                    ' i ': ' I ',
                    ' im ': ' I\'m ',
                    ' id ': ' I\'d ',
                    ' ill ': ' I\'ll ',
                    ' its ': ' it\'s ',
                    ' cant ': ' can\'t ',
                    ' wont ': ' won\'t ',
                    ' dont ': ' don\'t ',
                    ' youre ': ' you\'re ',
                    ' theyre ': ' they\'re ',
                    ' were ': ' we\'re '
                }
                
                for error, correction in common_fixes.items():
                    if error in ' ' + enhanced_text.lower() + ' ':
                        # Apply correction while preserving case context
                        words = enhanced_text.split()
                        for i, word in enumerate(words):
                            if word.lower() == error.strip():
                                words[i] = correction.strip()
                        enhanced_text = ' '.join(words)
                
                # 3. Remove excessive repetitions (but keep intentional repetitions)
                words = enhanced_text.split()
                if len(words) > 2:
                    # Remove more than 2 consecutive identical words
                    cleaned_words = []
                    i = 0
                    while i < len(words):
                        word = words[i]
                        count = 1
                        
                        # Count consecutive identical words
                        while i + count < len(words) and words[i + count].lower() == word.lower():
                            count += 1
                        
                        # Keep maximum 2 repetitions for emphasis
                        repetitions = min(count, 2)
                        cleaned_words.extend([word] * repetitions)
                        i += count
                    
                    enhanced_text = ' '.join(cleaned_words)
                
                # 4. Add basic punctuation for sentence boundaries
                if enhanced_text and not enhanced_text[-1] in '.!?':
                    # Check if this looks like end of sentence (next segment starts with capital or is different speaker)
                    enhanced_text += '.'
                
                # Create enhanced segment
                enhanced_segment = segment.copy()
                enhanced_segment['text'] = enhanced_text
                enhanced_segment['original_text'] = text  # Keep original for debugging
                
                enhanced.append(enhanced_segment)
            
            return enhanced
        
        # Apply filtering and enhancement
        original_count = len(transcription_data)
        transcription_data = filter_transcription_duplicates(transcription_data)
        filtered_count = len(transcription_data)
        
        # ENHANCED: Apply text post-processing
        transcription_data = enhance_transcription_text(transcription_data)
        enhanced_count = len(transcription_data)
        
        logger.info(f"Transcription processing: {original_count} raw → {filtered_count} filtered → {enhanced_count} enhanced segments")
        
        if original_count > 0 and filtered_count == 0:
            logger.warning("All transcription segments were filtered out - this may indicate an issue with filtering logic or audio quality")
        
        # Clean up preprocessed file if it was created
        if "preprocessed_file" in wave_info:
            try:
                os.unlink(wave_info["preprocessed_file"])
                logger.info("Cleaned up preprocessed audio file")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up preprocessed file: {cleanup_error}")
        
        # Final yield
        yield 100, transcription_data
        
        logger.info(f"Transcription completed: {filtered_count} segments (after filtering)")
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise TranscriptionError(f"Transcription error: {str(e)}")