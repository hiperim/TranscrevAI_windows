# Real Whisper implementation for multiprocessing
import time
import whisper
import numpy as np
import json
from pathlib import Path
from src.logging_setup import setup_app_logging
from whisper_fast_download import load_whisper_fast

class RealOptimizedTranscriber:
    """Real Whisper transcriber for multiprocessing workers"""

    def __init__(self, model_name: str = "medium"):
        self.model_name = model_name
        self.model = None
        self.logger = setup_app_logging(logger_name="transcrevai.real_transcriber")

        # Load model immediately
        self.load_model()

    def load_model(self):
        """Load real Whisper model with fast download"""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            start_time = time.time()

            # Use fast downloader if model is not cached
            self.model = load_whisper_fast(self.model_name)

            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def transcribe_chunk(self, chunk_info):
        """Real transcription of audio chunk"""
        chunk_id, start_time, end_time = chunk_info
        duration = end_time - start_time

        if not self.model:
            self.logger.error("Model not loaded!")
            return None

        try:
            self.logger.info(f"Transcribing {chunk_id}: {duration:.1f}s")
            process_start = time.time()

            # Real Whisper transcription
            result = self.model.transcribe(
                chunk_id,
                language="pt",
                task="transcribe",
                verbose=False
            )

            process_time = time.time() - process_start
            ratio = process_time / duration if duration > 0 else 0

            self.logger.info(f"Processed {chunk_id}: {process_time:.1f}s (ratio: {ratio:.2f}x)")

            # Format result for compatibility
            formatted_result = {
                "chunk_id": chunk_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "processing_time": process_time,
                "ratio": ratio,
                "text": result["text"].strip(),
                "segments": [
                    {
                        "start": seg["start"] + start_time,
                        "end": seg["end"] + start_time,
                        "text": seg["text"].strip()
                    }
                    for seg in result["segments"]
                ],
                "language": result.get("language", "pt")
            }

            return formatted_result

        except Exception as e:
            self.logger.error(f"Error transcribing {chunk_id}: {e}")
            return None

def process_audio_file_real_whisper(audio_file_path: str, language: str = "pt"):
    """
    Real Whisper transcription for multiprocessing workers
    """
    logger = setup_app_logging(logger_name="transcrevai.real_multiprocessing")

    try:
        # Load audio using Whisper's built-in loader
        logger.info(f"Loading audio: {audio_file_path}")

        # Create transcriber
        transcriber = RealOptimizedTranscriber(model_name="medium")

        # Get audio info first
        import librosa
        audio_data, sr = librosa.load(audio_file_path, sr=16000)
        duration = len(audio_data) / sr

        logger.info(f"Audio loaded: {duration:.2f}s duration")

        if duration < 0.1:  # Too short
            logger.warning(f"Audio too short: {duration:.3f}s")
            return False

        # Split into chunks for processing
        chunk_duration = 30  # seconds
        results = []

        num_chunks = int(np.ceil(duration / chunk_duration))

        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, duration)

            # Extract chunk
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk_audio = audio_data[start_sample:end_sample]

            # Save temporary chunk file
            import tempfile
            temp_dir = tempfile.gettempdir()
            chunk_file = f"{temp_dir}/chunk_{i}.wav"
            import soundfile as sf
            sf.write(chunk_file, chunk_audio, sr)

            logger.info(f"Processing chunk {i+1}/{num_chunks} ({start_time:.1f}s-{end_time:.1f}s)")

            # Transcribe chunk
            chunk_result = transcriber.transcribe_chunk((chunk_file, start_time, end_time))

            if chunk_result and chunk_result.get('text', '').strip():
                results.append(chunk_result)

            # Clean up temp file
            try:
                Path(chunk_file).unlink()
            except:
                pass

        # Combine results
        if results:
            combined_text = ' '.join([r['text'] for r in results if r['text'].strip()])

            final_result = {
                'transcription': results,
                'full_text': combined_text,
                'duration': duration,
                'chunks_processed': len(results),
                'processing_method': 'real_whisper_multiprocessing',
                'model': 'medium',
                'language': language
            }

            # Save result
            output_file = str(audio_file_path).replace(".wav", "_real_transcription.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)

            logger.info(f"Real transcription completed: {output_file}")
            logger.info(f"Full text: {combined_text[:100]}...")

            return True
        else:
            logger.error("No transcription results obtained")
            return False

    except Exception as e:
        logger.error(f"Error in real Whisper transcription: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return False