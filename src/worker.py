# src/worker.py

"""
Dedicated worker process for handling the CPU-intensive audio processing pipeline.
This runs in a separate process to avoid blocking the main web server.
"""

import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, Any, Optional

from src.audio_processing import AudioQualityAnalyzer
from src.diarization import PyannoteDiarizer
from src.transcription import TranscriptionService
from src.subtitle_generator import generate_srt
from src.file_manager import FileManager

# Configure logging for the worker process
logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Worker-Global Services ---
# These will be initialized once per worker process by the initializer
transcription_service: Optional[TranscriptionService] = None
diarization_service: Optional[PyannoteDiarizer] = None
audio_quality_analyzer: Optional[AudioQualityAnalyzer] = None
file_manager: Optional[FileManager] = None

def init_worker(config: Dict[str, Any]):
    """
    Initializer function for each worker process in the pool.
    Loads expensive models once per process.
    """
    global transcription_service, diarization_service, audio_quality_analyzer, file_manager
    
    logger.info(f"Initializing worker process...")
    try:
        transcription_service = TranscriptionService(model_name=config["model_name"], device=config["device"])
        diarization_service = PyannoteDiarizer(device=config["device"])
        audio_quality_analyzer = AudioQualityAnalyzer()
        file_manager = FileManager()
        logger.info("Worker process initialized successfully.")
    except Exception as e:
        logger.error(f"Error during worker initialization: {e}", exc_info=True)
        # Re-raise to ensure the pool knows the worker failed to initialize
        raise

def process_audio_task(audio_path: str, session_id: str, config: Dict[str, Any], communication_queue):
    """The entry point and main logic for the audio processing worker."""
    try:
        # Check if services are initialized
        if not all([transcription_service, diarization_service, audio_quality_analyzer, file_manager]):
            raise RuntimeError("Worker services not initialized. The pool initializer may have failed.")

        logger.info(f"[Worker-{session_id}] Starting audio processing task.")
        
        # Run the asyncio pipeline within the synchronous worker function
        # Services are now passed from the worker's global scope
        asyncio.run(run_async_pipeline(
            audio_path, 
            session_id, 
            config, 
            communication_queue,
            transcription_service,
            diarization_service,
            audio_quality_analyzer,
            file_manager
        ))
        logger.info(f"[Worker-{session_id}] Task completed successfully.")
    except Exception as e:
        logger.error(f"[Worker-{session_id}] An error occurred: {e}", exc_info=True)
        communication_queue.put({'type': 'error', 'message': str(e)})

async def run_async_pipeline(audio_path: str, session_id: str, config: Dict[str, Any], communication_queue, transcription_service, diarization_service, audio_quality_analyzer, file_manager):
    """The core audio processing pipeline, adapted to use a queue for communication."""
    
    def send_message(message: Dict[str, Any]):
        """Helper to put a message on the communication queue."""
        communication_queue.put(message)

    try:
        process = psutil.Process()
        mem_baseline = process.memory_info().rss / (1024 * 1024)
        peak_mem_mb = mem_baseline
        logger.info(f"[MEMORY PROFILING] Worker Baseline: {mem_baseline:.2f} MB")

        import librosa
        audio_duration = librosa.get_duration(path=audio_path)
        pipeline_start_time = time.time()

        send_message({'type': 'progress', 'stage': 'start', 'percentage': 5, 'message': 'Analisando qualidade do áudio...'})
        
        quality_metrics = audio_quality_analyzer.analyze_audio_quality(audio_path)
        if quality_metrics.has_issues and quality_metrics.warnings:
            for warning in quality_metrics.warnings:
                send_message({'type': 'warning', 'message': warning})
            await asyncio.sleep(4)

        send_message({'type': 'progress', 'stage': 'transcription', 'percentage': 10, 'message': 'Iniciando transcrição...'})
        
        transcription_result = await transcription_service.transcribe_with_enhancements(audio_path, word_timestamps=True)
        peak_mem_mb = max(peak_mem_mb, process.memory_info().rss / (1024 * 1024))
        logger.info(f"[MEMORY PROFILING] After Transcription: {peak_mem_mb:.2f} MB")

        send_message({'type': 'progress', 'stage': 'diarization', 'percentage': 50, 'message': 'Transcrição concluída. Identificando falantes...'})

        diarization_result = await diarization_service.diarize(audio_path, transcription_result.segments)
        peak_mem_mb = max(peak_mem_mb, process.memory_info().rss / (1024 * 1024))
        logger.info(f"[MEMORY PROFILING] After Diarization: {peak_mem_mb:.2f} MB")

        send_message({'type': 'progress', 'stage': 'srt', 'percentage': 80, 'message': 'Gerando legendas...'})

        srt_path = await generate_srt(diarization_result["segments"], output_path=file_manager.get_data_path("temp"), filename=f"{session_id}.srt")

        pipeline_end_time = time.time()
        actual_processing_time = pipeline_end_time - pipeline_start_time
        processing_ratio = actual_processing_time / audio_duration if audio_duration > 0 else 0

        logger.info(f"Processing complete: {actual_processing_time:.2f}s for {audio_duration:.2f}s audio (ratio: {processing_ratio:.2f}x)")

        final_result = {
            "segments": diarization_result["segments"],
            "num_speakers": diarization_result["num_speakers"],
            "processing_time": round(actual_processing_time, 2),
            "processing_ratio": round(processing_ratio, 2),
            "audio_duration": round(audio_duration, 2),
            "srt_path": srt_path, # Pass the srt_path back to the main process
            "peak_memory_mb": round(peak_mem_mb, 2)
        }

        send_message({'type': 'complete', 'result': final_result})

    except Exception as e:
        logger.error(f"Pipeline failed for session {session_id}: {e}", exc_info=True)
        send_message({'type': 'error', 'message': str(e)})
    finally:
        logger.info(f"[Worker-{session_id}] Task finished. Triggering aggressive garbage collection.")
        # Explicitly delete large objects to aid garbage collection in long-lived workers
        if 'transcription_result' in locals():
            del transcription_result
        if 'diarization_result' in locals():
            del diarization_result
        if 'final_result' in locals():
            del final_result
        # Force garbage collection
        gc.collect()
