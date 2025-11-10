"""
Performance test for multiprocessing architecture
Tests with worker pool configuration from main.py
"""

import asyncio
import time
import logging
import multiprocessing
from pathlib import Path
import librosa

from src.worker import init_worker, process_audio_task

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_multiprocessing_pool():
    """Test with multiprocessing pool (current master architecture)"""

    # Load audio file
    audio_path = "data/recordings/d.speakers.wav"

    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        return

    # Get audio duration
    audio_duration = librosa.get_duration(path=audio_path)
    logger.info(f"Audio duration: {audio_duration:.2f}s")

    # Worker configuration (from main.py)
    worker_config = {
        "model_name": "medium",
        "device": "cpu"
    }

    # Determine number of workers (same as main.py TEST B)
    num_workers = 3
    logger.info(f"Initializing process pool with {num_workers} workers...")

    # Measure initialization time
    init_start = time.time()

    # Create pool with initializer (eager loading)
    pool = multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(worker_config,)
    )

    init_time = time.time() - init_start
    logger.info(f"Pool initialized in {init_time:.2f}s")

    # Create communication queue
    manager = multiprocessing.Manager()
    communication_queue = manager.Queue()

    # Run processing
    logger.info("Starting audio processing...")
    processing_start = time.time()

    # Submit task to pool
    result = pool.apply_async(
        process_audio_task,
        args=(audio_path, "test_session", worker_config, communication_queue)
    )

    # Monitor progress
    final_result = None
    transcription_time = 0
    diarization_time = 0

    while True:
        try:
            message = communication_queue.get(timeout=1)

            if message['type'] == 'complete':
                final_result = message['result']
                logger.info("Processing completed!")
                break
            elif message['type'] == 'error':
                logger.error(f"Processing error: {message['message']}")
                break
            elif message['type'] == 'progress':
                logger.info(f"[{message['stage']}] {message['percentage']}% - {message['message']}")

        except Exception as e:
            # Check if async result is ready
            if result.ready():
                try:
                    result.get(timeout=1)
                except Exception as task_error:
                    logger.error(f"Task error: {task_error}")
                break

    processing_time = time.time() - processing_start

    # Cleanup
    pool.close()
    pool.join()

    if final_result:
        # Calculate metrics
        total_processing_time = final_result['processing_time']
        processing_ratio = final_result['processing_ratio']

        # Results
        results = {
            "audio_file": "d.speakers.wav",
            "audio_duration": audio_duration,
            "initialization_time": init_time,
            "total_processing_time": total_processing_time,
            "processing_ratio": processing_ratio,
            "num_speakers_detected": final_result['num_speakers'],
            "num_segments": len(final_result['segments']),
            "peak_memory_mb": final_result.get('peak_memory_mb', 0),
            "num_workers": num_workers
        }

        # Print results
        print("\n" + "="*60)
        print("TESTE B: 3 WORKERS MULTIPROCESSING (Master Branch)")
        print("="*60)
        print(f"Audio file: {results['audio_file']}")
        print(f"Audio duration: {results['audio_duration']:.2f}s")
        print(f"Pool initialization time: {results['initialization_time']:.2f}s")
        print(f"Total processing time: {results['total_processing_time']:.2f}s")
        print(f"Processing ratio: {results['processing_ratio']:.2f}x")
        print(f"Speakers detected: {results['num_speakers_detected']}")
        print(f"Segments: {results['num_segments']}")
        print(f"Peak memory: {results['peak_memory_mb']:.2f} MB")
        print(f"Number of workers: {results['num_workers']}")
        print("="*60)

        return results
    else:
        logger.error("No final result received")
        return None

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    results = asyncio.run(test_multiprocessing_pool())
