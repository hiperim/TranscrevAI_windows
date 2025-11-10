"""
Simple performance test script for TranscrevAI
Tests single audio file and measures performance
"""

import asyncio
import time
import logging
from pathlib import Path
import torch
import psutil
import os

# Import services directly (simulating single-process architecture)
from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from src.audio_processing import AudioQualityAnalyzer
import librosa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_single_process():
    """Test with single process architecture (like commit 98619b4)"""

    # --- Dynamic Performance Tuning ---
    # This logic dynamically sets thread counts based on hardware detection for optimal performance.
    num_physical_cores = psutil.cpu_count(logical=False)
    if not num_physical_cores:
        num_physical_cores = os.cpu_count() or 4
        logger.warning(f"Could not detect physical cores. Falling back to logical cores: {num_physical_cores}")

    # Heuristic based on benchmark results:
    # - Give PyTorch (diarization) a small, fixed number of threads.
    # - Give OpenMP (transcription) the majority of the remaining cores.
    torch_threads = 2
    omp_threads = max(1, num_physical_cores - torch_threads)

    torch.set_num_threads(torch_threads)
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    logger.info(f"PERFORMANCE TUNING: Detected {num_physical_cores} physical cores. Settings: OMP_NUM_THREADS={omp_threads}, torch_threads={torch_threads}")

    # Load audio file
    audio_path = "data/recordings/d.speakers.wav"

    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        return

    # Get audio duration
    audio_duration = librosa.get_duration(path=audio_path)
    logger.info(f"Audio duration: {audio_duration:.2f}s")

    # Initialize services ONCE (single process)
    logger.info("Initializing services...")
    init_start = time.time()
    transcription_service = TranscriptionService(model_name="medium")
    diarization_service = PyannoteDiarizer(device="cpu")
    audio_quality_analyzer = AudioQualityAnalyzer()
    init_time = time.time() - init_start
    logger.info(f"Services initialized in {init_time:.2f}s")

    # Run transcription
    logger.info("Starting transcription...")
    transcription_start = time.time()
    transcription_result = await transcription_service.transcribe_with_enhancements(
        audio_path,
        word_timestamps=True
    )
    transcription_time = time.time() - transcription_start
    logger.info(f"Transcription completed in {transcription_time:.2f}s")

    # Run diarization
    logger.info("Starting diarization...")
    diarization_start = time.time()
    diarization_result = await diarization_service.diarize(
        audio_path,
        transcription_result.segments
    )
    diarization_time = time.time() - diarization_start
    logger.info(f"Diarization completed in {diarization_time:.2f}s")

    # Calculate metrics
    total_processing_time = transcription_time + diarization_time
    processing_ratio = total_processing_time / audio_duration

    # Results
    results = {
        "audio_file": "d.speakers.wav",
        "audio_duration": audio_duration,
        "initialization_time": init_time,
        "transcription_time": transcription_time,
        "diarization_time": diarization_time,
        "total_processing_time": total_processing_time,
        "processing_ratio": processing_ratio,
        "num_speakers_detected": diarization_result["num_speakers"],
        "num_segments": len(diarization_result["segments"])
    }

    # Print results
    print("\n" + "="*60)
    print("TESTE A: SINGLE PROCESS (Commit 98619b4)")
    print("="*60)
    print(f"Audio file: {results['audio_file']}")
    print(f"Audio duration: {results['audio_duration']:.2f}s")
    print(f"Initialization time: {results['initialization_time']:.2f}s")
    print(f"Transcription time: {results['transcription_time']:.2f}s")
    print(f"Diarization time: {results['diarization_time']:.2f}s")
    print(f"Total processing time: {results['total_processing_time']:.2f}s")
    print(f"Processing ratio: {results['processing_ratio']:.2f}x")
    print(f"Speakers detected: {results['num_speakers_detected']}")
    print(f"Segments: {results['num_segments']}")
    print("="*60)

    return results

if __name__ == "__main__":
    results = asyncio.run(test_single_process())
