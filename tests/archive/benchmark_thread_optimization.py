"""
Automated Thread Optimization Benchmark for TranscrevAI
Tests different OMP_NUM_THREADS and torch thread configurations
to find optimal CPU utilization without oversubscription
"""

import asyncio
import time
import logging
import os
import json
from pathlib import Path
from datetime import datetime
import torch

# Import services
from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
import librosa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configurations
TEST_CONFIGS = [
    # OMP_NUM_THREADS, torch_threads, description
    (1, 1, "minimal_1t_each"),
    (2, 2, "conservative_2t_each"),
    (4, 2, "balanced_4t_whisper_2t_torch"),
    (4, 4, "moderate_4t_each"),
    (6, 2, "whisper_heavy_6t_2t"),
    (6, 4, "balanced_6t_4t"),
    (8, 2, "whisper_heavy_8t_2t"),
    (8, 4, "balanced_8t_4t"),
    (8, 8, "aggressive_8t_each"),
    (12, 2, "whisper_max_12t_2t"),
    (12, 4, "whisper_max_12t_4t"),
    (14, 1, "all_cores_whisper_14t_1t"),
]

async def benchmark_single_config(omp_threads: int, torch_threads: int, config_name: str):
    """Benchmark a single thread configuration"""

    audio_path = "data/recordings/d.speakers.wav"

    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        return None

    # Get audio duration
    audio_duration = librosa.get_duration(path=audio_path)

    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {config_name}")
    logger.info(f"OMP_NUM_THREADS={omp_threads}, torch.set_num_threads({torch_threads})")
    logger.info(f"{'='*80}")

    # Set environment variable for OpenMP (faster-whisper/CTranslate2)
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    # Set PyTorch threads (pyannote)
    torch.set_num_threads(torch_threads)

    try:
        # Initialize services
        init_start = time.time()
        transcription_service = TranscriptionService(model_name="medium", device="cpu")
        diarization_service = PyannoteDiarizer(device="cpu")
        init_time = time.time() - init_start

        # Transcription
        logger.info("Starting transcription...")
        trans_start = time.time()
        transcription_result = await transcription_service.transcribe_with_enhancements(
            audio_path,
            word_timestamps=True
        )
        trans_time = time.time() - trans_start
        logger.info(f"Transcription: {trans_time:.2f}s")

        # Diarization
        logger.info("Starting diarization...")
        diar_start = time.time()
        diarization_result = await diarization_service.diarize(
            audio_path,
            transcription_result.segments
        )
        diar_time = time.time() - diar_start
        logger.info(f"Diarization: {diar_time:.2f}s")

        # Calculate metrics
        total_time = trans_time + diar_time
        processing_ratio = total_time / audio_duration

        result = {
            "config_name": config_name,
            "omp_threads": omp_threads,
            "torch_threads": torch_threads,
            "audio_duration": audio_duration,
            "initialization_time": init_time,
            "transcription_time": trans_time,
            "diarization_time": diar_time,
            "total_processing_time": total_time,
            "processing_ratio": processing_ratio,
            "speakers_detected": diarization_result["num_speakers"],
            "segments": len(diarization_result["segments"])
        }

        logger.info(f"Result: {processing_ratio:.2f}x processing ratio")
        logger.info(f"Total: {total_time:.2f}s for {audio_duration:.2f}s audio")

        return result

    except Exception as e:
        logger.error(f"Benchmark failed for {config_name}: {e}", exc_info=True)
        return {
            "config_name": config_name,
            "omp_threads": omp_threads,
            "torch_threads": torch_threads,
            "error": str(e)
        }
    finally:
        # Cleanup
        if 'transcription_service' in locals() and transcription_service.model:
            await transcription_service.unload_model()
        del transcription_service, diarization_service
        import gc
        gc.collect()

async def run_all_benchmarks():
    """Run all benchmark configurations"""

    results = []

    logger.info(f"\n{'#'*80}")
    logger.info(f"# THREAD OPTIMIZATION BENCHMARK - TranscrevAI")
    logger.info(f"# Testing {len(TEST_CONFIGS)} configurations")
    logger.info(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*80}\n")

    for i, (omp_threads, torch_threads, config_name) in enumerate(TEST_CONFIGS, 1):
        logger.info(f"\n[{i}/{len(TEST_CONFIGS)}] Configuration: {config_name}")

        result = await benchmark_single_config(omp_threads, torch_threads, config_name)

        if result:
            results.append(result)

        # Small delay between tests to allow system to stabilize
        await asyncio.sleep(2)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmarks/thread_optimization_{timestamp}.json"

    os.makedirs("benchmarks", exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "test_file": "d.speakers.wav",
            "configurations_tested": len(TEST_CONFIGS),
            "results": results
        }, f, indent=2)

    # Analyze and print summary
    print_summary(results, output_file)

def print_summary(results, output_file):
    """Print benchmark summary"""

    # Filter successful results
    successful = [r for r in results if "error" not in r]

    if not successful:
        logger.error("No successful benchmarks!")
        return

    # Sort by processing ratio
    successful.sort(key=lambda x: x["processing_ratio"])

    logger.info(f"\n{'='*100}")
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info(f"{'='*100}")
    logger.info(f"{'Rank':<6} {'Config':<30} {'OMP':<5} {'Torch':<6} {'Ratio':<8} {'Trans(s)':<10} {'Diar(s)':<10} {'Total(s)':<10}")
    logger.info(f"{'-'*100}")

    for i, result in enumerate(successful, 1):
        logger.info(
            f"{i:<6} {result['config_name']:<30} {result['omp_threads']:<5} "
            f"{result['torch_threads']:<6} {result['processing_ratio']:<8.2f} "
            f"{result['transcription_time']:<10.2f} {result['diarization_time']:<10.2f} "
            f"{result['total_processing_time']:<10.2f}"
        )

    logger.info(f"{'-'*100}")

    # Best result
    best = successful[0]
    logger.info(f"\n🏆 BEST CONFIGURATION:")
    logger.info(f"   Config: {best['config_name']}")
    logger.info(f"   OMP_NUM_THREADS: {best['omp_threads']}")
    logger.info(f"   torch.set_num_threads: {best['torch_threads']}")
    logger.info(f"   Processing Ratio: {best['processing_ratio']:.2f}x")
    logger.info(f"   Total Time: {best['total_processing_time']:.2f}s")

    # Improvement from worst
    worst = successful[-1]
    improvement = ((worst['processing_ratio'] - best['processing_ratio']) / worst['processing_ratio']) * 100
    logger.info(f"\n📈 IMPROVEMENT:")
    logger.info(f"   Best vs Worst: {improvement:.1f}% faster")
    logger.info(f"   Worst config: {worst['config_name']} ({worst['processing_ratio']:.2f}x)")

    logger.info(f"\n💾 Full results saved to: {output_file}")
    logger.info(f"{'='*100}\n")

if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
