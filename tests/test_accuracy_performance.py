"""
Consolidated accuracy and performance testing for TranscrevAI.

Measures:
1. Transcription Accuracy (WER - Word Error Rate)
2. Diarization Accuracy (speaker count validation)
3. Processing Speed Ratio (processing_time / audio_duration)
4. CPU Profiling (cProfile)
5. Memory Usage (peak RAM)

Tests multiple audio files to establish robust baseline metrics.
"""

import pytest
import asyncio
import cProfile
import pstats
import time
import psutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import librosa

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from tests.metrics import calculate_dual_wer
from tests.utils import MemoryMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
AUDIO_FILES = [
    {
        "name": "d.speakers.wav",
        "path": Path(__file__).parent.parent / "data" / "recordings" / "d.speakers.wav",
        "ground_truth": Path(__file__).parent / "ground_truth" / "d_speakers.txt",
        "expected_speakers": 2
    },
    {
        "name": "q.speakers.wav",
        "path": Path(__file__).parent.parent / "data" / "recordings" / "q.speakers.wav",
        "ground_truth": Path(__file__).parent / "ground_truth" / "q_speakers.txt",
        "expected_speakers": 4
    }
]

LOGS_DIR = Path(__file__).parent / "logs" / "accuracy_performance"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PROF_DIR = Path(__file__).parent / "prof"
PROF_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def transcription_service():
    """Shared TranscriptionService for all tests"""
    return TranscriptionService(model_name="medium", device="cpu")


@pytest.fixture(scope="module")
def diarizer():
    """Shared PyannoteDiarizer for all tests"""
    return PyannoteDiarizer(device="cpu")


def load_ground_truth(file_path: Path) -> str:
    """Load and return ground truth text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file."""
    return librosa.get_duration(path=str(audio_path))


async def _process_single_audio(
    audio_config: Dict[str, Any],
    transcription_service: TranscriptionService,
    diarizer: PyannoteDiarizer
) -> Dict[str, Any]:
    """
    Test a single audio file and return comprehensive metrics.
    """
    audio_name = audio_config["name"]
    audio_path = audio_config["path"]
    ground_truth_path = audio_config["ground_truth"]
    expected_speakers = audio_config["expected_speakers"]

    logger.info(f"\n{'='*80}")
    logger.info(f"🎵 TESTING: {audio_name}")
    logger.info(f"{'='*80}\n")

    # Validate files exist
    if not audio_path.exists():
        pytest.skip(f"Audio file not found: {audio_path}")
    if not ground_truth_path.exists():
        pytest.skip(f"Ground truth not found: {ground_truth_path}")

    # Load ground truth and audio info
    ground_truth = load_ground_truth(ground_truth_path)
    audio_duration = get_audio_duration(audio_path)

    logger.info(f"📁 Audio file: {audio_path}")
    logger.info(f"⏱️  Audio duration: {audio_duration:.3f}s")
    logger.info(f"📄 Ground truth: {len(ground_truth)} chars, {len(ground_truth.split())} words\n")

    # Use context manager for memory monitoring
    with MemoryMonitor() as memory_monitor:
        # Run transcription
        logger.info("🎤 Running transcription...")
        transcription_start = time.time()

        transcription_result = await transcription_service.transcribe_with_enhancements(
            str(audio_path),
            word_timestamps=True
        )

        transcription_time = time.time() - transcription_start
        logger.info(f"✅ Transcription complete: {transcription_time:.2f}s")
        logger.info(f"📝 Transcribed text: {transcription_result.text[:100]}...")
        logger.info(f"📊 Confidence: {transcription_result.confidence:.4f}\n")

        # Run diarization
        logger.info("👥 Running diarization...")
        diarization_start = time.time()

        diarization_result = await diarizer.diarize(
            str(audio_path),
            transcription_result.segments
        )

        diarization_time = time.time() - diarization_start
        speakers_detected = diarization_result['num_speakers']
        logger.info(f"✅ Diarization complete: {diarization_time:.2f}s")
        logger.info(f"👥 Speakers detected: {speakers_detected}\n")

    # Get peak memory after context exits
    peak_memory_mb = memory_monitor.peak_memory_mb

    # Save peak memory to prof directory
    peak_memory_file = PROF_DIR / "peak_memory.txt"
    with open(peak_memory_file, 'w') as f:
        f.write(f"{peak_memory_mb:.2f}")

    # Calculate metrics
    dual_wer = calculate_dual_wer(ground_truth, transcription_result.text)
    wer = dual_wer['wer_traditional']
    wer_normalized = dual_wer['wer_normalized']
    transcription_accuracy = dual_wer['accuracy_traditional_percent']
    transcription_accuracy_normalized = dual_wer['accuracy_normalized_percent']

    diarization_correct = speakers_detected == expected_speakers
    diarization_accuracy_pct = 100.0 if diarization_correct else 0.0

    total_time = transcription_time + diarization_time
    processing_ratio = total_time / audio_duration

    # Compile results
    results = {
        "audio_name": audio_name,
        "audio_duration_s": audio_duration,
        "transcription_time_s": transcription_time,
        "diarization_time_s": diarization_time,
        "total_processing_time_s": total_time,
        "processing_ratio": processing_ratio,
        "wer_traditional": wer,
        "wer_normalized": wer_normalized,
        "transcription_accuracy_pct": transcription_accuracy,
        "transcription_accuracy_normalized_pct": transcription_accuracy_normalized,
        "speakers_expected": expected_speakers,
        "speakers_detected": speakers_detected,
        "diarization_correct": diarization_correct,
        "diarization_accuracy_pct": diarization_accuracy_pct,
        "peak_memory_mb": peak_memory_mb,
        "confidence": transcription_result.confidence
    }

    # Log results
    logger.info(f"\n📊 RESULTS for {audio_name}:")
    logger.info(f"  WER (Traditional): {wer:.4f} ({transcription_accuracy:.2f}% accuracy)")
    logger.info(f"  WER (Normalized): {wer_normalized:.4f} ({transcription_accuracy_normalized:.2f}% accuracy)")
    logger.info(f"  Diarization: {speakers_detected}/{expected_speakers} speakers ({diarization_accuracy_pct:.0f}% accuracy)")
    logger.info(f"  Processing Ratio: {processing_ratio:.2f}x")
    logger.info(f"  Peak Memory: {peak_memory_mb:.2f} MB\n")

    return results


@pytest.mark.asyncio
@pytest.mark.accuracy
async def test_transcription_diarization_accuracy():
    """
    Test transcription and diarization accuracy across multiple audio files.
    """
    logger.info("\n" + "="*80)
    logger.info("🚀 STARTING ACCURACY & PERFORMANCE TEST")
    logger.info("="*80 + "\n")

    # Initialize services
    transcription_service = TranscriptionService(model_name="medium", device="cpu")
    diarizer = PyannoteDiarizer(device="cpu")

    # Test each audio file
    results_list = []
    for audio_config in AUDIO_FILES:
        result = await _process_single_audio(
            audio_config,
            transcription_service,
            diarizer
        )
        results_list.append(result)

    # Calculate averages
    avg_wer_traditional = sum(r["wer_traditional"] for r in results_list) / len(results_list)
    avg_wer_normalized = sum(r["wer_normalized"] for r in results_list) / len(results_list)
    avg_transcription_accuracy = sum(r["transcription_accuracy_pct"] for r in results_list) / len(results_list)
    avg_transcription_accuracy_normalized = sum(r["transcription_accuracy_normalized_pct"] for r in results_list) / len(results_list)
    avg_diarization_accuracy = sum(r["diarization_accuracy_pct"] for r in results_list) / len(results_list)
    avg_processing_ratio = sum(r["processing_ratio"] for r in results_list) / len(results_list)
    avg_peak_memory = sum(r["peak_memory_mb"] for r in results_list) / len(results_list)

    # Log summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 SUMMARY ACROSS ALL AUDIO FILES")
    logger.info(f"{'='*80}")
    logger.info(f"  Average WER (Traditional): {avg_wer_traditional:.4f} ({avg_transcription_accuracy:.2f}% accuracy)")
    logger.info(f"  Average WER (Normalized): {avg_wer_normalized:.4f} ({avg_transcription_accuracy_normalized:.2f}% accuracy)")
    logger.info(f"  Average Diarization Accuracy: {avg_diarization_accuracy:.2f}%")
    logger.info(f"  Average Processing Ratio: {avg_processing_ratio:.2f}x")
    logger.info(f"  Average Peak Memory: {avg_peak_memory:.2f} MB")
    logger.info(f"{'='*80}\n")

    # Save results to JSON
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = LOGS_DIR / f"accuracy_results_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "individual_results": results_list,
        "averages": {
            "wer_traditional": avg_wer_traditional,
            "wer_normalized": avg_wer_normalized,
            "transcription_accuracy_pct": avg_transcription_accuracy,
            "transcription_accuracy_normalized_pct": avg_transcription_accuracy_normalized,
            "diarization_accuracy_pct": avg_diarization_accuracy,
            "processing_ratio": avg_processing_ratio,
            "peak_memory_mb": avg_peak_memory
        }
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    logger.info(f"💾 Results saved to: {results_file}")

    # Assertions for CI/CD
    assert avg_transcription_accuracy_normalized >= 85.0, f"Transcription accuracy too low: {avg_transcription_accuracy_normalized:.2f}%"
    assert avg_diarization_accuracy >= 50.0, f"Diarization accuracy too low: {avg_diarization_accuracy:.2f}%"
    assert avg_processing_ratio <= 2.0, f"Processing too slow: {avg_processing_ratio:.2f}x"
    assert avg_peak_memory <= 3072, f"Memory usage too high: {avg_peak_memory:.2f} MB"


@pytest.mark.profiling
def test_cpu_profiling():
    """
    Run CPU profiling on the full pipeline.
    """
    logger.info("\n🔬 Starting CPU profiling...")

    profiler = cProfile.Profile()
    profiler.enable()

    # Run the accuracy test
    asyncio.run(test_transcription_diarization_accuracy())

    profiler.disable()

    # Save profiling stats
    stats_file = PROF_DIR / "full_profile.stats"
    profiler.dump_stats(str(stats_file))

    # Print top 20 time-consuming functions
    logger.info("\n📊 Top 20 time-consuming functions:")
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    logger.info(f"\n💾 Full profiling data saved to: {stats_file}")
    logger.info(f"   View with: python -m pstats {stats_file}")


if __name__ == "__main__":
    # Run tests directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        test_cpu_profiling()
    else:
        asyncio.run(test_transcription_diarization_accuracy())
