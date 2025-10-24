"""
Systematic Optimization Testing for TranscrevAI
Tests transcription accuracy, diarization accuracy, and processing speed.

Usage:
    python tests/test_systematic_optimization.py [phase_name]

Logs are saved to tests/logs/ without overwriting previous results.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import librosa

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from tests.metrics import calculate_wer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
AUDIO_FILE = Path(__file__).parent.parent / "data" / "recordings" / "d.speakers.wav"
GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth" / "d_speakers.txt"
LOGS_DIR = Path(__file__).parent / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)


def load_ground_truth() -> str:
    """Load and return ground truth text."""
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_audio_duration() -> float:
    """Get duration of test audio file."""
    return librosa.get_duration(filename=str(AUDIO_FILE))


async def run_full_pipeline_test(
    phase_name: str,
    description: str,
    test_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run complete pipeline test: transcription + diarization + metrics.

    Args:
        phase_name: Identifier for this test phase
        description: Human-readable description of what's being tested
        test_config: Optional configuration overrides

    Returns:
        Complete metrics dictionary
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🧪 PHASE: {phase_name}")
    logger.info(f"📝 Description: {description}")
    logger.info(f"{'='*80}\n")

    # Load ground truth
    ground_truth = load_ground_truth()
    audio_duration = get_audio_duration()

    logger.info(f"📁 Audio file: {AUDIO_FILE}")
    logger.info(f"⏱️  Audio duration: {audio_duration:.3f}s")
    logger.info(f"📄 Ground truth: {len(ground_truth)} chars, {len(ground_truth.split())} words\n")

    # Initialize services
    logger.info("🔧 Initializing services...")
    init_start = time.time()

    transcription_service = TranscriptionService(model_name="medium", device="cpu")
    await transcription_service.initialize(
        compute_type=test_config.get("compute_type", "int8") if test_config else "int8"
    )

    embedding_batch_size = test_config.get("embedding_batch_size", 1) if test_config else 1
    diarizer = PyannoteDiarizer(embedding_batch_size=embedding_batch_size)

    init_time = time.time() - init_start
    logger.info(f"✅ Initialization complete: {init_time:.2f}s\n")

    # Run transcription
    logger.info("🎤 Running transcription...")
    transcription_start = time.time()

    transcription_result = await transcription_service.transcribe_with_enhancements(
        str(AUDIO_FILE),
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
        str(AUDIO_FILE),
        transcription_result.segments
    )

    diarization_time = time.time() - diarization_start
    speakers_detected = diarization_result['num_speakers']
    logger.info(f"✅ Diarization complete: {diarization_time:.2f}s")
    logger.info(f"👥 Speakers detected: {speakers_detected}\n")

    # Calculate metrics
    logger.info("📊 Calculating metrics...")

    # Transcription accuracy (WER)
    wer = calculate_wer(ground_truth, transcription_result.text)
    transcription_accuracy = (1 - wer) * 100

    # Diarization accuracy
    speakers_expected = 2  # Known for d_speakers.wav
    diarization_correct = speakers_detected == speakers_expected
    diarization_accuracy = "Correct" if diarization_correct else f"Wrong ({speakers_detected}/{speakers_expected})"

    # Processing speed
    total_time = init_time + transcription_time + diarization_time
    processing_ratio = total_time / audio_duration

    # Compile results
    results = {
        "phase": phase_name,
        "description": description,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "configuration": {
            "model": "medium",
            "device": "cpu",
            "compute_type": test_config.get("compute_type", "int8") if test_config else "int8",
            "ptbr_corrections_count": len(transcription_service.ptbr_corrections),
            "test_config": test_config or {}
        },
        "files": {
            "audio": str(AUDIO_FILE),
            "ground_truth": str(GROUND_TRUTH_FILE),
            "audio_duration_sec": audio_duration
        },
        "metrics": {
            "transcription": {
                "wer": float(wer),
                "accuracy_percent": float(transcription_accuracy),
                "text": transcription_result.text,
                "word_count": transcription_result.word_count,
                "confidence": float(transcription_result.confidence)
            },
            "diarization": {
                "speakers_detected": int(speakers_detected),
                "speakers_expected": int(speakers_expected),
                "accuracy": diarization_accuracy,
                "correct": bool(diarization_correct),
                "segments_with_speakers": len(diarization_result['segments'])
            },
            "performance": {
                "init_time_sec": float(init_time),
                "transcription_time_sec": float(transcription_time),
                "diarization_time_sec": float(diarization_time),
                "total_time_sec": float(total_time),
                "processing_ratio": f"{processing_ratio:.2f}x",
                "processing_ratio_numeric": float(processing_ratio)
            }
        },
        "ground_truth": {
            "text": ground_truth,
            "word_count": len(ground_truth.split())
        }
    }

    # Display summary
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 RESULTS SUMMARY - {phase_name}")
    logger.info(f"{'='*80}")
    logger.info(f"")
    logger.info(f"🎯 TRANSCRIPTION ACCURACY: {transcription_accuracy:.2f}%")
    logger.info(f"   - WER: {wer:.4f}")
    logger.info(f"   - Confidence: {transcription_result.confidence:.4f}")
    logger.info(f"")
    logger.info(f"👥 DIARIZATION ACCURACY: {diarization_accuracy}")
    logger.info(f"   - Detected: {speakers_detected} speakers")
    logger.info(f"   - Expected: {speakers_expected} speakers")
    logger.info(f"")
    logger.info(f"⚡ PROCESSING SPEED: {processing_ratio:.2f}x real-time")
    logger.info(f"   - Init: {init_time:.2f}s")
    logger.info(f"   - Transcription: {transcription_time:.2f}s")
    logger.info(f"   - Diarization: {diarization_time:.2f}s")
    logger.info(f"   - Total: {total_time:.2f}s (audio: {audio_duration:.2f}s)")
    logger.info(f"")
    logger.info(f"{'='*80}\n")

    # Save results to file (unique filename with timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOGS_DIR / f"{timestamp}_{phase_name}_results.json"

    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Results saved to: {log_filename}\n")

    # Unload model to free memory
    await transcription_service.unload_model()

    return results


async def main():
    """Main test runner."""
    import sys

    # Get phase name from command line or use default
    if len(sys.argv) > 1:
        phase_name = sys.argv[1]
        description = f"Custom test phase: {phase_name}"
    else:
        # Default: test current version
        phase_name = "current_baseline"
        description = "Baseline test of current transcription.py configuration"

    # Configure test based on phase name
    test_config = {}
    if "int8_float16" in phase_name:
        test_config["compute_type"] = "int8_float16"
        description += " with int8_float16 compute type"
    elif "float16" in phase_name:
        test_config["compute_type"] = "float16"
        description += " with float16 compute type"
    elif "batch" in phase_name:
        # Extract batch size from phase name (e.g., test4_batch8)
        import re
        match = re.search(r'batch(\d+)', phase_name)
        if match:
            batch_size = int(match.group(1))
            test_config["embedding_batch_size"] = batch_size
            description += f" with embedding_batch_size={batch_size}"

    logger.info(f"\n🚀 Starting systematic optimization test")
    logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        results = await run_full_pipeline_test(
            phase_name=phase_name,
            description=description,
            test_config=test_config
        )

        # Check if targets are met
        accuracy = results['metrics']['transcription']['accuracy_percent']
        ratio = results['metrics']['performance']['processing_ratio_numeric']
        speakers_correct = results['metrics']['diarization']['correct']

        logger.info(f"🎯 TARGET ANALYSIS:")
        logger.info(f"   Accuracy ≥92%: {'✅ PASS' if accuracy >= 92 else f'❌ FAIL ({accuracy:.2f}%)'}")
        logger.info(f"   Ratio <1.9x: {'✅ PASS' if ratio < 1.9 else f'❌ FAIL ({ratio:.2f}x)'}")
        logger.info(f"   Speakers 2/2: {'✅ PASS' if speakers_correct else '❌ FAIL'}")
        logger.info(f"")

        if accuracy >= 92 and ratio < 1.9 and speakers_correct:
            logger.info(f"🎉 ALL TARGETS MET! This configuration is ready for production.")
        else:
            logger.info(f"⚠️  Some targets not met. Further optimization needed.")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)
        raise

    logger.info(f"\n✅ Test complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
