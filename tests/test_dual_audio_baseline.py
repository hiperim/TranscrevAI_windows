"""
Dual Audio Baseline Testing for TranscrevAI
Tests transcription accuracy, diarization accuracy, and processing speed on TWO audio files
to establish a robust baseline and avoid overfitting.

Usage:
    python tests/test_dual_audio_baseline.py [phase_name]

Audio files tested:
    - d.speakers.wav (21.056s, 50 words)
    - q.speakers.wav (14.507s, ~50 words)

Results include:
    - Individual metrics for EACH audio file
    - Average metrics across BOTH files

Logs are saved to tests/logs/dual_baseline/ without overwriting previous results.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import librosa

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from tests.metrics import calculate_wer, calculate_dual_wer

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

LOGS_DIR = Path(__file__).parent / "logs" / "dual_baseline"

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth(file_path: Path) -> str:
    """Load and return ground truth text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file."""
    return librosa.get_duration(path=str(audio_path))


async def test_single_audio(
    audio_config: Dict[str, Any],
    transcription_service: TranscriptionService,
    diarizer: PyannoteDiarizer,
    test_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Test a single audio file and return metrics.
    """
    audio_name = audio_config["name"]
    audio_path = audio_config["path"]
    ground_truth_path = audio_config["ground_truth"]
    expected_speakers = audio_config["expected_speakers"]

    logger.info(f"\n{'='*80}")
    logger.info(f"🎵 TESTING: {audio_name}")
    logger.info(f"{'='*80}\n")

    # Load ground truth and audio info
    ground_truth = load_ground_truth(ground_truth_path)
    audio_duration = get_audio_duration(audio_path)

    logger.info(f"📁 Audio file: {audio_path}")
    logger.info(f"⏱️  Audio duration: {audio_duration:.3f}s")
    logger.info(f"📄 Ground truth: {len(ground_truth)} chars, {len(ground_truth.split())} words\n")

    # Run transcription
    logger.info("🎤 Running transcription...")
    transcription_start = time.time()

    # Get whisper_params from test_config if provided
    whisper_params = test_config.get("whisper_params") if test_config else None

    transcription_result = await transcription_service.transcribe_with_enhancements(
        str(audio_path),
        word_timestamps=True,
        whisper_params=whisper_params
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

    # Calculate metrics (both traditional and normalized WER)
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
        "audio_path": str(audio_path),
        "audio_duration_sec": audio_duration,
        "ground_truth": {
            "text": ground_truth,
            "word_count": len(ground_truth.split())
        },
        "transcription": {
            "wer": float(wer),
            "wer_normalized": float(wer_normalized),
            "accuracy_percent": float(transcription_accuracy),
            "accuracy_normalized_percent": float(transcription_accuracy_normalized),
            "text": transcription_result.text,
            "word_count": transcription_result.word_count,
            "confidence": float(transcription_result.confidence),
            "time_sec": float(transcription_time)
        },
        "diarization": {
            "speakers_detected": int(speakers_detected),
            "speakers_expected": int(expected_speakers),
            "correct": bool(diarization_correct),
            "accuracy_percent": float(diarization_accuracy_pct),
            "segments_with_speakers": len(diarization_result['segments']),
            "time_sec": float(diarization_time)
        },
        "performance": {
            "total_time_sec": float(total_time),
            "processing_ratio": f"{processing_ratio:.2f}x",
            "processing_ratio_numeric": float(processing_ratio)
        }
    }

    # Display summary
    logger.info(f"📊 RESULTS for {audio_name}:")
    logger.info(f"   🎯 Transcription Accuracy (Traditional): {transcription_accuracy:.2f}%")
    logger.info(f"   🎯 Transcription Accuracy (Normalized): {transcription_accuracy_normalized:.2f}%")
    logger.info(f"   👥 Diarization Accuracy: {diarization_accuracy_pct:.0f}% ({speakers_detected}/{expected_speakers} speakers)")
    logger.info(f"   ⚡ Processing Speed: {processing_ratio:.2f}x")
    logger.info(f"")

    return results


async def run_dual_audio_baseline(
    phase_name: str,
    description: str,
    test_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run complete dual audio baseline test.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🧪 DUAL AUDIO BASELINE TEST")
    logger.info(f"📝 Phase: {phase_name}")
    logger.info(f"📋 Description: {description}")
    logger.info(f"{'='*80}\n")

    # Initialize services ONCE
    logger.info("🔧 Initializing services...")
    init_start = time.time()

    transcription_service = TranscriptionService(model_name="medium", device="cpu")
    await transcription_service.initialize(
        compute_type=test_config.get("compute_type", "int8") if test_config else "int8"
    )

    embedding_batch_size = test_config.get("embedding_batch_size", 8) if test_config else 8
    diarizer = PyannoteDiarizer(embedding_batch_size=embedding_batch_size)

    init_time = time.time() - init_start
    logger.info(f"✅ Initialization complete: {init_time:.2f}s\n")

    # Test each audio file
    individual_results = []
    for audio_config in AUDIO_FILES:
        result = await test_single_audio(
            audio_config,
            transcription_service,
            diarizer,
            test_config
        )
        individual_results.append(result)

    # Calculate averages
    avg_transcription_accuracy = sum(r["transcription"]["accuracy_percent"] for r in individual_results) / len(individual_results)
    avg_transcription_accuracy_normalized = sum(r["transcription"]["accuracy_normalized_percent"] for r in individual_results) / len(individual_results)
    avg_transcription_wer = sum(r["transcription"]["wer"] for r in individual_results) / len(individual_results)
    avg_transcription_wer_normalized = sum(r["transcription"]["wer_normalized"] for r in individual_results) / len(individual_results)
    avg_confidence = sum(r["transcription"]["confidence"] for r in individual_results) / len(individual_results)

    avg_diarization_accuracy = sum(r["diarization"]["accuracy_percent"] for r in individual_results) / len(individual_results)

    avg_processing_ratio = sum(r["performance"]["processing_ratio_numeric"] for r in individual_results) / len(individual_results)

    # Compile final results
    final_results = {
        "phase": phase_name,
        "description": description,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "configuration": {
            "model": "medium",
            "device": "cpu",
            "compute_type": test_config.get("compute_type", "int8") if test_config else "int8",
            "ptbr_corrections_count": len(transcription_service.ptbr_corrections),
            "embedding_batch_size": embedding_batch_size,
            "test_config": test_config or {}
        },
        "initialization_time_sec": float(init_time),
        "individual_results": individual_results,
        "averages": {
            "transcription_accuracy_percent": float(avg_transcription_accuracy),
            "transcription_accuracy_normalized_percent": float(avg_transcription_accuracy_normalized),
            "transcription_wer": float(avg_transcription_wer),
            "transcription_wer_normalized": float(avg_transcription_wer_normalized),
            "transcription_confidence": float(avg_confidence),
            "diarization_accuracy_percent": float(avg_diarization_accuracy),
            "processing_ratio_numeric": float(avg_processing_ratio),
            "processing_ratio": f"{avg_processing_ratio:.2f}x"
        }
    }

    # Display final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 FINAL RESULTS - DUAL AUDIO BASELINE")
    logger.info(f"{'='*80}")
    logger.info(f"")
    logger.info(f"📋 INDIVIDUAL RESULTS:")
    logger.info(f"")
    for result in individual_results:
        logger.info(f"   {result['audio_name']}:")
        logger.info(f"      Transcription Accuracy (Traditional): {result['transcription']['accuracy_percent']:.2f}%")
        logger.info(f"      Transcription Accuracy (Normalized): {result['transcription']['accuracy_normalized_percent']:.2f}%")
        logger.info(f"      Diarization Accuracy: {result['diarization']['accuracy_percent']:.0f}%")
        logger.info(f"      Processing Ratio: {result['performance']['processing_ratio']}")
        logger.info(f"")

    logger.info(f"📊 AVERAGE RESULTS:")
    logger.info(f"")
    logger.info(f"   🎯 TRANSCRIPTION:")
    logger.info(f"      Average Accuracy (Traditional): {avg_transcription_accuracy:.2f}%")
    logger.info(f"      Average Accuracy (Normalized): {avg_transcription_accuracy_normalized:.2f}%")
    logger.info(f"      Average WER (Traditional): {avg_transcription_wer:.4f}")
    logger.info(f"      Average WER (Normalized): {avg_transcription_wer_normalized:.4f}")
    logger.info(f"      Average Confidence: {avg_confidence:.4f}")
    logger.info(f"")
    logger.info(f"   👥 DIARIZATION:")
    logger.info(f"      Average Accuracy: {avg_diarization_accuracy:.0f}%")
    logger.info(f"")
    logger.info(f"   ⚡ PROCESSING SPEED:")
    logger.info(f"      Average Ratio: {avg_processing_ratio:.2f}x real-time")
    logger.info(f"")
    logger.info(f"{'='*80}\n")

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOGS_DIR / f"{timestamp}_{phase_name}_results.json"

    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Results saved to: {log_filename}\n")

    # Unload model to free memory (synchronous call)
    transcription_service.unload_model()

    return final_results


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
        description = "Dual audio baseline test of current configuration"

    logger.info(f"\n🚀 Starting dual audio baseline test")
    logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        results = await run_dual_audio_baseline(
            phase_name=phase_name,
            description=description,
            test_config={}
        )

        # Check if targets are met
        avg_transcription_accuracy = results['averages']['transcription_accuracy_percent']
        avg_transcription_accuracy_normalized = results['averages']['transcription_accuracy_normalized_percent']
        avg_diarization_accuracy = results['averages']['diarization_accuracy_percent']
        avg_ratio = results['averages']['processing_ratio_numeric']

        # Normalized accuracy target: 92% with ±1% tolerance (91% is acceptable)
        normalized_target = 92.0
        normalized_tolerance = 1.0
        normalized_passes = avg_transcription_accuracy_normalized >= (normalized_target - normalized_tolerance)
        normalized_gap = normalized_target - avg_transcription_accuracy_normalized

        logger.info(f"🎯 TARGET ANALYSIS:")
        logger.info(f"   Transcription Accuracy (Normalized) ≥92% (±1%): {'✅ PASS' if normalized_passes else f'❌ FAIL'} ({avg_transcription_accuracy_normalized:.2f}%) - Gap: {normalized_gap:.2f}pp")
        logger.info(f"   Transcription Accuracy (Traditional) ≥72%: {'✅ PASS' if avg_transcription_accuracy >= 72 else f'❌ FAIL ({avg_transcription_accuracy:.2f}%)'}")
        logger.info(f"   Diarization Accuracy 100%: {'✅ PASS' if avg_diarization_accuracy == 100 else f'❌ FAIL ({avg_diarization_accuracy:.0f}%)'}")
        logger.info(f"   Processing Ratio <1.9x: {'✅ PASS' if avg_ratio < 1.9 else f'❌ FAIL ({avg_ratio:.2f}x)'}")
        logger.info(f"")

        if normalized_passes and avg_diarization_accuracy == 100 and avg_ratio < 1.9:
            logger.info(f"🎉 ALL TARGETS MET! This configuration is ready for production.")
            logger.info(f"   Normalized transcription accuracy: {avg_transcription_accuracy_normalized:.2f}% (within ±1% of 92% target)")
        else:
            logger.info(f"⚠️  Some targets not met. Further optimization needed.")
            if not normalized_passes:
                logger.info(f"   Gap to 92% normalized transcription (with ±1% tolerance): {max(0, normalized_gap - normalized_tolerance):.2f} percentage points")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)
        raise

    logger.info(f"\n✅ Dual audio baseline test complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
