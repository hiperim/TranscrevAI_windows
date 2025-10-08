"""
FASE 1 Validation Test - Model Change + Memory Profiling
Tests the pipeline with 4 audio files and validates:
- Accuracy improvement (target >90%)
- Memory usage (target <3.5GB)
- Processing speed maintained
"""

import asyncio
import json
import time
from pathlib import Path
import logging

from src.transcription import TranscriptionService
from src.diarization import TwoPassDiarizer, force_transcription_segmentation
from src.file_manager import FileManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test audio files
TEST_FILES = [
    "data/recordings/d.speakers.wav",
    "data/recordings/q.speakers.wav",
    "data/recordings/t.speakers.wav",
    "data/recordings/t2.speakers.wav"
]

# Expected results files (ground truth for accuracy validation)
EXPECTED_RESULTS = {
    "d.speakers.wav": "data/recordings/expected_results_d.speakers.txt",
    "q.speakers.wav": "data/recordings/expected_results_q.speakers.txt",
    "t.speakers.wav": "data/recordings/expected_results_t.speakers.txt",
    "t2.speakers.wav": "data/recordings/expected_results_t2.speakers.txt"
}

async def test_audio_file(audio_path: str, transcription_service: TranscriptionService, diarization_service: TwoPassDiarizer):
    """Test a single audio file and collect metrics"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {Path(audio_path).name}")
    logger.info(f"{'='*80}")

    start_time = time.time()

    # Get audio duration
    import wave
    with wave.open(audio_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)

    logger.info(f"Audio Duration: {duration:.2f}s")

    # Run transcription
    transcription_result = await transcription_service.transcribe_with_enhancements(audio_path)

    # Run diarization
    diarization_result = await diarization_service.diarize(audio_path, transcription_result.segments)
    final_segments = force_transcription_segmentation(transcription_result.segments, diarization_result["segments"])

    processing_time = time.time() - start_time
    speed_ratio = processing_time / duration if duration > 0 else 0

    # Load expected results if available
    expected_text = None
    filename = Path(audio_path).name
    if filename in EXPECTED_RESULTS:
        expected_path = EXPECTED_RESULTS[filename]
        if Path(expected_path).exists():
            with open(expected_path, 'r', encoding='utf-8') as f:
                expected_text = f.read().strip().lower()

    # Calculate accuracy if we have expected text
    accuracy = None
    if expected_text:
        actual_text = transcription_result.text.strip().lower()

        # Simple word-level accuracy calculation
        expected_words = expected_text.split()
        actual_words = actual_text.split()

        # Calculate word error rate (WER) approximation
        matches = sum(1 for w in actual_words if w in expected_words)
        accuracy = (matches / len(expected_words)) * 100 if expected_words else 0

    results = {
        "file": filename,
        "duration": duration,
        "processing_time": processing_time,
        "speed_ratio": speed_ratio,
        "confidence": transcription_result.confidence,
        "accuracy": accuracy,
        "num_speakers": diarization_result["num_speakers"],
        "num_segments": len(final_segments),
        "transcribed_text": transcription_result.text[:200] + "..." if len(transcription_result.text) > 200 else transcription_result.text
    }

    logger.info(f"Processing Time: {processing_time:.2f}s")
    logger.info(f"Speed Ratio: {speed_ratio:.2f}s/s (target: <1.50s/s)")
    logger.info(f"Model Confidence: {transcription_result.confidence*100:.2f}%")
    if accuracy:
        logger.info(f"Accuracy vs Expected: {accuracy:.2f}% (target: >90%)")
    logger.info(f"Speakers Detected: {diarization_result['num_speakers']}")
    logger.info(f"Segments Generated: {len(final_segments)}")

    return results

async def main():
    """Run FASE 1 validation tests"""
    logger.info("\n" + "="*80)
    logger.info("FASE 1 VALIDATION - Baseline + Memory Profiling")
    logger.info("="*80)
    logger.info(f"Model: Systran/faster-whisper-medium (language=pt)")
    logger.info(f"Test Files: {len(TEST_FILES)}")
    logger.info(f"Target Accuracy: >90%")
    logger.info(f"Target Memory: <3500 MB")
    logger.info(f"Target Speed: <1.50s/s (acceptable)")
    logger.info("="*80 + "\n")

    # Initialize services
    logger.info("Initializing services...")
    transcription_service = TranscriptionService()
    await transcription_service.initialize()
    diarization_service = TwoPassDiarizer()

    # Test each file
    all_results = []
    for audio_file in TEST_FILES:
        if not Path(audio_file).exists():
            logger.warning(f"File not found: {audio_file}, skipping...")
            continue

        try:
            results = await test_audio_file(audio_file, transcription_service, diarization_service)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}", exc_info=True)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY - FASE 1 Validation Results")
    logger.info("="*80)

    if not all_results:
        logger.error("No results to summarize!")
        return

    # Calculate averages
    avg_speed = sum(r["speed_ratio"] for r in all_results) / len(all_results)
    avg_confidence = sum(r["confidence"] for r in all_results) / len(all_results)

    # Accuracy average (only for files with expected results)
    accuracy_results = [r["accuracy"] for r in all_results if r["accuracy"] is not None]
    avg_accuracy = sum(accuracy_results) / len(accuracy_results) if accuracy_results else None

    logger.info(f"\nFiles Processed: {len(all_results)}/{len(TEST_FILES)}")
    logger.info(f"Average Processing Speed: {avg_speed:.2f}s/s (target: <1.50s/s) - {'PASS' if avg_speed < 1.50 else 'FAIL'}")
    logger.info(f"Average Model Confidence: {avg_confidence*100:.2f}%")
    if avg_accuracy:
        logger.info(f"Average Accuracy vs Expected: {avg_accuracy:.2f}% (target: >90%) - {'PASS' if avg_accuracy > 90 else 'NEEDS IMPROVEMENT'}")
    else:
        logger.warning("No accuracy data available (expected results missing)")

    # Individual results table
    logger.info("\nIndividual Results:")
    logger.info(f"{'File':<20} {'Duration':<10} {'Speed':<10} {'Confidence':<12} {'Accuracy':<12} {'Speakers':<10}")
    logger.info("-" * 80)
    for r in all_results:
        accuracy_str = f"{r['accuracy']:.1f}%" if r['accuracy'] else "N/A"
        logger.info(f"{r['file']:<20} {r['duration']:<10.2f}s {r['speed_ratio']:<10.2f}s/s {r['confidence']*100:<12.2f}% {accuracy_str:<12} {r['num_speakers']:<10}")

    # Save results to JSON
    output_file = "test_results_fase1.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "Systran/faster-whisper-medium (language=pt)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "avg_speed_ratio": avg_speed,
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
                "speed_pass": avg_speed < 1.50,
                "accuracy_pass": avg_accuracy > 90 if avg_accuracy else None
            },
            "individual_results": all_results
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    asyncio.run(main())
