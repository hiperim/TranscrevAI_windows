"""
Systematic Performance Testing with Detailed Logging
Tests each optimization phase and logs all metrics separately
"""

import asyncio
import time
import logging
import os
import json
from pathlib import Path
from datetime import datetime
import sys

# Import services
from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
import librosa

# Create logs directory
LOGS_DIR = Path("tests/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Create timestamp for this test run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Configure logging
def setup_logging(phase_name: str):
    """Setup logging for a specific test phase"""
    log_file = LOGS_DIR / f"{TIMESTAMP}_{phase_name}.log"

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create new handlers
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"=" * 80)
    logger.info(f"SYSTEMATIC TEST - PHASE: {phase_name}")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"=" * 80)

    return logger, log_file

def save_results(phase_name: str, results: dict):
    """Save results to JSON file"""
    results_file = LOGS_DIR / f"{TIMESTAMP}_{phase_name}_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results_file

async def test_phase_baseline():
    """Phase 0: Current baseline (before any changes)"""
    phase_name = "phase0_baseline"
    logger, log_file = setup_logging(phase_name)

    logger.info("PHASE 0: BASELINE TEST (Current Configuration)")
    logger.info("Configuration:")
    logger.info("  - beam_size: 5 (current)")
    logger.info("  - best_of: 5 (current)")
    logger.info("  - No optimization thresholds")

    audio_path = "data/recordings/d.speakers.wav"
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        return None

    audio_duration = librosa.get_duration(path=audio_path)
    logger.info(f"Audio duration: {audio_duration:.2f}s")

    # Initialize services
    logger.info("Initializing services...")
    init_start = time.time()
    transcription_service = TranscriptionService(model_name="medium", device="cpu")
    diarization_service = PyannoteDiarizer(device="cpu")
    init_time = time.time() - init_start
    logger.info(f"Services initialized in {init_time:.2f}s")

    # Transcription
    logger.info("Starting transcription...")
    trans_start = time.time()
    transcription_result = await transcription_service.transcribe_with_enhancements(
        audio_path,
        word_timestamps=True
    )
    trans_time = time.time() - trans_start
    logger.info(f"Transcription completed in {trans_time:.2f}s")
    logger.info(f"Transcription text preview: {transcription_result.text[:200]}...")

    # Diarization
    logger.info("Starting diarization...")
    diar_start = time.time()
    diarization_result = await diarization_service.diarize(
        audio_path,
        transcription_result.segments
    )
    diar_time = time.time() - diar_start
    logger.info(f"Diarization completed in {diar_time:.2f}s")
    logger.info(f"Speakers detected: {diarization_result['num_speakers']}")

    # Calculate metrics
    total_time = trans_time + diar_time
    processing_ratio = total_time / audio_duration

    results = {
        "phase": "Phase 0: Baseline",
        "timestamp": TIMESTAMP,
        "configuration": {
            "beam_size": 5,
            "best_of": 5,
            "optimization_thresholds": False,
            "condition_on_previous_text": False
        },
        "metrics": {
            "audio_duration": audio_duration,
            "initialization_time": init_time,
            "transcription_time": trans_time,
            "diarization_time": diar_time,
            "total_processing_time": total_time,
            "processing_ratio": processing_ratio
        },
        "accuracy": {
            "speakers_detected": diarization_result["num_speakers"],
            "segments_count": len(diarization_result["segments"]),
            "transcription_confidence": transcription_result.confidence,
            "text_preview": transcription_result.text[:500]
        },
        "log_file": str(log_file)
    }

    results_file = save_results(phase_name, results)

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 0 RESULTS:")
    logger.info(f"  Processing Ratio: {processing_ratio:.2f}x")
    logger.info(f"  Transcription: {trans_time:.2f}s")
    logger.info(f"  Diarization: {diar_time:.2f}s")
    logger.info(f"  Total: {total_time:.2f}s")
    logger.info(f"  Speakers: {diarization_result['num_speakers']}")
    logger.info(f"  Results saved to: {results_file}")
    logger.info("=" * 80)

    return results

async def test_phase1_whisper_params():
    """Phase 1: Full Whisper parameters (beam=10, thresholds, etc)"""
    phase_name = "phase1_whisper_full"
    logger, log_file = setup_logging(phase_name)

    logger.info("PHASE 1: FULL WHISPER PARAMETERS TEST")
    logger.info("Configuration:")
    logger.info("  - beam_size: 10 (vs 5)")
    logger.info("  - best_of: 10 (vs 5)")
    logger.info("  - condition_on_previous_text: True")
    logger.info("  - compression_ratio_threshold: 1.6")
    logger.info("  - no_speech_threshold: 0.85")
    logger.info("  - hallucination_silence_threshold: 0.8")
    logger.info("  - temperature: 0.0")
    logger.info("  - vad_parameters: optimized")

    audio_path = "data/recordings/d.speakers.wav"
    audio_duration = librosa.get_duration(path=audio_path)

    # Initialize services
    init_start = time.time()
    transcription_service = TranscriptionService(model_name="medium", device="cpu")
    diarization_service = PyannoteDiarizer(device="cpu")
    init_time = time.time() - init_start

    # Transcription WITH FULL PARAMETERS
    logger.info("Starting transcription with FULL optimized parameters...")
    trans_start = time.time()

    # IMPORTANT: We'll pass parameters directly
    # This test assumes we'll modify transcription.py to accept these
    transcription_result = await transcription_service.transcribe_with_enhancements(
        audio_path,
        word_timestamps=True,
        beam_size=10,
        best_of=10,
        vad_parameters={
            "threshold": 0.4,
            "min_speech_duration_ms": 150,
            "min_silence_duration_ms": 1000,
        }
    )
    trans_time = time.time() - trans_start
    logger.info(f"Transcription completed in {trans_time:.2f}s")
    logger.info(f"Transcription text preview: {transcription_result.text[:200]}...")

    # Diarization
    logger.info("Starting diarization...")
    diar_start = time.time()
    diarization_result = await diarization_service.diarize(
        audio_path,
        transcription_result.segments
    )
    diar_time = time.time() - diar_start
    logger.info(f"Diarization completed in {diar_time:.2f}s")

    total_time = trans_time + diar_time
    processing_ratio = total_time / audio_duration

    results = {
        "phase": "Phase 1: Full Whisper Parameters",
        "timestamp": TIMESTAMP,
        "configuration": {
            "beam_size": 10,
            "best_of": 10,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 1.6,
            "no_speech_threshold": 0.85,
            "hallucination_silence_threshold": 0.8,
            "temperature": 0.0,
            "vad_optimized": True
        },
        "metrics": {
            "audio_duration": audio_duration,
            "initialization_time": init_time,
            "transcription_time": trans_time,
            "diarization_time": diar_time,
            "total_processing_time": total_time,
            "processing_ratio": processing_ratio
        },
        "accuracy": {
            "speakers_detected": diarization_result["num_speakers"],
            "segments_count": len(diarization_result["segments"]),
            "transcription_confidence": transcription_result.confidence,
            "text_preview": transcription_result.text[:500]
        },
        "log_file": str(log_file)
    }

    results_file = save_results(phase_name, results)

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1 RESULTS:")
    logger.info(f"  Processing Ratio: {processing_ratio:.2f}x")
    logger.info(f"  Transcription: {trans_time:.2f}s")
    logger.info(f"  Diarization: {diar_time:.2f}s")
    logger.info(f"  Total: {total_time:.2f}s")
    logger.info(f"  Speakers: {diarization_result['num_speakers']}")
    logger.info(f"  Results saved to: {results_file}")
    logger.info("=" * 80)

    return results

async def run_all_tests():
    """Run all test phases systematically"""
    print("\n" + "=" * 80)
    print("SYSTEMATIC PERFORMANCE TESTING")
    print("=" * 80)
    print(f"Test Run ID: {TIMESTAMP}")
    print(f"Logs Directory: {LOGS_DIR}")
    print("=" * 80)
    print("")

    all_results = {}

    # Phase 0: Baseline
    print("\n[1/2] Running Phase 0: Baseline...")
    baseline_results = await test_phase_baseline()
    if baseline_results:
        all_results["phase0_baseline"] = baseline_results
        print(f"✅ Phase 0 Complete: {baseline_results['metrics']['processing_ratio']:.2f}x")

    # Phase 1: Full Whisper Parameters
    print("\n[2/2] Running Phase 1: Full Whisper Parameters...")
    phase1_results = await test_phase1_whisper_params()
    if phase1_results:
        all_results["phase1_whisper_full"] = phase1_results
        print(f"✅ Phase 1 Complete: {phase1_results['metrics']['processing_ratio']:.2f}x")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if "phase0_baseline" in all_results and "phase1_whisper_full" in all_results:
        baseline_ratio = all_results["phase0_baseline"]["metrics"]["processing_ratio"]
        phase1_ratio = all_results["phase1_whisper_full"]["metrics"]["processing_ratio"]
        improvement = ((baseline_ratio - phase1_ratio) / baseline_ratio) * 100

        print(f"\nPhase 0 (Baseline):  {baseline_ratio:.2f}x")
        print(f"Phase 1 (Optimized): {phase1_ratio:.2f}x")
        print(f"Improvement: {improvement:+.1f}%")

        if phase1_ratio <= 1.64:
            print(f"\n🎯 TARGET ACHIEVED! (≤1.64x)")
        elif phase1_ratio <= 1.8:
            print(f"\n✅ GOOD PERFORMANCE (≤1.8x)")
        else:
            print(f"\n⚠️ Need more optimization")

    # Save comprehensive summary
    summary_file = LOGS_DIR / f"{TIMESTAMP}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Complete results saved to: {summary_file}")
    print(f"📁 All logs in: {LOGS_DIR}")
    print("=" * 80)

    return all_results

if __name__ == "__main__":
    print("\n🔬 Starting Systematic Performance Testing...")
    print("This will test each optimization phase and log everything separately.\n")

    results = asyncio.run(run_all_tests())

    print("\n✅ Testing complete!")
