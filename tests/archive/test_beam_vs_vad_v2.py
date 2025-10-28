import asyncio
import time
import logging
import os
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path to allow importing from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
import librosa

# --- Configuration ---
LOGS_DIR = Path("tests/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
AUDIO_FILE = "data/recordings/d.speakers.wav"

# --- Helper Functions ---
def setup_logging(phase_name: str):
    """Sets up logging for a specific test phase."""
    log_file = LOGS_DIR / f"{TIMESTAMP}_{phase_name}.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"--- STARTING TEST PHASE: {phase_name} ---")
    return logger, log_file

def save_results(phase_name: str, results: dict):
    """Saves results to a JSON file."""
    results_file = LOGS_DIR / f"{TIMESTAMP}_{phase_name}_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to: {results_file}")
    return results_file

def analyze_and_get_accuracy(transcribed_text: str, ground_truth_path: str):
    """Analyzes transcription accuracy against a ground truth file."""
    from jiwer import wer

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        reference = f.read().strip()

    error = wer(reference, transcribed_text)
    accuracy = (1 - error) * 100

    logging.info(f"Reference:    {reference}")
    logging.info(f"Hypothesis:   {transcribed_text}")
    logging.info(f"WER: {error:.2%}, Accuracy: {accuracy:.2f}%")
    return accuracy, error

async def run_test(phase_name: str, test_config: dict):
    """
    Generic test runner for a given configuration.
    V2: ADAPTED for old TranscriptionService API (no compute_type in __init__)
    """
    logger, log_file = setup_logging(phase_name)

    audio_path = test_config.get("audio_file", AUDIO_FILE)
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        return None

    audio_duration = librosa.get_duration(path=audio_path)
    logger.info(f"Test File: {audio_path}, Duration: {audio_duration:.2f}s")

    # --- Services Initialization ---
    init_start = time.perf_counter()
    # V2: OLD API - no compute_type parameter in __init__
    transcription_service = TranscriptionService(
        model_name="medium",
        device="cpu"
    )
    diarization_service = PyannoteDiarizer(device="cpu")
    init_time = time.perf_counter() - init_start
    logger.info(f"Services initialized in {init_time:.2f}s")

    # --- Transcription ---
    trans_start = time.perf_counter()
    # V2: OLD API - transcribe_with_enhancements doesn't accept beam_size/vad_parameters
    # We need to call transcribe directly on the model
    # For testing purposes, we'll use the simplified API
    transcription_result = await transcription_service.transcribe_with_enhancements(
        audio_path,
        word_timestamps=True
    )
    trans_time = time.perf_counter() - trans_start
    logger.info(f"Transcription completed in {trans_time:.2f}s")
    logger.info(f"NOTE: V2 - using old API, custom beam_size/VAD params NOT applied in this test")

    # --- Diarization ---
    diar_start = time.perf_counter()
    diarization_result = await diarization_service.diarize(
        audio_path,
        transcription_result.segments
    )
    diar_time = time.perf_counter() - diar_start
    logger.info(f"Diarization completed in {diar_time:.2f}s")

    # --- Metrics Calculation ---
    total_time = trans_time + diar_time
    processing_ratio = total_time / audio_duration

    # Accuracy
    ground_truth_file = "tests/ground_truth/d_speakers.txt"
    accuracy, wer_value = analyze_and_get_accuracy(transcription_result.text, ground_truth_file)

    # --- Results ---
    results = {
        "phase": phase_name,
        "timestamp": TIMESTAMP,
        "configuration": test_config,
        "note": "V2: Old API - beam_size/VAD params NOT applied (hardcoded in transcription.py)",
        "performance": {
            "audio_duration_sec": audio_duration,
            "init_time_sec": init_time,
            "transcription_time_sec": trans_time,
            "diarization_time_sec": diar_time,
            "total_processing_time_sec": total_time,
            "processing_ratio": f"{processing_ratio:.2f}x"
        },
        "quality": {
            "wer": wer_value,
            "accuracy_percent": accuracy,
            "speakers_detected": diarization_result["num_speakers"],
            "ground_truth_speakers": 2
        },
        "transcription": {
            "text": transcription_result.text,
            "word_count": transcription_result.word_count,
            "confidence": transcription_result.confidence
        }
    }

    results_file = save_results(phase_name, results)
    logger.info(f"--- FINISHED TEST PHASE: {phase_name} ---")
    print(f"✅ Phase '{phase_name}' Complete. Ratio: {processing_ratio:.2f}x | Accuracy: {accuracy:.2f}% | Speakers: {diarization_result['num_speakers']} | PT-BR Corrections: {len(transcription_service.ptbr_corrections)}")

    return results

async def main():
    """
    V2: Tests OLD COMMITTED VERSION (27e2c59) with 291 PT-BR corrections
    Note: This version uses hardcoded beam_size=5 and VAD params in transcription.py
    """

    # Configuration for each test phase
    test_configs = {
        "committed_version_baseline": {
            "description": "OLD COMMITTED VERSION (27e2c59): 291 PT-BR corrections, lowercase/capitalize method, hardcoded beam=5",
            "transcribe_params": {}  # NOTE: These are ignored by old API
        }
    }

    all_results = {}

    print("\n" + "=" * 80)
    print("TESTING COMMITTED VERSION (27e2c59) - 291 PT-BR Corrections")
    print(f"Test Run ID: {TIMESTAMP}")
    print("=" * 80 + "\n")

    for i, (name, config) in enumerate(test_configs.items()):
        print(f"\n[{i+1}/{len(test_configs)}] Running Phase: {name}")
        print(f"Description: {config['description']}")
        results = await run_test(name, config)
        if results:
            all_results[name] = results

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for name, res in all_results.items():
        print(f"-> {name}:")
        print(f"  - PT-BR Corrections: 291 (lowercase→replace→capitalize)")
        print(f"  - Ratio: {res['performance']['processing_ratio']}")
        print(f"  - Transcription Accuracy: {res['quality']['accuracy_percent']:.2f}%")
        print(f"  - WER: {res['quality']['wer']:.2%}")
        print(f"  - Speakers Detected: {res['quality']['speakers_detected']}/2")
        print(f"  - Transcription: \"{res['transcription']['text']}\"")

    summary_file = LOGS_DIR / f"{TIMESTAMP}_committed_version_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n📊 Full summary saved to: {summary_file}")
    print(f"📁 All logs in: {LOGS_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
