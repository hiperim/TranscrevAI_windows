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

# Import services
from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
import librosa
from jiwer import wer as calculate_wer

# --- Configuration ---
LOGS_DIR = Path("tests/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
AUDIO_FILE = "data/recordings/d.speakers.wav"
GROUND_TRUTH_FILE = "tests/ground_truth/d_speakers.txt"

# --- Helper Functions ---
def setup_logging(phase_name: str):
    """Sets up logging for a specific test phase."""
    log_file = LOGS_DIR / f"{TIMESTAMP}_{phase_name}.log"
    # Remove all handlers from the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Add new handlers
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

def get_ground_truth(path: str) -> str:
    """Reads the ground truth text from a file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

async def run_test(phase_name: str, test_config: dict):
    """
    Generic test runner for a given configuration.
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
    transcription_service = TranscriptionService(model_name="medium", device="cpu")
    await transcription_service.initialize(compute_type=test_config.get("compute_type", "int8"))
    diarization_service = PyannoteDiarizer(device="cpu")
    init_time = time.perf_counter() - init_start
    logger.info(f"Services initialized in {init_time:.2f}s")

    # --- Transcription ---
    trans_start = time.perf_counter()
    transcription_result = await transcription_service.transcribe_with_enhancements(
        audio_path,
        word_timestamps=True,
        **test_config.get("transcribe_params", {})
    )
    trans_time = time.perf_counter() - trans_start
    logger.info(f"Transcription completed in {trans_time:.2f}s")

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
    
    ground_truth_text = get_ground_truth(GROUND_TRUTH_FILE)
    word_error_rate = calculate_wer(ground_truth_text, transcription_result.text)
    accuracy = (1 - word_error_rate) * 100
    
    diarization_correct = diarization_result['num_speakers'] == 2

    # --- Results ---
    results = {
        "phase": phase_name,
        "timestamp": TIMESTAMP,
        "configuration": test_config,
        "performance": {
            "audio_duration_sec": audio_duration,
            "init_time_sec": init_time,
            "transcription_time_sec": trans_time,
            "diarization_time_sec": diar_time,
            "total_processing_time_sec": total_time,
            "processing_ratio": f"{processing_ratio:.2f}x"
        },
        "quality": {
            "wer": f"{word_error_rate:.2%}",
            "accuracy_percent": f"{accuracy:.2f}%",
            "speakers_detected": diarization_result["num_speakers"],
            "ground_truth_speakers": 2,
            "diarization_correct": diarization_correct
        }
    }
    
    save_results(phase_name, results)
    logger.info(f"--- FINISHED TEST PHASE: {phase_name} ---")
    print(f"✅ Phase '{phase_name}' Complete. Ratio: {results['performance']['processing_ratio']} | Accuracy: {results['quality']['accuracy_percent']} | Speakers: {results['quality']['speakers_detected']}")
    
    return results

async def main():
    """Defines and runs all test configurations."""
    
    test_configs = {
        "baseline_current": {
            "description": "Current 'slow' baseline configuration.",
            "transcribe_params": {
                "beam_size": 5,
                "best_of": 5,
                "vad_parameters": {"threshold": 0.5, "min_silence_duration_ms": 2000}
            }
        },
        "full_parameters": {
            "description": "The 'Full Parameters' configuration from previous tests.",
            "transcribe_params": {
                "beam_size": 10,
                "best_of": 10,
                "temperature": 0.0,
                "condition_on_previous_text": True,
                "compression_ratio_threshold": 1.6,
                "no_speech_threshold": 0.85,
                "hallucination_silence_threshold": 0.8,
                "vad_parameters": {"threshold": 0.4, "min_speech_duration_ms": 150, "min_silence_duration_ms": 1000}
            }
        },
        "medium_parameters": {
            "description": "The 'Medium Parameters' configuration from previous tests.",
            "transcribe_params": {
                "beam_size": 8,
                "best_of": 8,
                "vad_parameters": {"threshold": 0.4, "min_speech_duration_ms": 150, "min_silence_duration_ms": 1000}
            }
        },
        "restore_1.6x_config": {
            "description": "The 'Golden' configuration from Oct 15th, aiming for ~1.6x performance.",
            "transcribe_params": {
                "beam_size": 10,
                "best_of": 10,
                "vad_parameters": {"threshold": 0.4, "min_speech_duration_ms": 150, "min_silence_duration_ms": 1000}
            }
        }
    }

    all_results = {}
    
    print("\n" + "=" * 80)
    print("SYSTEMATIC PERFORMANCE & ACCURACY TEST")
    print(f"Test Run ID: {TIMESTAMP}")
    print("=" * 80 + "\n")
    
    i = 1
    for name, config in test_configs.items():
        print(f"\n[{i}/{len(test_configs)}] Running Phase: {name}")
        print(f"Description: {config['description']}")
        results = await run_test(name, config)
        if results:
            all_results[name] = results
        i += 1

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for name, res in all_results.items():
        print(f"-> {name}:")
        print(f"  - Ratio: {res['performance']['processing_ratio']}")
        print(f"  - Transcription Accuracy: {res['quality']['accuracy_percent']}")
        print(f"  - Speakers Detected: {res['quality']['speakers_detected']}")
    
    summary_file = LOGS_DIR / f"{TIMESTAMP}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n📊 Complete results saved to: {summary_file}")
    print(f"📁 All logs in: {LOGS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())