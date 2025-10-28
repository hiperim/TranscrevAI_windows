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
    transcription_service = TranscriptionService(
        model_name="medium", 
        device="cpu", 
        compute_type=test_config.get("compute_type", "int8")
    )
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
    
    # Accuracy
    ground_truth_file = "tests/ground_truth/d_speakers.txt"
    accuracy, wer = analyze_and_get_accuracy(transcription_result.text, ground_truth_file)

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
            "wer": wer,
            "accuracy_percent": accuracy,
            "speakers_detected": diarization_result["num_speakers"],
            "ground_truth_speakers": 2
        }
    }
    
    results_file = save_results(phase_name, results)
    logger.info(f"--- FINISHED TEST PHASE: {phase_name} ---")
    print(f"✅ Phase '{phase_name}' Complete. Ratio: {processing_ratio:.2f}x | Accuracy: {accuracy:.2f}% | Speakers: {diarization_result['num_speakers']}")
    
    return results

async def main():
    """Defines and runs all test configurations."""
    
    # Configuration for each test phase
    test_configs = {
        "baseline_current": {
            "description": "Current 'slow' baseline configuration.",
            "transcribe_params": {
                "beam_size": 5,
                "vad_parameters": {"threshold": 0.5, "min_silence_duration_ms": 2000}
            }
        },
        "beam10_default_vad": {
            "description": "Your suggestion: Increase beam size, but use default VAD.",
            "transcribe_params": {
                "beam_size": 10,
                "best_of": 10
            }
        },
        "beam10_aggressive_vad": {
            "description": "The configuration from Oct 15th that was faster.",
            "transcribe_params": {
                "beam_size": 10,
                "best_of": 10,
                "vad_parameters": {"threshold": 0.4, "min_speech_duration_ms": 150, "min_silence_duration_ms": 1000}
            }
        }
    }

    all_results = {}
    
    print("\n" + "=" * 80)
    print("PERFORMANCE & ACCURACY COMPARISON: BEAM SIZE, VAD, and AGGRESSIVENESS")
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
        print(f"  - Ratio: {res['performance']['processing_ratio']}")
        print(f"  - Transcription Accuracy: {res['quality']['accuracy_percent']:.2f}%")
        print(f"  - Speakers Detected: {res['quality']['speakers_detected']}")
    
    summary_file = LOGS_DIR / f"{TIMESTAMP}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nFull summary saved to: {summary_file}")

if __name__ == "__main__":
    asyncio.run(main())
