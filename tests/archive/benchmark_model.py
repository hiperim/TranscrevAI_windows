# tests/benchmark_model.py
"""
Final, reliable benchmark script for A/B testing compute types.

Uses the new, user-verified ground truth data to measure:
1. Transcription Accuracy (Word Error Rate)
2. Diarization Accuracy (Speaker Count)
3. Processing Speed (Ratio)
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Any
import librosa

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from tests.metrics import calculate_wer

# ===========================================================
# CONFIGURATION
# ===========================================================

# Change this value to test different compute types
COMPUTE_TYPE = "float32"  # Options: "int8", "float32", "int16"
MODEL_NAME = "C://TranscrevAI_windows//models//pierreguillou-medium-ptbr-ct2-float32"

# Ground truth data (verified by user)
GROUND_TRUTH = {
    "d.speakers.wav": {
        "text_file": "d_speakers.txt",
        "speakers": 2
    },
    "q.speakers.wav": {
        "text_file": "q_speakers.txt",
        "speakers": 4
    },
    "t.speakers.wav": {
        "text_file": "t_speakers.txt",
        "speakers": 3
    },
    "t2.speakers.wav": {
        "text_file": "t2_speakers.txt",
        "speakers": 3
    }
}

# Paths
AUDIO_DIR = Path(__file__).parent.parent / "data" / "recordings"
TRUTH_DIR = Path(__file__).parent / "ground_truth"

# ===========================================================

async def run_single_test(service: TranscriptionService, diarizer: PyannoteDiarizer, audio_path: Path, ground_truth: Dict) -> Dict:
    """Runs a full test on a single file and returns a results dictionary."""
    print(f"---\nProcessing: {audio_path.name}")
    
    # 1. Get audio duration
    audio_duration = librosa.get_duration(path=str(audio_path))
    
    # 2. Read ground truth text
    truth_text_path = TRUTH_DIR / ground_truth["text_file"]
    expected_text = truth_text_path.read_text(encoding="utf-8").strip()
    expected_speakers = ground_truth["speakers"]

    # 3. Run full pipeline
    start_time = time.time()
    
    transcription_result = await service.transcribe_with_enhancements(
        str(audio_path),
        beam_size=5, # Using baseline beam_size
        best_of=5
    )
    diarization_result = await diarizer.diarize(str(audio_path), transcription_result.segments)
    
    end_time = time.time()
    
    # 4. Collect results
    processing_time = end_time - start_time
    processing_ratio = processing_time / audio_duration
    actual_text = transcription_result.text
    detected_speakers = diarization_result["num_speakers"]
    
    # 5. Calculate metrics
    wer = calculate_wer(expected_text, actual_text)
    transcription_accuracy = max(0, 1 - wer) * 100
    diarization_accuracy = 100.0 if detected_speakers == expected_speakers else 0.0

    print(f"  Speed: {processing_ratio:.2f}x ({processing_time:.2f}s)")
    print(f"  Transcription Accuracy (1-WER): {transcription_accuracy:.2f}%")
    print(f"  Diarization Accuracy: {detected_speakers}/{expected_speakers} speakers detected ({diarization_accuracy:.0f}%)")

    return {
        "file": audio_path.name,
        "speed_ratio": processing_ratio,
        "transcription_accuracy": transcription_accuracy,
        "diarization_accuracy": diarization_accuracy,
        "detected_speakers": detected_speakers,
        "expected_speakers": expected_speakers
    }

async def main():
    """Main function to run the benchmark."""
    print("="*50)
    print(f"FINAL BENCHMARK: {COMPUTE_TYPE.upper()}")
    print("="*50)

    # Initialize services
    try:
        transcription_service = TranscriptionService(model_name=MODEL_NAME, compute_type=COMPUTE_TYPE)
        await transcription_service.initialize()
        diarizer = PyannoteDiarizer()
    except Exception as e:
        print(f"Error during service initialization: {e}")
        return

    all_results = []
    for audio_filename, truth_data in GROUND_TRUTH.items():
        audio_path = AUDIO_DIR / audio_filename
        if not audio_path.exists():
            print(f"[WARNING] Audio file not found: {audio_path}")
            continue
        
        result = await run_single_test(transcription_service, diarizer, audio_path, truth_data)
        all_results.append(result)

    # --- Final Summary ---
    if not all_results:
        print("No files were tested.")
        return

    avg_speed = sum(r["speed_ratio"] for r in all_results) / len(all_results)
    avg_trans_acc = sum(r["transcription_accuracy"] for r in all_results) / len(all_results)
    avg_diar_acc = sum(r["diarization_accuracy"] for r in all_results) / len(all_results)

    print("\n" + "="*50)
    print(f"AVERAGE RESULTS ({COMPUTE_TYPE.upper()})")
    print("="*50)
    print(f"  Processing Speed Ratio: {avg_speed:.2f}x")
    print(f"  Transcription Accuracy: {avg_trans_acc:.2f}%")
    print(f"  Diarization Accuracy:   {avg_diar_acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
