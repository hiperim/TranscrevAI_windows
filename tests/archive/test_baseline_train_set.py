# tests/test_baseline_train_set.py
"""
Baseline Test - Training Set Only (d.speakers.wav, q.speakers.wav)

IMPORTANT: This test uses ONLY the training set files.
Files t.speakers.wav and t2.speakers.wav are RESERVED for final validation.

Purpose:
- Establish baseline accuracy on training set
- Analyze errors for rule creation
- Measure improvements without overfitting

Training Set:
- d.speakers.wav: 2 speakers, clear audio
- q.speakers.wav: 4 speakers, complex conversation

Validation Set (DO NOT USE for development):
- t.speakers.wav: Reserved for final validation
- t2.speakers.wav: Reserved for final validation
"""

import asyncio
import time
import sys
import re
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from tests.metrics import calculate_wer, calculate_cer

# TRAINING SET ONLY - d and q files
TRAIN_GROUND_TRUTH = {
    "d.speakers.wav": {"text_file": "d_speakers.txt", "speakers": 2},
    "q.speakers.wav": {"text_file": "q_speakers.txt", "speakers": 4}
}

AUDIO_DIR = Path(__file__).parent.parent / "data" / "recordings"
TRUTH_DIR = Path(__file__).parent / "ground_truth"
REPORT_DIR = Path(__file__).parent.parent / ".claude" / "test_reports"

def normalize_text_for_wer(text: str) -> str:
    """Normaliza texto para comparação justa."""
    text = text.lower()
    text = re.sub(r'[^\w\sàáâãèéêìíîòóôõùúûç]', '', text)
    text = ' '.join(text.split())
    return text.strip()

async def test_baseline_train_set():
    """Testa baseline ULTRA ACCURACY em TRAINING SET (d, q apenas)."""

    print("\n" + "="*70)
    print("BASELINE TEST - TRAINING SET ONLY")
    print("Model: medium int8 ULTRA ACCURACY")
    print("="*70)
    print("\nTRAINING SET (development):")
    print("  ✅ d.speakers.wav (2 speakers, clear)")
    print("  ✅ q.speakers.wav (4 speakers, complex)")
    print("\nVALIDATION SET (reserved for final test):")
    print("  🔒 t.speakers.wav - NOT USED")
    print("  🔒 t2.speakers.wav - NOT USED")
    print("")
    print("Configuration: ULTRA ACCURACY")
    print("  - VAD: threshold 0.4, silence 1000ms")
    print("  - Beam: 10/10")
    print("  - PT-BR: 12 safe rules")
    print("="*70)

    # Initialize service (ULTRA ACCURACY defaults)
    service = TranscriptionService(
        model_name="medium",
        compute_type="int8"
    )
    await service.initialize()

    results = []

    for audio_file, truth_data in TRAIN_GROUND_TRUTH.items():
        audio_path = AUDIO_DIR / audio_file
        if not audio_path.exists():
            print(f"\n⚠️  WARNING: {audio_file} not found, skipping...")
            continue

        print(f"\n{audio_file}...", end=" ", flush=True)

        # Ground truth
        truth_path = TRUTH_DIR / truth_data["text_file"]
        if not truth_path.exists():
            print(f"⚠️  Ground truth not found, skipping...")
            continue

        expected_raw = truth_path.read_text(encoding="utf-8").strip()

        # Transcribe
        start = time.time()
        result = await service.transcribe_with_enhancements(str(audio_path))
        processing_time = time.time() - start

        actual_raw = result.text

        # Metrics
        expected_norm = normalize_text_for_wer(expected_raw)
        actual_norm = normalize_text_for_wer(actual_raw)

        wer = calculate_wer(expected_norm, actual_norm)
        cer = calculate_cer(expected_norm, actual_norm)
        accuracy = max(0, 1 - wer) * 100

        print(f"WER: {wer:.2%} | Acc: {accuracy:.1f}% | Time: {processing_time:.1f}s")

        results.append({
            "file": audio_file,
            "wer": wer,
            "cer": cer,
            "accuracy": accuracy,
            "processing_time": processing_time,
            "expected_raw": expected_raw,
            "actual_raw": actual_raw,
            "expected_normalized": expected_norm,
            "actual_normalized": actual_norm
        })

    await service.unload_model()

    if len(results) == 0:
        print("\n❌ No results - check if audio/ground truth files exist")
        return None

    # Summary
    avg_wer = sum(r['wer'] for r in results) / len(results)
    avg_cer = sum(r['cer'] for r in results) / len(results)
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
    avg_time = sum(r['processing_time'] for r in results) / len(results)

    print("\n" + "="*70)
    print("TRAINING SET BASELINE RESULTS")
    print("="*70)
    print(f"Files tested:   {len(results)}/2")
    print(f"Avg. Accuracy:  {avg_accuracy:.2f}%")
    print(f"Avg. WER:       {avg_wer:.2%}")
    print(f"Avg. CER:       {avg_cer:.2%}")
    print(f"Avg. Time:      {avg_time:.1f}s")
    print("="*70)

    print("\nPer-file breakdown:")
    for r in results:
        print(f"  {r['file']:20s} - Acc: {r['accuracy']:5.1f}% | WER: {r['wer']:5.2%}")

    print("\n" + "="*70)
    print("NOTE: This is TRAINING SET performance only.")
    print("Final validation will use t.speakers.wav and t2.speakers.wav")
    print("="*70)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORT_DIR / f"baseline_train_set_{timestamp}.json"
    json_path.write_text(json.dumps({
        "timestamp": timestamp,
        "dataset": "training_set",
        "files_included": list(TRAIN_GROUND_TRUTH.keys()),
        "files_excluded": ["t.speakers.wav", "t2.speakers.wav"],
        "config": {
            "model": "medium",
            "compute_type": "int8",
            "vad": {
                "threshold": 0.4,
                "min_speech_duration_ms": 150,
                "min_silence_duration_ms": 1000
            },
            "beam_search": {
                "beam_size": 10,
                "best_of": 10
            },
            "ptbr_corrections": "12 safe rules"
        },
        "avg_accuracy": avg_accuracy,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_time": avg_time,
        "results": results
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"\n✅ Report saved: {json_path}")

    return {
        "avg_accuracy": avg_accuracy,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_time": avg_time,
        "results": results
    }

if __name__ == "__main__":
    asyncio.run(test_baseline_train_set())
