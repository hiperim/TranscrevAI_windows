"""
Diarization Accuracy Test - Training Set Only
Tests speaker detection accuracy on d.speakers.wav and q.speakers.wav

Ground truth:
- d.speakers.wav: 2 speakers
- q.speakers.wav: 4 speakers

This test is separate from transcription tests to:
1. Isolate diarization issues
2. Run faster (no transcription overhead)
3. Focus on training set only (no validation set)
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diarization import PyannoteDiarizer
from src.transcription import TranscriptionService

# Ground truth for training set
TRAIN_GROUND_TRUTH = {
    "d.speakers.wav": {"expected_speakers": 2, "description": "2 speakers, formal conversation"},
    "q.speakers.wav": {"expected_speakers": 4, "description": "4 speakers, group discussion"}
}

# Audio files location
DATA_DIR = Path(__file__).parent.parent / "data" / "recordings"

async def test_diarization_accuracy():
    """Test diarization accuracy on training set"""

    print("=" * 60)
    print("DIARIZATION ACCURACY TEST - TRAINING SET")
    print("=" * 60)
    print()

    # Initialize services
    print("Initializing services...")
    diarizer = PyannoteDiarizer(device="cpu")
    transcription_service = TranscriptionService(model_name="medium", device="cpu", compute_type="int8")

    if not diarizer.pipeline:
        print("❌ CRITICAL: Diarization pipeline failed to load!")
        print("Check HUGGING_FACE_HUB_TOKEN in .env file")
        return

    print(f"✅ Diarization pipeline loaded (threshold=0.35)")
    print(f"✅ Transcription service ready")
    print()

    results = []
    total_files = len(TRAIN_GROUND_TRUTH)
    correct_count = 0

    for idx, (filename, ground_truth) in enumerate(TRAIN_GROUND_TRUTH.items(), 1):
        print(f"[{idx}/{total_files}] Testing: {filename}")
        print(f"  Description: {ground_truth['description']}")
        print(f"  Expected speakers: {ground_truth['expected_speakers']}")

        audio_path = DATA_DIR / filename

        if not audio_path.exists():
            print(f"  ❌ ERROR: File not found at {audio_path}")
            results.append({
                "file": filename,
                "status": "error",
                "error": "File not found"
            })
            continue

        try:
            start_time = time.time()

            # Step 1: Transcribe (needed for diarization alignment)
            print(f"  → Transcribing audio...")
            transcription_result = await transcription_service.transcribe_with_enhancements(
                str(audio_path),
                beam_size=10,
                best_of=10,
                word_timestamps=True
            )

            # Step 2: Diarize
            print(f"  → Diarizing audio...")
            diarization_result = await diarizer.diarize(
                str(audio_path),
                transcription_result.segments  # Access attribute, not dict key
            )

            elapsed = time.time() - start_time

            detected_speakers = diarization_result["num_speakers"]
            expected_speakers = ground_truth["expected_speakers"]

            is_correct = detected_speakers == expected_speakers
            if is_correct:
                correct_count += 1

            # Status symbol
            status_symbol = "✅" if is_correct else "❌"

            print(f"  {status_symbol} Detected: {detected_speakers} speakers")
            print(f"  Time: {elapsed:.1f}s")
            print()

            results.append({
                "file": filename,
                "expected_speakers": expected_speakers,
                "detected_speakers": detected_speakers,
                "correct": is_correct,
                "processing_time": round(elapsed, 2),
                "status": "pass" if is_correct else "fail"
            })

        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            print()
            results.append({
                "file": filename,
                "status": "error",
                "error": str(e)
            })

    # Summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()

    accuracy = (correct_count / total_files) * 100 if total_files > 0 else 0

    for result in results:
        if result.get("status") == "error":
            print(f"❌ {result['file']}: ERROR - {result.get('error')}")
        else:
            symbol = "✅" if result["correct"] else "❌"
            print(f"{symbol} {result['file']}: {result['detected_speakers']}/{result['expected_speakers']} speakers ({result['processing_time']:.1f}s)")

    print()
    print(f"Accuracy: {correct_count}/{total_files} = {accuracy:.1f}%")
    print()

    # Interpretation
    if accuracy == 100:
        print("🎉 PERFECT! Diarization is working correctly on training set.")
    elif accuracy >= 50:
        print("⚠️  PARTIAL: Some files correct, needs investigation.")
    else:
        print("❌ CRITICAL: Diarization not working properly.")

    # Save report
    report_dir = Path(__file__).parent.parent / ".claude" / "test_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"diarization_train_set_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "dataset": "training_set",
        "files_tested": list(TRAIN_GROUND_TRUTH.keys()),
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_files": total_files,
        "config": {
            "model": "pyannote/speaker-diarization-3.1",
            "device": "cpu",
            "threshold": 0.35
        },
        "results": results
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report saved: {report_path}")
    print("=" * 60)

    return accuracy == 100

if __name__ == "__main__":
    success = asyncio.run(test_diarization_accuracy())
    sys.exit(0 if success else 1)
