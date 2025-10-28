"""
Debug script to isolate and diagnose diarization speaker detection issues.

This script tests pyannote diarization in isolation to understand:
1. Why it detects 3 speakers instead of 4
2. Why .crop() returns empty annotations
3. Whether the issue is deterministic or random

Expected: 4 speakers for q.speakers.wav (based on baseline results)
Actual: 3 speakers currently detected
"""

import os
import torch
from pathlib import Path
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Configuration
AUDIO_FILE = Path("C:/transcrevai_windows/data/recordings/q.speakers.wav")
EXPECTED_SPEAKERS = 4

def test_diarization_with_threshold(threshold=0.35):
    """Test pyannote diarization with specified clustering threshold."""

    print(f"\n{'='*80}")
    print(f"Testing Diarization with threshold={threshold}")
    print(f"{'='*80}\n")

    # Load environment and token
    load_dotenv()
    auth_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not auth_token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not set")

    # Load pipeline
    print("Loading pyannote pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token
    ).to(torch.device("cpu"))

    # Configure clustering
    pipeline.instantiate({
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 15,
            "threshold": threshold
        }
    })

    print(f"Pipeline loaded with threshold={threshold}\n")

    # Run diarization
    print(f"Processing {AUDIO_FILE.name}...")
    diarization_result = pipeline(str(AUDIO_FILE))

    # Extract results
    speakers = diarization_result.labels()
    num_speakers = len(speakers)

    print(f"\n{'='*80}")
    print(f"RESULTS: {num_speakers} speakers detected (expected: {EXPECTED_SPEAKERS})")
    print(f"{'='*80}\n")

    print(f"Speakers found: {sorted(speakers)}\n")

    # Print all segments
    print("Diarization Segments:")
    print(f"{'Start':<10} {'End':<10} {'Duration':<10} {'Speaker':<15}")
    print("-" * 50)

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        duration = turn.end - turn.start
        print(f"{turn.start:<10.3f} {turn.end:<10.3f} {duration:<10.3f} {speaker:<15}")

    print("\n" + "="*80)

    return diarization_result, num_speakers


def test_crop_functionality(diarization_result):
    """Test .crop() method to understand why it returns empty annotations."""

    print(f"\n{'='*80}")
    print("Testing .crop() Functionality")
    print(f"{'='*80}\n")

    # Test some specific timestamps
    test_cases = [
        (1.0, 3.0, "Early segment"),
        (5.0, 7.0, "Mid segment"),
        (10.0, 12.0, "Late segment"),
        (1.07, 1.15, "Very short word-like segment"),
        (3.43, 3.52, "Another short segment"),
    ]

    print(f"{'Start':<10} {'End':<10} {'Description':<30} {'Empty?':<10} {'Speaker':<15}")
    print("-" * 80)

    for start, end, description in test_cases:
        cropped = diarization_result.crop(start, end)
        is_empty = cropped.is_empty() if cropped else True

        speaker = "N/A"
        if cropped and not is_empty:
            try:
                speaker = cropped.argmax()
            except:
                speaker = "ERROR"

        print(f"{start:<10.2f} {end:<10.2f} {description:<30} {str(is_empty):<10} {speaker:<15}")

    print("\n" + "="*80)


def test_alternative_approach(diarization_result):
    """Test alternative approach using timeline/segments directly."""

    print(f"\n{'='*80}")
    print("Testing Alternative Approach: Direct Segment Lookup")
    print(f"{'='*80}\n")

    # Convert to simple list of segments for easier lookup
    segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })

    print(f"Total segments: {len(segments)}\n")

    # Test finding speaker at specific timestamps
    test_timestamps = [1.07, 3.43, 6.81, 10.43, 13.43]

    print(f"{'Timestamp':<15} {'Speaker (overlap)':<20} {'Speaker (contains)':<20}")
    print("-" * 60)

    for ts in test_timestamps:
        # Find segments that overlap with timestamp
        overlapping = [s for s in segments if s['start'] <= ts <= s['end']]
        # Find segments that fully contain a small range around timestamp
        margin = 0.05  # 50ms margin
        containing = [s for s in segments if s['start'] <= ts - margin and s['end'] >= ts + margin]

        overlap_speakers = [s['speaker'] for s in overlapping]
        contain_speakers = [s['speaker'] for s in containing]

        print(f"{ts:<15.2f} {str(overlap_speakers):<20} {str(contain_speakers):<20}")

    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DIARIZATION DEBUG SCRIPT")
    print("="*80)
    print(f"Audio file: {AUDIO_FILE}")
    print(f"Expected speakers: {EXPECTED_SPEAKERS}")

    # Test 1: Default threshold (0.35 - same as baseline)
    diarization_result, num_speakers = test_diarization_with_threshold(0.35)

    # Test 2: Check .crop() functionality
    test_crop_functionality(diarization_result)

    # Test 3: Alternative approach
    test_alternative_approach(diarization_result)

    # Test 4: Try different thresholds if we didn't get 4 speakers
    if num_speakers != EXPECTED_SPEAKERS:
        print(f"\n\n{'='*80}")
        print(f"TRYING ALTERNATIVE THRESHOLDS (current: {num_speakers}, expected: {EXPECTED_SPEAKERS})")
        print(f"{'='*80}")

        for threshold in [0.30, 0.25, 0.40]:
            result, count = test_diarization_with_threshold(threshold)
            if count == EXPECTED_SPEAKERS:
                print(f"\n✅ SUCCESS: threshold={threshold} detected {count} speakers!")
                break

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80 + "\n")
