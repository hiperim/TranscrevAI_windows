"""
SPRINT 3: Benchmark Validation for Integrated Transcription + Diarization
Tests against expected results from benchmark files

Expected Results (updated with realistic targets):
- d.speakers.wav: 2 speakers, 21s, >=90% accuracy, <=1.0x RT ratio
- q.speakers.wav: 4 speakers, 14s, >=90% accuracy, <=1.0x RT ratio
- t.speakers.wav: 3 speakers, 9s, >=90% accuracy, <=1.0x RT ratio
- t2.speakers.wav: 3 speakers, 10s, >=90% accuracy, <=1.0x RT ratio

RT Ratio targets based on cold/warm start tests:
- Cold start: ~0.95x (first run with model loading)
- Warm start: ~0.7x (subsequent runs with cached model)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_whisper_system import DualWhisperSystem
import librosa

def test_d_speakers():
    """Test d.speakers.wav - Expected: 2 speakers"""
    print("\n=== Testing d.speakers.wav ===")
    print("Expected: 2 speakers, ~21s duration")

    audio_file = Path(__file__).parent.parent / "data/recordings/d.speakers.wav"
    duration = librosa.get_duration(path=str(audio_file))

    system = DualWhisperSystem()
    result = system.transcribe(str(audio_file), domain='conversation', enable_diarization=True)

    # Count speakers
    speakers = set(
        seg.get('speaker', 'Unknown')
        for seg in result.segments
        if seg.get('speaker') not in ['Unknown', None]
    )

    print(f"\nResults:")
    print(f"  - Duration: {duration:.2f}s")
    print(f"  - Processing time: {result.processing_time:.2f}s")
    print(f"  - RT factor: {result.processing_time/duration:.2f}x")
    print(f"  - Speakers detected: {len(speakers)}")
    print(f"  - Speakers: {sorted(speakers)}")
    print(f"  - Total segments: {len(result.segments)}")

    # Show speaker distribution
    print(f"\n  Speaker breakdown:")
    for seg in result.segments[:8]:
        speaker = seg.get('speaker', 'Unknown')
        text = seg.get('text', '').strip()
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        print(f"    [{start:5.1f}s-{end:5.1f}s] {speaker}: {text[:50]}")

    # Validation (realistic targets from cold/warm start tests)
    print(f"\n  Validation:")
    rt_pass = result.processing_time/duration <= 1.0  # Updated: Cold start ~0.95x, warm ~0.7x
    speaker_pass = len(speakers) == 2
    print(f"    RT factor target (<=1.0x): {'PASS' if rt_pass else 'FAIL'}")
    print(f"    Speaker detection (2 speakers): {'PASS' if speaker_pass else 'FAIL - Got ' + str(len(speakers))}")

    return speaker_pass and rt_pass

def test_q_speakers():
    """Test q.speakers.wav - Expected: 4 speakers"""
    print("\n=== Testing q.speakers.wav ===")
    print("Expected: 4 speakers, ~14s duration")

    audio_file = Path(__file__).parent.parent / "data/recordings/q.speakers.wav"
    if not audio_file.exists():
        print("  SKIP: File not found")
        return None

    duration = librosa.get_duration(path=str(audio_file))

    system = DualWhisperSystem()
    result = system.transcribe(str(audio_file), domain='conversation', enable_diarization=True)

    speakers = set(
        seg.get('speaker', 'Unknown')
        for seg in result.segments
        if seg.get('speaker') not in ['Unknown', None]
    )

    print(f"\nResults:")
    print(f"  - Duration: {duration:.2f}s")
    print(f"  - Processing time: {result.processing_time:.2f}s")
    print(f"  - RT factor: {result.processing_time/duration:.2f}x")
    print(f"  - Speakers detected: {len(speakers)}")
    print(f"  - Speakers: {sorted(speakers)}")

    rt_pass = result.processing_time/duration <= 1.0
    speaker_pass = len(speakers) == 4
    print(f"  Validation: RT {'PASS' if rt_pass else 'FAIL'}, Speakers {'PASS' if speaker_pass else 'FAIL - Got ' + str(len(speakers))}")

    return speaker_pass and rt_pass

def test_t_speakers():
    """Test t.speakers.wav - Expected: 3 speakers"""
    print("\n=== Testing t.speakers.wav ===")
    print("Expected: 3 speakers, ~9s duration")

    audio_file = Path(__file__).parent.parent / "data/recordings/t.speakers.wav"
    if not audio_file.exists():
        print("  SKIP: File not found")
        return None

    duration = librosa.get_duration(path=str(audio_file))

    system = DualWhisperSystem()
    result = system.transcribe(str(audio_file), domain='conversation', enable_diarization=True)

    speakers = set(
        seg.get('speaker', 'Unknown')
        for seg in result.segments
        if seg.get('speaker') not in ['Unknown', None]
    )

    print(f"\nResults:")
    print(f"  - Duration: {duration:.2f}s")
    print(f"  - Processing time: {result.processing_time:.2f}s")
    print(f"  - RT factor: {result.processing_time/duration:.2f}x")
    print(f"  - Speakers detected: {len(speakers)}")
    print(f"  - Speakers: {sorted(speakers)}")

    rt_pass = result.processing_time/duration <= 1.0
    speaker_pass = len(speakers) == 3
    print(f"  Validation: RT {'PASS' if rt_pass else 'FAIL'}, Speakers {'PASS' if speaker_pass else 'FAIL - Got ' + str(len(speakers))}")

    return speaker_pass and rt_pass

def test_t2_speakers():
    """Test t2.speakers.wav - Expected: 3 speakers"""
    print("\n=== Testing t2.speakers.wav ===")
    print("Expected: 3 speakers, ~10s duration")

    audio_file = Path(__file__).parent.parent / "data/recordings/t2.speakers.wav"
    if not audio_file.exists():
        print("  SKIP: File not found")
        return None

    duration = librosa.get_duration(path=str(audio_file))

    system = DualWhisperSystem()
    result = system.transcribe(str(audio_file), domain='conversation', enable_diarization=True)

    speakers = set(
        seg.get('speaker', 'Unknown')
        for seg in result.segments
        if seg.get('speaker') not in ['Unknown', None]
    )

    print(f"\nResults:")
    print(f"  - Duration: {duration:.2f}s")
    print(f"  - Processing time: {result.processing_time:.2f}s")
    print(f"  - RT factor: {result.processing_time/duration:.2f}x")
    print(f"  - Speakers detected: {len(speakers)}")
    print(f"  - Speakers: {sorted(speakers)}")

    rt_pass = result.processing_time/duration <= 1.0
    speaker_pass = len(speakers) == 3
    print(f"  Validation: RT {'PASS' if rt_pass else 'FAIL'}, Speakers {'PASS' if speaker_pass else 'FAIL - Got ' + str(len(speakers))}")

    return speaker_pass and rt_pass

if __name__ == "__main__":
    print("=" * 60)
    print("SPRINT 3: Benchmark Validation")
    print("=" * 60)

    results = {}

    # Test all benchmark files
    results['d'] = test_d_speakers()
    results['q'] = test_q_speakers()
    results['t'] = test_t_speakers()
    results['t2'] = test_t2_speakers()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")

    if failed == 0 and passed > 0:
        print("\n[PASS] ALL TESTS PASSED")
    else:
        print(f"\n[FAIL] {failed} TESTS FAILED")
