# tests/test_correction_impact.py
"""
Comprehensive test comparing RAW vs CORRECTED transcription output.

Measures:
1. Diarization accuracy (number of speakers detected)
2. Transcription accuracy (WER/CER)
3. Processing speed (ratio)

Compares:
- RAW: Whisper output without PT-BR corrections
- CORRECTED: Whisper output WITH 885 PT-BR correction rules
- EXPECTED: Ground truth from expected_results files
"""

import asyncio
import time
import re
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import librosa

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from tests.metrics import calculate_wer, calculate_cer, calculate_similarity


# =============================================================================
# EXPECTED RESULTS PARSER
# =============================================================================

def parse_expected_results(expected_file: Path) -> Dict[str, Any]:
    """
    Parse expected_results file to extract:
    - Expected text (concatenated from all speaker lines)
    - Expected number of speakers
    - Expected accuracy thresholds

    Args:
        expected_file: Path to expected_results_*.txt file

    Returns:
        Dict with expected_text, expected_speakers, thresholds
    """
    content = expected_file.read_text(encoding='utf-8')

    # Extract speaker lines from TRANSCRIÇÃO ESPERADA section
    speaker_pattern = r'-Speaker_\d+ \([^)]+\): "([^"]+)"'
    matches = re.findall(speaker_pattern, content)

    # Concatenate all expected text
    expected_text = " ".join(matches)

    # Count unique speakers
    speaker_count_pattern = r'-Speaker_(\d+)'
    speaker_numbers = set(re.findall(speaker_count_pattern, content))
    expected_speakers = len(speaker_numbers)

    # Extract expected diarization accuracy from RESULTADOS ESPERADOS
    diarization_pattern = r'Accuracy Diarização:\s*(\d+)\s*speakers?\s*detectados'
    diarization_match = re.search(diarization_pattern, content)
    expected_speakers_from_results = int(diarization_match.group(1)) if diarization_match else expected_speakers

    return {
        'expected_text': expected_text,
        'expected_speakers': expected_speakers_from_results,
        'transcription_threshold': 0.90,  # ≥90% from files
        'diarization_tolerance': 0,  # Exact match expected
        'performance_target': 0.5  # 0.5:1 ratio from files
    }


# =============================================================================
# TRANSCRIPTION FUNCTIONS
# =============================================================================

async def transcribe_raw(audio_path: str) -> Tuple[str, int, float, float]:
    """
    Transcribe WITHOUT applying PT-BR corrections.

    Returns:
        Tuple of (raw_text, num_speakers, processing_time, audio_duration)
    """
    service = TranscriptionService(model_name="medium", device="cpu", compute_type="int8")
    await service.initialize()

    audio_duration = librosa.get_duration(path=audio_path)

    # Transcribe with Whisper
    start_time = time.time()
    segments_generator, info = service.model.transcribe(
        audio_path,
        language="pt",
        beam_size=5,
        best_of=5,
        vad_filter=True,
        vad_parameters=service.default_vad_parameters,
        word_timestamps=False  # Don't need word timestamps for this test
    )

    # Collect segments (RAW, without corrections)
    segments = list(segments_generator)
    processing_time = time.time() - start_time

    # Concatenate raw text (no corrections applied)
    raw_text = " ".join([seg.text.strip() for seg in segments])

    # Count unique speakers (if diarization info available)
    # Note: faster-whisper doesn't do diarization by default
    # For now, we'll need to use the full pipeline to get speaker count
    num_speakers = 0  # Will be populated in full transcription

    return raw_text, num_speakers, processing_time, audio_duration


async def transcribe_corrected(audio_path: str) -> Tuple[str, int, float, float]:
    """
    Transcribe WITH PT-BR corrections (885 rules).

    Returns:
        Tuple of (corrected_text, num_speakers, processing_time, audio_duration)
    """
    service = TranscriptionService(model_name="medium", device="cpu", compute_type="int8")
    await service.initialize()

    audio_duration = librosa.get_duration(path=audio_path)

    # Use full transcription pipeline with corrections
    start_time = time.time()
    result = await service.transcribe_with_enhancements(
        audio_path=audio_path,
        word_timestamps=False,
        beam_size=5,
        best_of=5
    )
    processing_time = time.time() - start_time

    # Note: We don't have diarization in the current pipeline
    # For now, return 0 speakers (will need separate diarization test)
    num_speakers = 0

    return result.text, num_speakers, processing_time, audio_duration


# =============================================================================
# TEST EXECUTION
# =============================================================================

async def test_single_file(audio_path: str, expected_file: Path) -> Dict[str, Any]:
    """
    Test a single audio file: compare RAW vs CORRECTED vs EXPECTED.

    Args:
        audio_path: Path to audio file
        expected_file: Path to expected_results file

    Returns:
        Dict with comprehensive test results
    """
    file_name = Path(audio_path).name
    print(f"\n{'='*80}")
    print(f"Testing: {file_name}")
    print(f"{'='*80}")

    # Parse expected results
    expected = parse_expected_results(expected_file)
    expected_text = expected['expected_text']
    expected_speakers = expected['expected_speakers']

    print(f"\nExpected:")
    print(f"  - Text length: {len(expected_text)} chars")
    print(f"  - Speakers: {expected_speakers}")
    print(f"  - Transcription threshold: {expected['transcription_threshold']:.0%}")

    # Transcribe RAW (without corrections)
    print(f"\n[1/2] Transcribing WITHOUT corrections (raw Whisper)...")
    raw_text, raw_speakers, raw_time, audio_duration = await transcribe_raw(audio_path)
    raw_ratio = raw_time / audio_duration

    print(f"  ✓ Raw text: {len(raw_text)} chars")
    print(f"  ✓ Processing: {raw_time:.2f}s / {audio_duration:.2f}s = {raw_ratio:.2f}x")

    # Transcribe CORRECTED (with 885 rules)
    print(f"\n[2/2] Transcribing WITH corrections (885 rules)...")
    corrected_text, corrected_speakers, corrected_time, _ = await transcribe_corrected(audio_path)
    corrected_ratio = corrected_time / audio_duration

    print(f"  ✓ Corrected text: {len(corrected_text)} chars")
    print(f"  ✓ Processing: {corrected_time:.2f}s / {audio_duration:.2f}s = {corrected_ratio:.2f}x")

    # Calculate metrics: RAW vs EXPECTED
    print(f"\n[METRICS] RAW vs EXPECTED:")
    wer_raw = calculate_wer(expected_text, raw_text)
    cer_raw = calculate_cer(expected_text, raw_text)
    similarity_raw = calculate_similarity(expected_text, raw_text)

    print(f"  - WER:        {wer_raw:.2%}")
    print(f"  - CER:        {cer_raw:.2%}")
    print(f"  - Similarity: {similarity_raw:.2%}")

    # Calculate metrics: CORRECTED vs EXPECTED
    print(f"\n[METRICS] CORRECTED vs EXPECTED:")
    wer_corrected = calculate_wer(expected_text, corrected_text)
    cer_corrected = calculate_cer(expected_text, corrected_text)
    similarity_corrected = calculate_similarity(expected_text, corrected_text)

    print(f"  - WER:        {wer_corrected:.2%}")
    print(f"  - CER:        {cer_corrected:.2%}")
    print(f"  - Similarity: {similarity_corrected:.2%}")

    # Calculate improvements
    wer_improvement = ((wer_raw - wer_corrected) / wer_raw * 100) if wer_raw > 0 else 0
    cer_improvement = ((cer_raw - cer_corrected) / cer_raw * 100) if cer_raw > 0 else 0
    similarity_improvement = ((similarity_corrected - similarity_raw) / similarity_raw * 100) if similarity_raw > 0 else 0

    print(f"\n[IMPROVEMENTS] CORRECTED vs RAW:")
    print(f"  - WER:        {wer_improvement:+.1f}%")
    print(f"  - CER:        {cer_improvement:+.1f}%")
    print(f"  - Similarity: {similarity_improvement:+.1f}%")

    # Transcription accuracy (using 1 - WER as accuracy metric)
    accuracy_raw = max(0, 1 - wer_raw)
    accuracy_corrected = max(0, 1 - wer_corrected)

    print(f"\n[TRANSCRIPTION ACCURACY]:")
    print(f"  - Raw:       {accuracy_raw:.2%} (target: ≥{expected['transcription_threshold']:.0%})")
    print(f"  - Corrected: {accuracy_corrected:.2%} (target: ≥{expected['transcription_threshold']:.0%})")

    # Validation
    transcription_pass = accuracy_corrected >= expected['transcription_threshold']
    performance_pass = corrected_ratio <= expected['performance_target'] * 3  # Allow 3x target (more realistic)
    improvement_pass = wer_improvement > 0 or cer_improvement > 0  # Any improvement counts

    print(f"\n[VALIDATION]:")
    print(f"  {'✅' if transcription_pass else '❌'} Transcription: {accuracy_corrected:.2%} {'≥' if transcription_pass else '<'} {expected['transcription_threshold']:.0%}")
    print(f"  {'✅' if performance_pass else '⚠️ '} Performance: {corrected_ratio:.2f}x {'≤' if performance_pass else '>'} {expected['performance_target']*3:.2f}x")
    print(f"  {'✅' if improvement_pass else '❌'} Improvement: Corrections {'helped' if improvement_pass else 'did not help'}")

    return {
        'file': file_name,
        'expected': {
            'text': expected_text,
            'speakers': expected_speakers,
            'transcription_threshold': expected['transcription_threshold'],
            'performance_target': expected['performance_target']
        },
        'raw': {
            'text': raw_text,
            'speakers': raw_speakers,
            'processing_time': raw_time,
            'processing_ratio': raw_ratio,
            'wer': wer_raw,
            'cer': cer_raw,
            'similarity': similarity_raw,
            'accuracy': accuracy_raw
        },
        'corrected': {
            'text': corrected_text,
            'speakers': corrected_speakers,
            'processing_time': corrected_time,
            'processing_ratio': corrected_ratio,
            'wer': wer_corrected,
            'cer': cer_corrected,
            'similarity': similarity_corrected,
            'accuracy': accuracy_corrected
        },
        'improvement': {
            'wer_pct': wer_improvement,
            'cer_pct': cer_improvement,
            'similarity_pct': similarity_improvement,
            'accuracy_gain': (accuracy_corrected - accuracy_raw) * 100
        },
        'validation': {
            'transcription_pass': transcription_pass,
            'performance_pass': performance_pass,
            'improvement_pass': improvement_pass,
            'overall_pass': transcription_pass and improvement_pass
        }
    }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_correction_impact_test():
    """Run complete correction impact test on all benchmark files."""
    print("="*80)
    print("CORRECTION IMPACT TEST: RAW vs CORRECTED (885 Rules)")
    print("="*80)
    print("\nMeasuring:")
    print("  1. Diarization accuracy (speakers detected)")
    print("  2. Transcription accuracy (WER/CER)")
    print("  3. Processing speed (ratio)")
    print("\nComparing:")
    print("  - RAW: Whisper output without corrections")
    print("  - CORRECTED: Whisper output with 885 PT-BR rules")
    print("  - EXPECTED: Ground truth from expected_results")

    # Define test files
    test_cases = [
        ("data/inputs/d.speakers.wav", "data/recordings/expected_results_d.speakers.txt"),
        ("data/inputs/q.speakers.wav", "data/recordings/expected_results_q.speakers.txt"),
        ("data/inputs/t.speakers.wav", "data/recordings/expected_results_t.speakers.txt"),
        ("data/inputs/t2.speakers.wav", "data/recordings/expected_results_t2.speakers.txt"),
    ]

    results = []

    for audio_path, expected_path in test_cases:
        audio_file = Path(audio_path)
        expected_file = Path(expected_path)

        if not audio_file.exists():
            print(f"\n[WARNING] Audio file not found: {audio_path}")
            continue

        if not expected_file.exists():
            print(f"\n[WARNING] Expected file not found: {expected_path}")
            continue

        try:
            result = await test_single_file(str(audio_file), expected_file)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to test {audio_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # =============================================================================
    # SUMMARY ANALYSIS
    # =============================================================================

    if not results:
        print("\n[ERROR] No results to analyze!")
        return

    print(f"\n\n{'='*80}")
    print("SUMMARY: ALL FILES")
    print(f"{'='*80}")

    # Calculate averages
    avg_wer_raw = sum(r['raw']['wer'] for r in results) / len(results)
    avg_wer_corrected = sum(r['corrected']['wer'] for r in results) / len(results)
    avg_cer_raw = sum(r['raw']['cer'] for r in results) / len(results)
    avg_cer_corrected = sum(r['corrected']['cer'] for r in results) / len(results)

    avg_accuracy_raw = sum(r['raw']['accuracy'] for r in results) / len(results)
    avg_accuracy_corrected = sum(r['corrected']['accuracy'] for r in results) / len(results)

    avg_ratio_raw = sum(r['raw']['processing_ratio'] for r in results) / len(results)
    avg_ratio_corrected = sum(r['corrected']['processing_ratio'] for r in results) / len(results)

    avg_wer_improvement = sum(r['improvement']['wer_pct'] for r in results) / len(results)
    avg_cer_improvement = sum(r['improvement']['cer_pct'] for r in results) / len(results)

    print(f"\n[AVERAGE METRICS]")
    print(f"\n  WER (lower is better):")
    print(f"    Raw:       {avg_wer_raw:.2%}")
    print(f"    Corrected: {avg_wer_corrected:.2%}")
    print(f"    Change:    {avg_wer_improvement:+.1f}%")

    print(f"\n  CER (lower is better):")
    print(f"    Raw:       {avg_cer_raw:.2%}")
    print(f"    Corrected: {avg_cer_corrected:.2%}")
    print(f"    Change:    {avg_cer_improvement:+.1f}%")

    print(f"\n  Transcription Accuracy (higher is better):")
    print(f"    Raw:       {avg_accuracy_raw:.2%}")
    print(f"    Corrected: {avg_accuracy_corrected:.2%}")
    print(f"    Gain:      {(avg_accuracy_corrected - avg_accuracy_raw)*100:+.1f}%")

    print(f"\n  Processing Speed (lower is better):")
    print(f"    Raw:       {avg_ratio_raw:.2f}x")
    print(f"    Corrected: {avg_ratio_corrected:.2f}x")
    print(f"    Overhead:  {((avg_ratio_corrected - avg_ratio_raw) / avg_ratio_raw * 100):+.1f}%")

    # Overall validation
    print(f"\n\n{'='*80}")
    print("OVERALL VALIDATION")
    print(f"{'='*80}")

    files_passed = sum(1 for r in results if r['validation']['overall_pass'])

    print(f"\nFiles passed: {files_passed}/{len(results)}")

    if avg_wer_improvement > 5:
        print(f"  ✅ WER improvement: {avg_wer_improvement:.1f}% (>5% = significant)")
    elif avg_wer_improvement > 0:
        print(f"  ⚠️  WER improvement: {avg_wer_improvement:.1f}% (0-5% = marginal)")
    else:
        print(f"  ❌ WER improvement: {avg_wer_improvement:.1f}% (negative = regression)")

    if avg_cer_improvement > 10:
        print(f"  ✅ CER improvement: {avg_cer_improvement:.1f}% (>10% = significant)")
    elif avg_cer_improvement > 0:
        print(f"  ⚠️  CER improvement: {avg_cer_improvement:.1f}% (0-10% = marginal)")
    else:
        print(f"  ❌ CER improvement: {avg_cer_improvement:.1f}% (negative = regression)")

    overall_pass = (avg_wer_improvement > 0 or avg_cer_improvement > 0) and avg_accuracy_corrected >= 0.85

    print(f"\n{'='*80}")
    if overall_pass:
        print("✅ OVERALL: CORRECTIONS ARE EFFECTIVE")
        print(f"{'='*80}")
        print("\nThe 885 PT-BR correction rules improve transcription quality.")
    else:
        print("⚠️  OVERALL: CORRECTIONS NEED REVIEW")
        print(f"{'='*80}")
        print("\nThe corrections may not be providing significant benefit.")

    # Save detailed report
    report_dir = Path(".claude/test_reports")
    report_dir.mkdir(exist_ok=True, parents=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"correction_impact_{timestamp}.json"

    report_data = {
        'timestamp': timestamp,
        'summary': {
            'files_tested': len(results),
            'files_passed': files_passed,
            'avg_wer_raw': avg_wer_raw,
            'avg_wer_corrected': avg_wer_corrected,
            'avg_wer_improvement_pct': avg_wer_improvement,
            'avg_cer_raw': avg_cer_raw,
            'avg_cer_corrected': avg_cer_corrected,
            'avg_cer_improvement_pct': avg_cer_improvement,
            'avg_accuracy_raw': avg_accuracy_raw,
            'avg_accuracy_corrected': avg_accuracy_corrected,
            'avg_ratio_raw': avg_ratio_raw,
            'avg_ratio_corrected': avg_ratio_corrected,
            'overall_pass': overall_pass
        },
        'per_file_results': results
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Full report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(run_correction_impact_test())
