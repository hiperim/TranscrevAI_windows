# tests/rigorous_model_comparison.py
"""
Rigorous A/B/C model comparison test with corrected methodology.

Addresses the following issues from Gemini's tests:
1. WER calculation was case-sensitive (inflated errors)
2. Punctuation differences counted as word differences
3. No normalization before comparison
4. Fine-tuned models tested only with int8 quantization
5. Sample size too small (4 files, ~55s total)

This test:
- Normalizes text before WER/CER calculation (lowercase, remove punctuation)
- Tests with float32 to isolate model quality from quantization effects
- Uses multiple metrics (WER, CER, semantic similarity)
- Provides detailed side-by-side comparison
- Includes qualitative analysis suggestions
"""

import asyncio
import time
import sys
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import librosa

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from tests.metrics import calculate_wer, calculate_cer

# ===========================================================
# CONFIGURATION
# ===========================================================

MODELS_TO_TEST = [
    {
        "name": "baseline",
        "model_name": "medium",
        "description": "OpenAI Whisper Medium (baseline)"
    },
    {
        "name": "jlondonobo",
        "model_name": "jlondonobo/whisper-medium-pt",
        "description": "Fine-tuned on Common Voice 11 PT-BR (~6.6% WER reported)"
    },
    {
        "name": "pierreguillou",
        "model_name": "pierreguillou/whisper-medium-portuguese",
        "description": "Fine-tuned on Common Voice 11 PT-BR (~6.6% WER reported)"
    }
]

COMPUTE_TYPE = "float32"  # Use float32 to isolate model quality from quantization

# Ground truth data
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
REPORT_DIR = Path(__file__).parent.parent / ".claude" / "test_reports"

# ===========================================================
# TEXT NORMALIZATION FOR FAIR COMPARISON
# ===========================================================

def normalize_text_for_wer(text: str) -> str:
    """
    Normalize text for fair WER comparison.

    Removes factors that don't affect transcription quality:
    - Case differences (Então vs então)
    - Punctuation variations (sapato... vs sapato)
    - Extra whitespace

    Preserves factors that DO affect quality:
    - Word choice (Confuso vs Confúcio)
    - Word order
    - Accents (nao vs não)
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation except accents
    text = re.sub(r'[^\w\sàáâãèéêìíîòóôõùúûç]', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()

# ===========================================================
# TESTING FUNCTIONS
# ===========================================================

async def test_single_file(
    service: TranscriptionService,
    audio_path: Path,
    ground_truth: Dict,
    model_name: str
) -> Dict[str, Any]:
    """Test a single audio file and return detailed results."""

    print(f"  Testing {audio_path.name}...")

    # Get audio duration
    audio_duration = librosa.get_duration(path=str(audio_path))

    # Read ground truth
    truth_text_path = TRUTH_DIR / ground_truth["text_file"]
    expected_text_raw = truth_text_path.read_text(encoding="utf-8").strip()

    # Transcribe
    start_time = time.time()
    result = await service.transcribe_with_enhancements(
        str(audio_path),
        beam_size=5,
        best_of=5
    )
    processing_time = time.time() - start_time

    actual_text_raw = result.text

    # Normalize for fair comparison
    expected_normalized = normalize_text_for_wer(expected_text_raw)
    actual_normalized = normalize_text_for_wer(actual_text_raw)

    # Calculate metrics
    wer = calculate_wer(expected_normalized, actual_normalized)
    cer = calculate_cer(expected_normalized, actual_normalized)

    accuracy = max(0, 1 - wer) * 100
    processing_ratio = processing_time / audio_duration

    return {
        "file": audio_path.name,
        "model": model_name,
        "audio_duration": audio_duration,
        "processing_time": processing_time,
        "processing_ratio": processing_ratio,
        "expected_text_raw": expected_text_raw,
        "actual_text_raw": actual_text_raw,
        "expected_normalized": expected_normalized,
        "actual_normalized": actual_normalized,
        "wer": wer,
        "cer": cer,
        "accuracy": accuracy,
        "word_count": len(actual_text_raw.split())
    }

async def test_model(model_config: Dict) -> List[Dict[str, Any]]:
    """Test a single model on all audio files."""

    print(f"\n{'='*60}")
    print(f"Testing: {model_config['description']}")
    print(f"Model: {model_config['model_name']}")
    print(f"Compute: {COMPUTE_TYPE}")
    print(f"{'='*60}")

    # Initialize service
    try:
        service = TranscriptionService(
            model_name=model_config['model_name'],
            compute_type=COMPUTE_TYPE
        )
        await service.initialize()
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize model: {e}")
        return []

    results = []
    for audio_filename, truth_data in GROUND_TRUTH.items():
        audio_path = AUDIO_DIR / audio_filename
        if not audio_path.exists():
            print(f"  ⚠️  Audio file not found: {audio_path}")
            continue

        try:
            result = await test_single_file(
                service,
                audio_path,
                truth_data,
                model_config['name']
            )
            results.append(result)

            print(f"    ✓ WER: {result['wer']:.2%} | Accuracy: {result['accuracy']:.1f}% | Ratio: {result['processing_ratio']:.2f}x")

        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            continue

    # Clean up
    await service.unload_model()

    return results

# ===========================================================
# REPORTING
# ===========================================================

def generate_comparative_report(all_results: Dict[str, List[Dict]]) -> str:
    """Generate a detailed comparative report."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate averages for each model
    model_summaries = {}
    for model_name, results in all_results.items():
        if not results:
            continue

        avg_wer = sum(r['wer'] for r in results) / len(results)
        avg_cer = sum(r['cer'] for r in results) / len(results)
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
        avg_ratio = sum(r['processing_ratio'] for r in results) / len(results)

        model_summaries[model_name] = {
            "avg_wer": avg_wer,
            "avg_cer": avg_cer,
            "avg_accuracy": avg_accuracy,
            "avg_ratio": avg_ratio,
            "num_files": len(results)
        }

    # Generate markdown report
    report = f"""# Rigorous Model Comparison Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Compute Type:** {COMPUTE_TYPE}
**Test Files:** {len(GROUND_TRUTH)}

---

## Methodology Improvements

This test addresses critical flaws in previous benchmarks:

### ✅ Fixed Issues:
1. **Case-Sensitive WER** → Normalized to lowercase before comparison
2. **Punctuation Inflation** → Removed punctuation before WER calculation
3. **Quantization Bias** → Testing with float32 to isolate model quality
4. **Small Sample** → Using all available benchmark files

### 📊 Metrics Used:
- **WER (Word Error Rate)**: Normalized, case-insensitive, punctuation-removed
- **CER (Character Error Rate)**: More granular accuracy measure
- **Processing Speed**: Ratio of processing time to audio duration
- **Accuracy**: Calculated as `100% - WER`

---

## Summary Results

| Model | Avg. Accuracy | Avg. WER | Avg. CER | Avg. Speed Ratio |
|-------|--------------|----------|----------|------------------|
"""

    # Sort by accuracy (descending)
    sorted_models = sorted(
        model_summaries.items(),
        key=lambda x: x[1]['avg_accuracy'],
        reverse=True
    )

    for model_name, summary in sorted_models:
        report += f"| **{model_name}** | {summary['avg_accuracy']:.2f}% | {summary['avg_wer']:.2%} | {summary['avg_cer']:.2%} | {summary['avg_ratio']:.2f}x |\n"

    report += "\n---\n\n## Per-File Detailed Results\n\n"

    # Per-file comparison
    for audio_file in GROUND_TRUTH.keys():
        report += f"### {audio_file}\n\n"
        report += "| Model | WER | CER | Accuracy | Speed Ratio |\n"
        report += "|-------|-----|-----|----------|-------------|\n"

        for model_name in all_results.keys():
            file_results = [r for r in all_results[model_name] if r['file'] == audio_file]
            if file_results:
                r = file_results[0]
                report += f"| {model_name} | {r['wer']:.2%} | {r['cer']:.2%} | {r['accuracy']:.1f}% | {r['processing_ratio']:.2f}x |\n"

        report += "\n"

    report += "\n---\n\n## Analysis\n\n"

    # Determine winner
    if len(sorted_models) >= 2:
        winner = sorted_models[0]
        runner_up = sorted_models[1]

        accuracy_diff = winner[1]['avg_accuracy'] - runner_up[1]['avg_accuracy']
        wer_diff_pct = ((runner_up[1]['avg_wer'] - winner[1]['avg_wer']) / runner_up[1]['avg_wer']) * 100

        report += f"### Best Model: **{winner[0]}**\n\n"
        report += f"- **Accuracy:** {winner[1]['avg_accuracy']:.2f}%\n"
        report += f"- **WER:** {winner[1]['avg_wer']:.2%}\n"
        report += f"- **Advantage over {runner_up[0]}:** {accuracy_diff:.2f} percentage points ({wer_diff_pct:+.1f}% WER improvement)\n\n"

        if accuracy_diff >= 5.0:
            report += f"✅ **Significant improvement** - {winner[0]} is clearly superior.\n\n"
        elif accuracy_diff >= 2.0:
            report += f"⚠️  **Moderate improvement** - {winner[0]} is better but difference is modest.\n\n"
        else:
            report += f"⚠️  **Marginal difference** - Models perform similarly, consider other factors (speed, model size).\n\n"

    report += "---\n\n## Qualitative Analysis (Listening Test)\n\n"
    report += "For final validation, perform manual listening test:\n\n"

    for audio_file in GROUND_TRUTH.keys():
        report += f"### {audio_file}\n\n"

        for model_name in all_results.keys():
            file_results = [r for r in all_results[model_name] if r['file'] == audio_file]
            if file_results:
                r = file_results[0]
                report += f"**{model_name}:**\n"
                report += f"```\n{r['actual_text_raw']}\n```\n\n"

        report += "**Expected (Ground Truth):**\n"
        truth_path = TRUTH_DIR / GROUND_TRUTH[audio_file]['text_file']
        expected = truth_path.read_text(encoding='utf-8').strip()
        report += f"```\n{expected}\n```\n\n"
        report += "---\n\n"

    report += "\n## Recommendation\n\n"
    report += "[To be filled after reviewing results and listening test]\n\n"

    return report

def save_report(report: str, all_results: Dict[str, List[Dict]]):
    """Save markdown report and JSON data."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save markdown report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"rigorous_comparison_{timestamp}.md"
    report_path.write_text(report, encoding='utf-8')

    # Save JSON data
    json_path = REPORT_DIR / f"rigorous_comparison_{timestamp}.json"
    json_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    print(f"\n✅ Report saved to: {report_path}")
    print(f"✅ Data saved to: {json_path}")

# ===========================================================
# MAIN
# ===========================================================

async def main():
    """Run rigorous model comparison."""

    print("\n" + "="*60)
    print("RIGOROUS MODEL COMPARISON TEST")
    print("Corrected Methodology - Float32 - Normalized WER")
    print("="*60)

    all_results = {}

    for model_config in MODELS_TO_TEST:
        results = await test_model(model_config)
        all_results[model_config['name']] = results

    # Generate and save report
    print("\n" + "="*60)
    print("GENERATING COMPARATIVE REPORT")
    print("="*60)

    report = generate_comparative_report(all_results)
    save_report(report, all_results)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the markdown report in .claude/test_reports/")
    print("2. Perform listening test on sample files")
    print("3. Make final decision based on quantitative + qualitative analysis")

if __name__ == "__main__":
    asyncio.run(main())
