# tests/test_ab_safe_optimization.py
"""
A/B Test: BASELINE vs SAFE_OPTIMIZATION
Compares baseline configuration with safe PT-BR optimized parameters.

Expected outcome:
- Accuracy: ≥82.5% (within 0.3% margin of 82.81% baseline)
- Speed: +5-15% improvement (lower processing ratio)
- Memory: Similar or better
"""

import asyncio
import time
import sys
import re
import json
from pathlib import Path
import librosa
from difflib import SequenceMatcher
from typing import Dict, Any, List

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService

# =============================================================================
# TEST CONFIGURATIONS
# =============================================================================

BASELINE_CONFIG = {
    "name": "BASELINE",
    "model_name": "medium",
    "compute_type": "int8",
    "beam_size": 5,
    "best_of": 5,
    "vad_threshold": 0.5,
    "vad_min_speech_duration_ms": 250,
    "vad_min_silence_duration_ms": 2000,
    # No additional whisper params - use defaults
}

SAFE_OPTIMIZATION_CONFIG = {
    "name": "SAFE_OPTIMIZATION",
    "model_name": "medium",
    "compute_type": "int8",
    "beam_size": 5,  # KEEP baseline
    "best_of": 5,    # KEEP baseline
    "vad_threshold": 0.5,
    "vad_min_speech_duration_ms": 250,
    "vad_min_silence_duration_ms": 2000,
    # SAFE OPTIMIZATION PARAMETERS (only faster-whisper supported params)
    "whisper_params": {
        "temperature": 0.0,  # KEEP baseline (deterministic)
        "condition_on_previous_text": True,  # KEEP baseline (context)
        "compression_ratio_threshold": 1.6,  # vs default 2.4 - PT-BR optimized
        "no_speech_threshold": 0.85,         # vs default 0.6 - more aggressive
        "hallucination_silence_threshold": 0.8,  # Reduce hallucinations (if supported)
        "suppress_tokens": [-1, 50256],      # Extended suppression
        "repetition_penalty": 1.0,           # Minimal repetition control
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_benchmark_files() -> List[str]:
    """Returns the 4 main benchmark files (same as test_accuracy_all_files.py)."""
    inputs_dir = Path(__file__).parent.parent / "data" / "inputs"

    # Use only the 4 main benchmark files (excluding test.wav which is corrupted)
    benchmark_names = ["d.speakers.wav", "q.speakers.wav", "t.speakers.wav", "t2.speakers.wav"]

    valid_files = []
    for name in benchmark_names:
        file_path = inputs_dir / name
        if file_path.exists():
            valid_files.append(str(file_path))
        else:
            print(f"[WARNING] Benchmark file not found: {file_path}")

    if not valid_files:
        print("[ERROR] No benchmark audio files found in data/inputs/.")
        return []
    return valid_files

# =============================================================================
# TEST EXECUTION
# =============================================================================

async def run_single_config_test(config: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    """Run transcription for a single configuration on a single file."""
    file_name = Path(file_path).name
    print(f"  Testing {config['name']}: {file_name}...")

    # Get audio duration
    audio_duration = librosa.get_duration(path=file_path)

    # Initialize service
    transcription_service = TranscriptionService(
        model_name=config["model_name"],
        device="cpu",
        compute_type=config["compute_type"]
    )
    await transcription_service.initialize()

    # Prepare VAD parameters
    vad_params = {
        "threshold": config["vad_threshold"],
        "min_speech_duration_ms": config["vad_min_speech_duration_ms"],
        "min_silence_duration_ms": config["vad_min_silence_duration_ms"],
    }

    # Prepare transcription parameters
    transcribe_params = {
        "audio_path": file_path,
        "word_timestamps": True,
        "beam_size": config["beam_size"],
        "best_of": config["best_of"],
        "vad_parameters": vad_params,
    }

    # Add SAFE OPTIMIZATION parameters if present
    if "whisper_params" in config:
        transcribe_params.update(config["whisper_params"])

    # Run transcription
    start_time = time.time()
    result = await transcription_service.transcribe_with_enhancements(**transcribe_params)
    end_time = time.time()

    processing_time = end_time - start_time
    processing_ratio = processing_time / audio_duration
    actual_text = result.text.strip()

    # Calculate accuracy using WORD PROBABILITY (same as baseline test)
    all_words = []
    for segment in result.segments:
        if 'words' in segment and segment['words']:
            for word in segment['words']:
                all_words.append({
                    'word': word['word'].strip(),
                    'probability': word.get('probability', 0.0)
                })

    # Accuracy = average word probability (same metric as baseline 82.81%)
    avg_word_probability = sum(w['probability'] for w in all_words) / len(all_words) if all_words else 0.0

    return {
        "file": file_name,
        "config": config["name"],
        "duration": audio_duration,
        "processing_time": processing_time,
        "processing_ratio": processing_ratio,
        "accuracy": avg_word_probability,  # Now using word probability like baseline
        "word_count": len(all_words),
        "actual_text": actual_text,
    }

async def run_ab_test():
    """Run complete A/B test comparing BASELINE vs SAFE_OPTIMIZATION."""
    print("=" * 80)
    print("A/B TEST: BASELINE vs SAFE_OPTIMIZATION")
    print("=" * 80)
    print()

    # Get benchmark files
    benchmark_files = get_benchmark_files()
    if not benchmark_files:
        print("[ERROR] No benchmark files found. Exiting.")
        return

    print(f"Found {len(benchmark_files)} benchmark files:\n")
    for f in benchmark_files:
        print(f"  - {Path(f).name}")
    print()

    # Run tests for both configurations
    results = []

    for config in [BASELINE_CONFIG, SAFE_OPTIMIZATION_CONFIG]:
        print(f"\n{'='*80}")
        print(f"Testing Configuration: {config['name']}")
        print(f"{'='*80}\n")

        if "whisper_params" in config:
            print("Whisper optimization parameters:")
            for key, value in config["whisper_params"].items():
                print(f"  {key}: {value}")
            print()

        for audio_path in benchmark_files:
            result = await run_single_config_test(config, audio_path)
            results.append(result)

            print(f"    ✓ {result['file']}: {result['accuracy']:.2%} accuracy, {result['processing_ratio']:.2f}x ratio")

    # =============================================================================
    # COMPARATIVE ANALYSIS
    # =============================================================================

    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")

    # Group results by file
    files_comparison = {}
    for result in results:
        file_name = result["file"]
        if file_name not in files_comparison:
            files_comparison[file_name] = {}
        files_comparison[file_name][result["config"]] = result

    # Calculate averages
    baseline_avg_accuracy = 0.0
    baseline_avg_ratio = 0.0
    optimization_avg_accuracy = 0.0
    optimization_avg_ratio = 0.0
    file_count = len(files_comparison)

    print("Per-File Comparison:\n")
    print(f"{'File':<20} | {'Config':<20} | {'Accuracy':<10} | {'Ratio':<8} | {'Delta Acc':<10} | {'Delta Ratio':<12}")
    print("-" * 100)

    for file_name in sorted(files_comparison.keys()):
        baseline = files_comparison[file_name]["BASELINE"]
        optimization = files_comparison[file_name]["SAFE_OPTIMIZATION"]

        delta_accuracy = (optimization["accuracy"] - baseline["accuracy"]) * 100
        delta_ratio = ((optimization["processing_ratio"] - baseline["processing_ratio"]) / baseline["processing_ratio"]) * 100

        # Print baseline
        print(f"{file_name:<20} | {'BASELINE':<20} | {baseline['accuracy']*100:>9.2f}% | {baseline['processing_ratio']:>7.2f}x | {'':<10} | {'':<12}")

        # Print optimization with deltas
        acc_symbol = "✅" if delta_accuracy >= -0.3 else "❌"  # Accept 0.3% loss
        ratio_symbol = "✅" if delta_ratio < 0 else "⚠️"  # Lower is better

        print(f"{'':<20} | {'SAFE_OPTIMIZATION':<20} | {optimization['accuracy']*100:>9.2f}% | {optimization['processing_ratio']:>7.2f}x | {delta_accuracy:>+9.2f}% {acc_symbol} | {delta_ratio:>+10.2f}% {ratio_symbol}")
        print()

        # Accumulate for averages
        baseline_avg_accuracy += baseline["accuracy"]
        baseline_avg_ratio += baseline["processing_ratio"]
        optimization_avg_accuracy += optimization["accuracy"]
        optimization_avg_ratio += optimization["processing_ratio"]

    # Calculate and display averages
    baseline_avg_accuracy /= file_count
    baseline_avg_ratio /= file_count
    optimization_avg_accuracy /= file_count
    optimization_avg_ratio /= file_count

    avg_delta_accuracy = (optimization_avg_accuracy - baseline_avg_accuracy) * 100
    avg_delta_ratio = ((optimization_avg_ratio - baseline_avg_ratio) / baseline_avg_ratio) * 100

    print("=" * 100)
    print("AVERAGE RESULTS:\n")
    print(f"{'Configuration':<20} | {'Accuracy':<12} | {'Processing Ratio':<18} | {'Delta Acc':<12} | {'Delta Ratio':<12}")
    print("-" * 100)
    print(f"{'BASELINE':<20} | {baseline_avg_accuracy*100:>11.2f}% | {baseline_avg_ratio:>17.2f}x | {'':<12} | {'':<12}")
    print(f"{'SAFE_OPTIMIZATION':<20} | {optimization_avg_accuracy*100:>11.2f}% | {optimization_avg_ratio:>17.2f}x | {avg_delta_accuracy:>+11.2f}% | {avg_delta_ratio:>+11.2f}%")
    print("=" * 100)

    # =============================================================================
    # VALIDATION
    # =============================================================================

    print(f"\n{'='*80}")
    print("VALIDATION")
    print(f"{'='*80}\n")

    # Accuracy validation
    accuracy_threshold = 82.5  # Allow 0.3% loss from 82.81% baseline
    accuracy_pass = (optimization_avg_accuracy * 100) >= accuracy_threshold

    print(f"Accuracy Check:")
    print(f"  Target: ≥{accuracy_threshold}%")
    print(f"  Achieved: {optimization_avg_accuracy*100:.2f}%")
    print(f"  Status: {'✅ PASS' if accuracy_pass else '❌ FAIL'}\n")

    # Speed improvement validation
    speed_improved = avg_delta_ratio < 0

    print(f"Speed Improvement Check:")
    print(f"  Baseline: {baseline_avg_ratio:.2f}x")
    print(f"  Optimized: {optimization_avg_ratio:.2f}x")
    print(f"  Improvement: {avg_delta_ratio:+.2f}%")
    print(f"  Status: {'✅ IMPROVED' if speed_improved else '⚠️ NO IMPROVEMENT'}\n")

    # Overall validation
    overall_pass = accuracy_pass

    print(f"{'='*80}")
    print(f"OVERALL VALIDATION: {'✅ PASS - SAFE_OPTIMIZATION APPROVED' if overall_pass else '❌ FAIL - KEEP BASELINE'}")
    print(f"{'='*80}")

    # =============================================================================
    # SAVE REPORT
    # =============================================================================

    report_dir = Path(__file__).parent.parent / ".claude" / "test_reports"
    report_dir.mkdir(exist_ok=True, parents=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"ab_test_safe_optimization_{timestamp}.json"

    report_data = {
        "timestamp": timestamp,
        "baseline": {
            "avg_accuracy": baseline_avg_accuracy,
            "avg_ratio": baseline_avg_ratio,
            "config": BASELINE_CONFIG
        },
        "safe_optimization": {
            "avg_accuracy": optimization_avg_accuracy,
            "avg_ratio": optimization_avg_ratio,
            "config": SAFE_OPTIMIZATION_CONFIG
        },
        "comparison": {
            "avg_delta_accuracy_pct": avg_delta_accuracy,
            "avg_delta_ratio_pct": avg_delta_ratio,
        },
        "validation": {
            "accuracy_pass": accuracy_pass,
            "speed_improved": speed_improved,
            "overall_pass": overall_pass
        },
        "per_file_results": results
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Full report saved to: {report_path}")
    print()

if __name__ == "__main__":
    asyncio.run(run_ab_test())
