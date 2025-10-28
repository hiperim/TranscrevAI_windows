"""
Phase 2.3: VAD min_speech_duration_ms Testing
Tests if filtering short speech segments improves transcription accuracy.

Expected impact: +0.5-1% accuracy (by filtering noise)

Tests:
    1. min_speech_duration_ms=250 (current baseline)
    2. min_speech_duration_ms=500 (filter short noises more aggressively)

Usage:
    python tests/test_phase2_3_vad_speech_duration.py
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_dual_audio_baseline import run_dual_audio_baseline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run Phase 2.3 tests for VAD min_speech_duration_ms parameter."""

    logger.info("\n" + "="*80)
    logger.info("🧪 PHASE 2.3: VAD MIN_SPEECH_DURATION_MS TESTING")
    logger.info("="*80 + "\n")

    # Test configurations
    test_configs = [
        {
            "name": "phase2_3_speech_250ms",
            "description": "Phase 2.3: min_speech_duration_ms=250 (baseline)",
            "whisper_params": {
                "vad_parameters": {
                    "min_speech_duration_ms": 250
                }
            }
        },
        {
            "name": "phase2_3_speech_500ms",
            "description": "Phase 2.3: min_speech_duration_ms=500 (filter short noises)",
            "whisper_params": {
                "vad_parameters": {
                    "min_speech_duration_ms": 500
                }
            }
        }
    ]

    results = []

    for i, config in enumerate(test_configs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 TEST {i}/{len(test_configs)}: {config['name']}")
        logger.info(f"{'='*80}\n")

        result = await run_dual_audio_baseline(
            phase_name=config["name"],
            description=config["description"],
            test_config=config
        )

        results.append({
            "config": config,
            "result": result
        })

        logger.info(f"\n✅ Test {i} complete!\n")

    # Compare results
    logger.info("\n" + "="*80)
    logger.info("📊 PHASE 2.3 COMPARISON")
    logger.info("="*80 + "\n")

    logger.info(f"{'Configuration':<40} {'Accuracy':<12} {'WER':<10} {'Confidence':<12} {'Speed':<10}")
    logger.info(f"{'-'*40} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    for item in results:
        config_name = item['config']['name'].replace('phase2_3_', '')
        avg = item['result']['averages']
        logger.info(
            f"{config_name:<40} "
            f"{avg['transcription_accuracy_percent']:>10.2f}%  "
            f"{avg['transcription_wer']:>8.4f}  "
            f"{avg['transcription_confidence']:>10.4f}  "
            f"{avg['processing_ratio']:>9}"
        )

    logger.info("")

    # Find best configuration
    best = max(results, key=lambda x: x['result']['averages']['transcription_accuracy_percent'])
    best_config = best['config']['name'].replace('phase2_3_', '')
    best_accuracy = best['result']['averages']['transcription_accuracy_percent']

    logger.info(f"🏆 BEST CONFIGURATION: {best_config}")
    logger.info(f"   Accuracy: {best_accuracy:.2f}%")
    logger.info(f"   VAD min_speech_duration: {best['config']['whisper_params']['vad_parameters']['min_speech_duration_ms']}ms")
    logger.info("")

    # Calculate improvement from baseline
    baseline_accuracy = 74.52  # From phase0_dual_baseline_results
    improvement = best_accuracy - baseline_accuracy

    if improvement > 0:
        logger.info(f"📈 IMPROVEMENT: +{improvement:.2f}pp from baseline ({baseline_accuracy:.2f}% → {best_accuracy:.2f}%)")
    elif improvement < 0:
        logger.info(f"📉 REGRESSION: {improvement:.2f}pp from baseline ({baseline_accuracy:.2f}% → {best_accuracy:.2f}%)")
    else:
        logger.info(f"➡️  NO CHANGE: Accuracy remains at {baseline_accuracy:.2f}%")

    logger.info("\n" + "="*80)
    logger.info("✅ Phase 2.3 testing complete!")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
