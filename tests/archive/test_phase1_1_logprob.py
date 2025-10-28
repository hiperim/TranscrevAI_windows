"""
Phase 1.1: Logprob Threshold Testing
Tests confidence filtering parameter to improve accuracy by filtering low-confidence segments.

Expected impact: +1-3% accuracy

Tests:
    1. logprob_threshold=-1.0 (baseline confidence filtering)
    2. logprob_threshold=-0.8 (stricter filtering - only high confidence)
    3. logprob_threshold=-1.5 (more permissive - allow lower confidence)

Usage:
    python tests/test_phase1_1_logprob.py
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
    """Run Phase 1.1 tests for logprob_threshold parameter."""

    logger.info("\n" + "="*80)
    logger.info("🧪 PHASE 1.1: LOGPROB THRESHOLD TESTING")
    logger.info("="*80 + "\n")

    # Test configurations
    test_configs = [
        {
            "name": "phase1_1_logprob_-1.0",
            "description": "Phase 1.1: log_prob_threshold=-1.0 (baseline - default)",
            "whisper_params": {
                "log_prob_threshold": -1.0
            }
        },
        {
            "name": "phase1_1_logprob_-0.5",
            "description": "Phase 1.1: log_prob_threshold=-0.5 (stricter filtering)",
            "whisper_params": {
                "log_prob_threshold": -0.5
            }
        },
        {
            "name": "phase1_1_logprob_-1.5",
            "description": "Phase 1.1: log_prob_threshold=-1.5 (more permissive)",
            "whisper_params": {
                "log_prob_threshold": -1.5
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
    logger.info("📊 PHASE 1.1 COMPARISON")
    logger.info("="*80 + "\n")

    logger.info(f"{'Configuration':<35} {'Accuracy':<12} {'WER':<10} {'Confidence':<12} {'Speed':<10}")
    logger.info(f"{'-'*35} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    for item in results:
        config_name = item['config']['name'].replace('phase1_1_', '')
        avg = item['result']['averages']
        logger.info(
            f"{config_name:<35} "
            f"{avg['transcription_accuracy_percent']:>10.2f}%  "
            f"{avg['transcription_wer']:>8.4f}  "
            f"{avg['transcription_confidence']:>10.4f}  "
            f"{avg['processing_ratio']:>9}"
        )

    logger.info("")

    # Find best configuration
    best = max(results, key=lambda x: x['result']['averages']['transcription_accuracy_percent'])
    best_config = best['config']['name'].replace('phase1_1_', '')
    best_accuracy = best['result']['averages']['transcription_accuracy_percent']

    logger.info(f"🏆 BEST CONFIGURATION: {best_config}")
    logger.info(f"   Accuracy: {best_accuracy:.2f}%")
    logger.info(f"   Parameters: {best['config']['whisper_params']}")
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
    logger.info("✅ Phase 1.1 testing complete!")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
