"""
Phase 2.2: VAD Threshold Testing
Tests if adjusting VAD sensitivity threshold improves transcription accuracy.

Expected impact: +0.5-1% accuracy

Tests:
    1. threshold=0.4 (more sensitive - captures more speech, risk of noise)
    2. threshold=0.5 (current baseline)
    3. threshold=0.6 (less sensitive - filters noise, risk of losing words)

Usage:
    python tests/test_phase2_2_vad_threshold.py
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
    """Run Phase 2.2 tests for VAD threshold parameter."""

    logger.info("\n" + "="*80)
    logger.info("🧪 PHASE 2.2: VAD THRESHOLD TESTING")
    logger.info("="*80 + "\n")

    # Test configurations
    test_configs = [
        {
            "name": "phase2_2_vad_threshold_0.4",
            "description": "Phase 2.2: VAD threshold=0.4 (more sensitive)",
            "whisper_params": {
                "vad_parameters": {
                    "threshold": 0.4
                }
            }
        },
        {
            "name": "phase2_2_vad_threshold_0.5",
            "description": "Phase 2.2: VAD threshold=0.5 (baseline)",
            "whisper_params": {
                "vad_parameters": {
                    "threshold": 0.5
                }
            }
        },
        {
            "name": "phase2_2_vad_threshold_0.6",
            "description": "Phase 2.2: VAD threshold=0.6 (less sensitive)",
            "whisper_params": {
                "vad_parameters": {
                    "threshold": 0.6
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
    logger.info("📊 PHASE 2.2 COMPARISON")
    logger.info("="*80 + "\n")

    logger.info(f"{'Configuration':<40} {'Accuracy':<12} {'WER':<10} {'Confidence':<12} {'Speed':<10}")
    logger.info(f"{'-'*40} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    for item in results:
        config_name = item['config']['name'].replace('phase2_2_', '')
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
    best_config = best['config']['name'].replace('phase2_2_', '')
    best_accuracy = best['result']['averages']['transcription_accuracy_percent']

    logger.info(f"🏆 BEST CONFIGURATION: {best_config}")
    logger.info(f"   Accuracy: {best_accuracy:.2f}%")
    logger.info(f"   VAD threshold: {best['config']['whisper_params']['vad_parameters']['threshold']}")
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
    logger.info("✅ Phase 2.2 testing complete!")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
