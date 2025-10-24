"""
Phase 0: Audio Normalization Testing
Tests if LUFS normalization improves transcription accuracy.

Expected impact: +1-3% accuracy

Tests normalized audio (-23 LUFS) vs original audio.

Usage:
    python tests/test_phase0_audio_normalization.py
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_dual_audio_baseline import run_dual_audio_baseline, AUDIO_FILES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run Phase 0 tests for audio normalization."""

    logger.info("\n" + "="*80)
    logger.info("🧪 PHASE 0: AUDIO NORMALIZATION (LUFS) TESTING")
    logger.info("="*80 + "\n")

    # Test configurations
    results = []

    # Test 1: Original audio (baseline)
    logger.info("\n" + "="*80)
    logger.info("📊 TEST 1/2: Original Audio (Baseline)")
    logger.info("="*80 + "\n")

    baseline_result = await run_dual_audio_baseline(
        phase_name="phase0_original_audio",
        description="Phase 0: Original audio (baseline for comparison)",
        test_config={}
    )

    results.append({
        "name": "Original Audio",
        "result": baseline_result
    })

    logger.info("\n✅ Test 1 complete!\n")

    # Test 2: Normalized audio (-23 LUFS)
    logger.info("\n" + "="*80)
    logger.info("📊 TEST 2/2: Normalized Audio (-23 LUFS)")
    logger.info("="*80 + "\n")

    # Create custom audio config pointing to normalized files
    normalized_audio_files = []
    for audio_config in AUDIO_FILES:
        normalized_config = audio_config.copy()
        # Point to normalized version
        original_path = Path(audio_config["path"])
        normalized_path = original_path.parent / f"{original_path.stem}_normalized{original_path.suffix}"
        normalized_config["path"] = normalized_path
        normalized_audio_files.append(normalized_config)

    # Temporarily modify the AUDIO_FILES for this test
    import tests.test_dual_audio_baseline as baseline_module
    original_audio_files = baseline_module.AUDIO_FILES
    baseline_module.AUDIO_FILES = normalized_audio_files

    try:
        normalized_result = await run_dual_audio_baseline(
            phase_name="phase0_normalized_lufs_-23",
            description="Phase 0: Normalized audio (-23 LUFS broadcast standard)",
            test_config={}
        )

        results.append({
            "name": "Normalized (-23 LUFS)",
            "result": normalized_result
        })
    finally:
        # Restore original AUDIO_FILES
        baseline_module.AUDIO_FILES = original_audio_files

    logger.info("\n✅ Test 2 complete!\n")

    # Compare results
    logger.info("\n" + "="*80)
    logger.info("📊 PHASE 0 COMPARISON")
    logger.info("="*80 + "\n")

    logger.info(f"{'Configuration':<30} {'Accuracy':<12} {'WER':<10} {'Confidence':<12} {'Speed':<10}")
    logger.info(f"{'-'*30} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    for item in results:
        name = item['name']
        avg = item['result']['averages']
        logger.info(
            f"{name:<30} "
            f"{avg['transcription_accuracy_percent']:>10.2f}%  "
            f"{avg['transcription_wer']:>8.4f}  "
            f"{avg['transcription_confidence']:>10.4f}  "
            f"{avg['processing_ratio']:>9}"
        )

    logger.info("")

    # Calculate improvement
    original_accuracy = results[0]['result']['averages']['transcription_accuracy_percent']
    normalized_accuracy = results[1]['result']['averages']['transcription_accuracy_percent']
    improvement = normalized_accuracy - original_accuracy

    baseline_accuracy = 74.52  # From phase0_dual_baseline_results

    logger.info(f"📊 DETAILED ANALYSIS:")
    logger.info(f"")
    logger.info(f"   Original Audio: {original_accuracy:.2f}%")
    logger.info(f"   Normalized Audio: {normalized_accuracy:.2f}%")
    logger.info(f"")

    if improvement > 0:
        logger.info(f"✅ IMPROVEMENT: +{improvement:.2f}pp with LUFS normalization")
        logger.info(f"   {original_accuracy:.2f}% → {normalized_accuracy:.2f}%")
        logger.info(f"")
        logger.info(f"📈 Total improvement from original baseline ({baseline_accuracy:.2f}%): +{normalized_accuracy - baseline_accuracy:.2f}pp")
    elif improvement < 0:
        logger.info(f"❌ REGRESSION: {improvement:.2f}pp with LUFS normalization")
        logger.info(f"   Normalization made accuracy worse!")
    else:
        logger.info(f"➡️  NO CHANGE: LUFS normalization had no impact on accuracy")

    logger.info("")
    logger.info("="*80)
    logger.info("✅ Phase 0 testing complete!")
    logger.info("="*80 + "\n")

    # Recommendation
    if improvement >= 1.0:
        logger.info("💡 RECOMMENDATION: Keep LUFS normalization - significant improvement!")
    elif improvement > 0:
        logger.info("⚠️  RECOMMENDATION: Small improvement - consider keeping for consistency")
    else:
        logger.info("❌ RECOMMENDATION: Skip LUFS normalization - no benefit for these audios")


if __name__ == "__main__":
    asyncio.run(main())
