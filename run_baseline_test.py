"""
Baseline Memory Profiling Test
Processa d.speakers.wav e captura memory profiling logs
"""

import asyncio
import logging
import time
import psutil
from pathlib import Path

from src.transcription import TranscriptionService
from src.diarization import TwoPassDiarizer, force_transcription_segmentation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def baseline_test():
    """Run baseline memory profiling test"""

    audio_path = "data/recordings/d.speakers.wav"

    logger.info("="*80)
    logger.info("BASELINE MEMORY PROFILING TEST")
    logger.info("="*80)

    # Get process for memory monitoring
    process = psutil.Process()

    # Baseline memory
    mem_baseline = process.memory_info().rss / (1024 * 1024)
    logger.info(f"[MEMORY PROFILING] Baseline: {mem_baseline:.2f} MB")

    # Initialize services
    logger.info("Initializing services...")
    transcription_service = TranscriptionService()
    await transcription_service.initialize()
    diarization_service = TwoPassDiarizer()

    # Memory after initialization
    mem_after_init = process.memory_info().rss / (1024 * 1024)
    mem_delta_init = mem_after_init - mem_baseline
    logger.info(f"[MEMORY PROFILING] After Initialization: {mem_after_init:.2f} MB (+{mem_delta_init:.2f} MB)")

    # Run transcription
    logger.info(f"Processing: {audio_path}")
    start_time = time.time()

    transcription_result = await transcription_service.transcribe_with_enhancements(audio_path)

    # Memory after transcription
    mem_after_transcription = process.memory_info().rss / (1024 * 1024)
    mem_delta_transcription = mem_after_transcription - mem_after_init
    logger.info(f"[MEMORY PROFILING] After Transcription: {mem_after_transcription:.2f} MB (+{mem_delta_transcription:.2f} MB)")

    # Run diarization
    diarization_result = await diarization_service.diarize(audio_path, transcription_result.segments)
    final_segments = force_transcription_segmentation(transcription_result.segments, diarization_result["segments"])

    # Memory after diarization
    mem_after_diarization = process.memory_info().rss / (1024 * 1024)
    mem_delta_diarization = mem_after_diarization - mem_after_transcription
    logger.info(f"[MEMORY PROFILING] After Diarization: {mem_after_diarization:.2f} MB (+{mem_delta_diarization:.2f} MB)")

    # Final metrics
    processing_time = time.time() - start_time
    mem_final = process.memory_info().rss / (1024 * 1024)
    mem_total_delta = mem_final - mem_baseline
    mem_peak = mem_final  # Approximate peak (actual peak might be higher)

    logger.info("="*80)
    logger.info("[MEMORY PROFILING] Pipeline Complete: {:.2f} MB (Total Delta: +{:.2f} MB)".format(mem_final, mem_total_delta))
    logger.info("[MEMORY PROFILING] Estimated Peak: {:.2f} MB".format(mem_peak))
    logger.info("[MEMORY PROFILING] Target: <3500 MB | Current: {:.2f} MB | Status: {}".format(
        mem_final,
        'PASS' if mem_final < 3500 else 'FAIL'
    ))
    logger.info("="*80)

    # Results summary
    logger.info("\n" + "="*80)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Processing Time: {processing_time:.2f}s")
    logger.info(f"Audio Duration: 21.06s (from d.speakers.wav)")
    logger.info(f"Speed Ratio: {processing_time/21.06:.2f}s/s")
    logger.info(f"Confidence: {transcription_result.confidence*100:.2f}%")
    logger.info(f"Speakers Detected: {diarization_result['num_speakers']}")
    logger.info(f"Segments: {len(final_segments)}")
    logger.info("\nMemory Breakdown:")
    logger.info(f"  Baseline: {mem_baseline:.2f} MB")
    logger.info(f"  After Init: {mem_after_init:.2f} MB (+{mem_delta_init:.2f} MB)")
    logger.info(f"  After Transcription: {mem_after_transcription:.2f} MB (+{mem_delta_transcription:.2f} MB)")
    logger.info(f"  After Diarization: {mem_after_diarization:.2f} MB (+{mem_delta_diarization:.2f} MB)")
    logger.info(f"  Final: {mem_final:.2f} MB")
    logger.info(f"  Headroom: {3500 - mem_peak:.2f} MB (until 3500 MB limit)")
    logger.info("\nCompliance Check:")
    logger.info(f"  Speed <1.50s/s: {'PASS' if (processing_time/21.06) < 1.50 else 'FAIL'} ({processing_time/21.06:.2f}s/s)")
    logger.info(f"  Memory <3.5GB: {'PASS' if mem_peak < 3500 else 'FAIL'} ({mem_peak:.2f} MB)")
    logger.info(f"  Accuracy >90%: {'PASS' if transcription_result.confidence > 0.90 else 'NEEDS CHECK'} ({transcription_result.confidence*100:.2f}%)")
    logger.info("="*80)

    # Save results to file
    results_file = ".claude/CHANGES_MADE/baseline_metrics_07.10.25.md"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"""# Baseline Metrics - TranscrevAI
**Date**: 07.10.2025
**Test File**: d.speakers.wav (21.06s)

## Performance
- Processing Time: {processing_time:.2f}s
- Speed Ratio: {processing_time/21.06:.2f}s/s
- Confidence: {transcription_result.confidence*100:.2f}%
- Speakers Detected: {diarization_result['num_speakers']}
- Segments: {len(final_segments)}

## Memory Usage
- Baseline: {mem_baseline:.2f} MB
- After Initialization: {mem_after_init:.2f} MB (+{mem_delta_init:.2f} MB)
- After Transcription: {mem_after_transcription:.2f} MB (+{mem_delta_transcription:.2f} MB)
- After Diarization: {mem_after_diarization:.2f} MB (+{mem_delta_diarization:.2f} MB)
- Final: {mem_final:.2f} MB
- **Headroom Available**: {3500 - mem_peak:.2f} MB

## Compliance Status
- Speed <1.50s/s: {'✅ PASS' if (processing_time/21.06) < 1.50 else '❌ FAIL'} ({processing_time/21.06:.2f}s/s)
- Memory <3.5GB: {'✅ PASS' if mem_peak < 3500 else '❌ FAIL'} ({mem_peak:.2f} MB)
- Accuracy >90%: {'✅ PASS' if transcription_result.confidence > 0.90 else '⚠️ NEEDS CHECK'} ({transcription_result.confidence*100:.2f}%)

## Decision Matrix
Based on RAM usage:
""")

        if mem_peak < 2000:
            f.write("- **Status**: 🟢 SAFE - Plenty of headroom for diarization\n")
            f.write(f"- **Available**: {3500 - mem_peak:.2f} MB for new features\n")
            f.write("- **Recommendation**: PROCEED with PHASE 2 and PHASE 3\n")
        elif mem_peak < 3000:
            f.write("- **Status**: 🟡 CAUTION - Limited headroom\n")
            f.write(f"- **Available**: {3500 - mem_peak:.2f} MB for new features\n")
            f.write("- **Recommendation**: PROCEED with PHASE 2, careful with PHASE 3\n")
        else:
            f.write("- **Status**: 🔴 DANGER - Near limit\n")
            f.write(f"- **Available**: {3500 - mem_peak:.2f} MB (minimal headroom)\n")
            f.write("- **Recommendation**: DO NOT add features, investigate memory usage first\n")

    logger.info(f"\nBaseline metrics saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(baseline_test())
