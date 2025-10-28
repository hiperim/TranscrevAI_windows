"""
Script de Profiling: Diagnóstico de Performance por Arquivo
Objetivo: Identificar gargalo específico do d.speakers.wav (ratio 2.92x)

Dia 1 - Manhã: Diagnóstico Crítico
"""

import time
import psutil
import asyncio
import logging
import sys
from pathlib import Path

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def profile_file(audio_path: str, model_name: str = "medium"):
    """
    Perfila um arquivo de áudio específico, separando tempo de transcription e diarization.
    """
    process = psutil.Process()
    mem_start = process.memory_info().rss / (1024 * 1024)  # MB

    logger.info(f"\n{'='*80}")
    logger.info(f"PROFILING: {audio_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Memory baseline: {mem_start:.2f} MB")
    logger.info(f"{'='*80}")

    try:
        # Obter duração do áudio
        import librosa
        duration = librosa.get_duration(path=audio_path)
        logger.info(f"Audio duration: {duration:.2f}s")

        # ===== TRANSCRIPTION PROFILING =====
        logger.info("\n--- TRANSCRIPTION PHASE ---")
        trans_service = TranscriptionService(model_name=model_name, device="cpu")
        await trans_service.initialize()

        trans_start = time.time()
        mem_before_trans = process.memory_info().rss / (1024 * 1024)

        result = await trans_service.transcribe_with_enhancements(
            audio_path,
            word_timestamps=True
        )

        trans_time = time.time() - trans_start
        mem_after_trans = process.memory_info().rss / (1024 * 1024)
        trans_ratio = trans_time / duration if duration > 0 else 0

        logger.info(f"Transcription completed:")
        logger.info(f"  Time: {trans_time:.2f}s")
        logger.info(f"  Ratio: {trans_ratio:.2f}x")
        logger.info(f"  Segments: {len(result.segments)}")
        logger.info(f"  Words: {result.word_count}")
        logger.info(f"  Memory: {mem_before_trans:.2f} MB → {mem_after_trans:.2f} MB (Δ {mem_after_trans - mem_before_trans:.2f} MB)")

        # ===== DIARIZATION PROFILING =====
        logger.info("\n--- DIARIZATION PHASE ---")
        diar_service = PyannoteDiarizer(device="cpu")

        diar_start = time.time()
        mem_before_diar = process.memory_info().rss / (1024 * 1024)

        diar_result = await diar_service.diarize(audio_path, result.segments)

        diar_time = time.time() - diar_start
        mem_after_diar = process.memory_info().rss / (1024 * 1024)
        diar_ratio = diar_time / duration if duration > 0 else 0

        logger.info(f"Diarization completed:")
        logger.info(f"  Time: {diar_time:.2f}s")
        logger.info(f"  Ratio: {diar_ratio:.2f}x")
        logger.info(f"  Speakers detected: {diar_result['num_speakers']}")
        logger.info(f"  Memory: {mem_before_diar:.2f} MB → {mem_after_diar:.2f} MB (Δ {mem_after_diar - mem_before_diar:.2f} MB)")

        # ===== TOTAL ANALYSIS =====
        total_time = trans_time + diar_time
        total_ratio = total_time / duration if duration > 0 else 0
        peak_mem = max(mem_after_trans, mem_after_diar)

        logger.info(f"\n--- TOTAL RESULTS ---")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Total ratio: {total_ratio:.2f}x")
        logger.info(f"Peak memory: {peak_mem:.2f} MB ({peak_mem/1024:.2f} GB)")
        logger.info(f"\nBreakdown:")
        logger.info(f"  Transcription: {trans_time:.2f}s ({trans_time/total_time*100:.1f}%)")
        logger.info(f"  Diarization:   {diar_time:.2f}s ({diar_time/total_time*100:.1f}%)")

        # Análise específica
        if total_ratio > 2.0:
            logger.warning(f"\n⚠️  PERFORMANCE ISSUE: Ratio {total_ratio:.2f}x exceeds target (<2.0x)")
            if trans_ratio > diar_ratio:
                logger.warning(f"   → Bottleneck: TRANSCRIPTION ({trans_ratio:.2f}x)")
                logger.warning(f"   → Recommendation: Consider faster model or VAD optimization")
            else:
                logger.warning(f"   → Bottleneck: DIARIZATION ({diar_ratio:.2f}x)")
                logger.warning(f"   → Recommendation: Optimize PyAnnote batch size or clustering")
        else:
            logger.info(f"\n✅ Performance OK: Ratio {total_ratio:.2f}x within target")

        return {
            "file": Path(audio_path).name,
            "duration": duration,
            "trans_time": trans_time,
            "trans_ratio": trans_ratio,
            "diar_time": diar_time,
            "diar_ratio": diar_ratio,
            "total_time": total_time,
            "total_ratio": total_ratio,
            "peak_mem_mb": peak_mem,
            "num_speakers": diar_result['num_speakers']
        }

    except Exception as e:
        logger.error(f"Error profiling {audio_path}: {e}", exc_info=True)
        return None

async def profile_all_files():
    """
    Perfila todos os 4 arquivos de benchmark e gera relatório comparativo.
    """
    benchmark_dir = Path("data/inputs")
    files = [
        "d.speakers.wav",
        "q.speakers.wav",
        "t.speakers.wav",
        "t2.speakers.wav"
    ]

    results = []
    for filename in files:
        audio_path = benchmark_dir / filename
        if not audio_path.exists():
            logger.error(f"File not found: {audio_path}")
            continue

        result = await profile_file(str(audio_path))
        if result:
            results.append(result)

        # Pequena pausa entre arquivos para estabilizar memória
        logger.info("\n" + "="*80 + "\n")
        await asyncio.sleep(2)

    # ===== COMPARATIVE ANALYSIS =====
    if results:
        logger.info("\n" + "="*80)
        logger.info("COMPARATIVE ANALYSIS")
        logger.info("="*80 + "\n")

        # Tabela formatada
        print(f"{'File':<20} {'Duration':>10} {'Trans':>8} {'Diar':>8} {'Total':>8} {'Ratio':>8} {'Peak RAM':>10} {'Speakers':>10}")
        print("-" * 100)

        for r in results:
            print(f"{r['file']:<20} {r['duration']:>9.2f}s {r['trans_time']:>7.2f}s {r['diar_time']:>7.2f}s {r['total_time']:>7.2f}s {r['total_ratio']:>7.2f}x {r['peak_mem_mb']:>9.2f}MB {r['num_speakers']:>10}")

        # Estatísticas
        avg_ratio = sum(r['total_ratio'] for r in results) / len(results)
        max_ratio = max(r['total_ratio'] for r in results)
        max_ratio_file = max(results, key=lambda x: x['total_ratio'])['file']

        print("-" * 100)
        print(f"{'AVERAGE':<20} {'':<10} {'':<8} {'':<8} {'':<8} {avg_ratio:>7.2f}x")
        print(f"{'MAX (worst)':<20} {max_ratio_file:<10} {'':<8} {'':<8} {'':<8} {max_ratio:>7.2f}x")

        # Análise de padrões
        logger.info("\n--- PATTERN ANALYSIS ---")

        # Ordenar por ratio
        sorted_by_ratio = sorted(results, key=lambda x: x['total_ratio'], reverse=True)
        sorted_by_duration = sorted(results, key=lambda x: x['duration'], reverse=True)

        logger.info("\nFiles by processing ratio (worst to best):")
        for i, r in enumerate(sorted_by_ratio, 1):
            status = "❌ FAIL" if r['total_ratio'] > 2.0 else "✅ PASS"
            logger.info(f"  {i}. {r['file']}: {r['total_ratio']:.2f}x {status}")

        logger.info("\nFiles by duration (longest to shortest):")
        for i, r in enumerate(sorted_by_duration, 1):
            logger.info(f"  {i}. {r['file']}: {r['duration']:.2f}s (ratio: {r['total_ratio']:.2f}x)")

        # Correlação entre duração e ratio?
        if sorted_by_duration[0]['file'] == sorted_by_ratio[0]['file']:
            logger.warning("\n⚠️  PATTERN DETECTED: Longest file is also slowest")
            logger.warning("   → Possible issue: Cold start, or scaling problem with longer audio")

        # Comparar transcription vs diarization bottleneck
        trans_bottleneck = sum(1 for r in results if r['trans_ratio'] > r['diar_ratio'])
        diar_bottleneck = len(results) - trans_bottleneck

        logger.info(f"\nBottleneck analysis:")
        logger.info(f"  Transcription bottleneck: {trans_bottleneck}/{len(results)} files")
        logger.info(f"  Diarization bottleneck:   {diar_bottleneck}/{len(results)} files")

        if trans_bottleneck > diar_bottleneck:
            logger.info(f"\n💡 RECOMMENDATION: Focus on transcription optimization")
            logger.info(f"   → Consider: Whisper model change, VAD tuning, or compute optimization")
        elif diar_bottleneck > trans_bottleneck:
            logger.info(f"\n💡 RECOMMENDATION: Focus on diarization optimization")
            logger.info(f"   → Consider: PyAnnote batch size, clustering threshold, or embedding optimization")
        else:
            logger.info(f"\n💡 RECOMMENDATION: Balance optimization needed")
            logger.info(f"   → Consider: Both transcription and diarization need attention")

        logger.info("\n" + "="*80)
        logger.info("PROFILING COMPLETE")
        logger.info("="*80)

if __name__ == "__main__":
    logger.info("Starting performance profiling...")
    logger.info("This will take several minutes to complete.\n")
    asyncio.run(profile_all_files())
