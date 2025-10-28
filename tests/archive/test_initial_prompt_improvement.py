"""
A/B Test: initial_prompt PT-BR Impact
Objetivo: Validar improvement de accuracy com PT-BR vocabulary prompt

Comparação:
- Baseline: SEM initial_prompt (já testado: 82.81% avg accuracy)
- Test: COM initial_prompt (~220 tokens PT-BR vocabulary)

Expected: +5-10% accuracy improvement
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Baseline results (SEM initial_prompt) - do teste anterior
BASELINE_RESULTS = {
    "d.speakers.wav": {
        "avg_word_probability": 0.9191,
        "confidence": 0.7739,
        "word_count": 48,
        "low_confidence_count": 6,
        "accented_words_count": 8
    },
    "q.speakers.wav": {
        "avg_word_probability": 0.8620,
        "confidence": 0.7142,
        "word_count": 44,
        "low_confidence_count": 6,
        "accented_words_count": 12
    },
    "t.speakers.wav": {
        "avg_word_probability": 0.7549,
        "confidence": 0.6488,
        "word_count": 27,
        "low_confidence_count": 6,
        "accented_words_count": 2
    },
    "t2.speakers.wav": {
        "avg_word_probability": 0.7763,
        "confidence": 0.6641,
        "word_count": 13,
        "low_confidence_count": 3,
        "accented_words_count": 0
    }
}

BASELINE_AVG = 0.8281  # 82.81%

from typing import Optional

# ... (rest of imports)

async def test_with_prompt_single_file(audio_path: str) -> Optional[dict]:
    """
    Testa accuracy COM initial_prompt de um arquivo específico.
    """
    filename = Path(audio_path).name
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING WITH PROMPT: {filename}")
    logger.info(f"{'='*80}")

    try:
        # Obter duração
        import librosa
        duration = librosa.get_duration(path=audio_path)
        logger.info(f"Audio duration: {duration:.2f}s")

        # Inicializar serviço de transcrição (COM initial_prompt)
        logger.info(f"Initializing medium model WITH PT-BR initial_prompt...")
        trans_service = TranscriptionService(model_name="medium", device="cpu")
        await trans_service.initialize()

        # Transcrever com word timestamps
        logger.info("Transcribing with PT-BR vocabulary prompt...")
        result = await trans_service.transcribe_with_enhancements(
            audio_path,
            word_timestamps=True
        )

        # Extrair texto completo
        full_text = result.text
        logger.info(f"\nFULL TRANSCRIPTION:")
        logger.info(f"{'='*60}")
        logger.info(full_text)
        logger.info(f"{'='*60}\n")

        # Extrair todas as palavras individuais
        all_words = []
        for segment in result.segments:
            if 'words' in segment and segment['words']:
                for word in segment['words']:
                    all_words.append({
                        'word': word['word'].strip(),
                        'start': word['start'],
                        'end': word['end'],
                        'probability': word.get('probability', 0.0)
                    })

        # Calcular métricas
        avg_probability = sum(w['probability'] for w in all_words) / len(all_words) if all_words else 0
        low_confidence_words = [w for w in all_words if w['probability'] < 0.7]
        accented_words = [w for w in all_words if any(c in w['word'] for c in 'áàâãéêíóôõúüç')]

        # Comparar com baseline
        baseline = BASELINE_RESULTS.get(filename, {})
        baseline_prob = baseline.get("avg_word_probability", 0)
        improvement = avg_probability - baseline_prob
        improvement_pct = (improvement / baseline_prob * 100) if baseline_prob > 0 else 0

        logger.info(f"Metrics (WITH prompt):")
        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
        logger.info(f"  - Word count: {len(all_words)}")
        logger.info(f"  - Confidence: {result.confidence:.2%}")
        logger.info(f"  - Avg word probability: {avg_probability:.2%}")
        logger.info(f"  - Low confidence words (<70%): {len(low_confidence_words)}")
        logger.info(f"  - Accented PT-BR words: {len(accented_words)}")

        logger.info(f"\nComparison vs Baseline (WITHOUT prompt):")
        logger.info(f"  - Baseline avg prob: {baseline_prob:.2%}")
        logger.info(f"  - New avg prob: {avg_probability:.2%}")
        logger.info(f"  - Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

        status = "✅ IMPROVED" if improvement > 0 else "⚠️ NO CHANGE" if improvement == 0 else "❌ DEGRADED"
        logger.info(f"  - Status: {status}")

        # Listar palavras de baixa confiança
        if low_confidence_words:
            logger.info(f"\nLow Confidence Words (<70%):")
            for w in low_confidence_words[:5]:
                logger.info(f"  - \"{w['word']}\" ({w['probability']:.2%})")

        return {
            "filename": filename,
            "duration": duration,
            "processing_time": result.processing_time,
            "transcription": {
                "full_text": full_text,
                "word_count": len(all_words),
                "segment_count": len(result.segments),
                "confidence": result.confidence,
                "avg_word_probability": avg_probability
            },
            "words": all_words,
            "analysis": {
                "low_confidence_count": len(low_confidence_words),
                "low_confidence_words": low_confidence_words[:10],
                "accented_words_count": len(accented_words),
                "accented_words": accented_words[:10]
            },
            "comparison": {
                "baseline_avg_prob": baseline_prob,
                "new_avg_prob": avg_probability,
                "improvement": improvement,
                "improvement_percentage": improvement_pct
            }
        }

    except Exception as e:
        logger.error(f"Error testing {filename}: {e}", exc_info=True)
        return None

async def main():
    """
    Testa accuracy COM initial_prompt em todos os 4 arquivos de benchmark.
    """
    logger.info("="*80)
    logger.info("A/B TEST: initial_prompt PT-BR Impact")
    logger.info("Model: whisper-medium (int8, beam_size=5, best_of=5)")
    logger.info("Prompt: ~221 tokens PT-BR vocabulary (98.7% of limit)")
    logger.info("="*80 + "\n")

    files = [
        "data/inputs/d.speakers.wav",
        "data/inputs/q.speakers.wav",
        "data/inputs/t.speakers.wav",
        "data/inputs/t2.speakers.wav"
    ]

    results = []
    for audio_path in files:
        if not Path(audio_path).exists():
            logger.error(f"File not found: {audio_path}")
            continue

        result = await test_with_prompt_single_file(audio_path)
        if result:
            results.append(result)

        # Pausa entre arquivos
        logger.info("\n" + "="*80 + "\n")
        await asyncio.sleep(2)

    # ===== FINAL COMPARISON =====
    if results:
        logger.info("\n" + "="*80)
        logger.info("FINAL COMPARISON: Baseline vs With Prompt")
        logger.info("="*80 + "\n")

        # Tabela formatada
        print(f"{'File':<20} {'Baseline':>12} {'With Prompt':>12} {'Improvement':>12} {'% Change':>10}")
        print("-" * 80)

        for r in results:
            comp = r['comparison']
            print(f"{r['filename']:<20} {comp['baseline_avg_prob']:>11.2%} {comp['new_avg_prob']:>11.2%} {comp['improvement']:>+11.4f} {comp['improvement_percentage']:>+9.1f}%")

        # Estatísticas gerais
        avg_new = sum(r['comparison']['new_avg_prob'] for r in results) / len(results)
        avg_improvement = avg_new - BASELINE_AVG
        avg_improvement_pct = (avg_improvement / BASELINE_AVG * 100) if BASELINE_AVG > 0 else 0

        print("-" * 80)
        print(f"{'AVERAGE':<20} {BASELINE_AVG:>11.2%} {avg_new:>11.2%} {avg_improvement:>+11.4f} {avg_improvement_pct:>+9.1f}%")

        logger.info(f"\n--- FINAL ASSESSMENT ---")
        logger.info(f"Baseline (without prompt): {BASELINE_AVG:.2%}")
        logger.info(f"New (with prompt): {avg_new:.2%}")
        logger.info(f"Overall improvement: {avg_improvement:+.4f} ({avg_improvement_pct:+.1f}%)")

        # Análise de resultados
        improved_count = sum(1 for r in results if r['comparison']['improvement'] > 0)
        degraded_count = sum(1 for r in results if r['comparison']['improvement'] < 0)
        no_change_count = len(results) - improved_count - degraded_count

        logger.info(f"\nFile-level analysis:")
        logger.info(f"  - Improved: {improved_count}/{len(results)} files")
        logger.info(f"  - Degraded: {degraded_count}/{len(results)} files")
        logger.info(f"  - No change: {no_change_count}/{len(results)} files")

        # Decisão final
        logger.info(f"\n--- DECISION ---")
        if avg_improvement_pct >= 3.0:
            logger.info(f"✅ SIGNIFICANT IMPROVEMENT: {avg_improvement_pct:+.1f}%")
            logger.info(f"   → initial_prompt is EFFECTIVE, keep it enabled!")
            decision = "KEEP"
        elif avg_improvement_pct >= 1.0:
            logger.info(f"⚠️ MARGINAL IMPROVEMENT: {avg_improvement_pct:+.1f}%")
            logger.info(f"   → initial_prompt shows benefit, recommend keeping")
            decision = "KEEP"
        elif avg_improvement_pct >= -1.0:
            logger.info(f"⚠️ NO SIGNIFICANT CHANGE: {avg_improvement_pct:+.1f}%")
            logger.info(f"   → initial_prompt has minimal impact")
            decision = "OPTIONAL"
        else:
            logger.info(f"❌ DEGRADATION: {avg_improvement_pct:+.1f}%")
            logger.info(f"   → initial_prompt is HARMFUL, disable it!")
            decision = "REMOVE"

        # Salvar relatório completo
        output_dir = Path("benchmarks")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"initial_prompt_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": "whisper-medium (int8, beam_size=5, best_of=5)",
            "prompt_tokens": 221,
            "baseline_avg": BASELINE_AVG,
            "new_avg": avg_new,
            "improvement": avg_improvement,
            "improvement_percentage": avg_improvement_pct,
            "decision": decision,
            "files": results,
            "baseline_data": BASELINE_RESULTS
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"REPORT SAVED: {output_file}")
        logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())
