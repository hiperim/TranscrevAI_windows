"""
A/B Test: whisper-medium vs whisper-small
Objetivo: Validar se small mantém accuracy com speedup

Dia 1 - Tarde: A/B Test Decisivo
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_model(audio_path: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Testa um modelo e retorna métricas.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING MODEL: {model_name}")
    logger.info(f"{'='*80}\n")

    try:
        import librosa
        import time

        duration = librosa.get_duration(path=audio_path)

        # Inicializar
        trans_service = TranscriptionService(model_name=model_name, device="cpu")
        init_start = time.time()
        await trans_service.initialize()
        init_time = time.time() - init_start

        logger.info(f"Model initialization: {init_time:.2f}s")

        # Transcrever
        start = time.time()
        result = await trans_service.transcribe_with_enhancements(
            audio_path,
            word_timestamps=True
        )
        proc_time = time.time() - start

        # Extrair palavras
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
        avg_prob = sum(w['probability'] for w in all_words) / len(all_words) if all_words else 0
        accented_words = [w for w in all_words if any(c in w['word'] for c in 'áàâãéêíóôõúüç')]
        low_conf_words = [w for w in all_words if w['probability'] < 0.7]

        logger.info(f"Transcription: {proc_time:.2f}s (ratio: {proc_time/duration:.2f}x)")
        logger.info(f"Words: {len(all_words)}")
        logger.info(f"Avg confidence: {avg_prob:.2%}")
        logger.info(f"Accented words: {len(accented_words)}")
        logger.info(f"Low confidence: {len(low_conf_words)}")

        return {
            "model": model_name,
            "audio_duration": duration,
            "init_time": init_time,
            "processing_time": proc_time,
            "processing_ratio": proc_time / duration,
            "transcription": {
                "full_text": result.text,
                "word_count": len(all_words),
                "segment_count": len(result.segments),
                "confidence": result.confidence,
                "avg_word_probability": avg_prob
            },
            "words": all_words,
            "analysis": {
                "accented_words_count": len(accented_words),
                "low_confidence_words_count": len(low_conf_words),
                "accented_words": accented_words[:10],
                "low_conf_words": low_conf_words[:10]
            }
        }

    except Exception as e:
        logger.error(f"Error testing {model_name}: {e}", exc_info=True)
        return None

def compare_models(medium_data: Dict, small_data: Dict) -> Dict[str, Any]:
    """
    Compara os dois modelos e gera relatório.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPARISON REPORT: MEDIUM vs SMALL")
    logger.info(f"{'='*80}\n")

    # Performance comparison
    logger.info("--- PERFORMANCE ---")
    medium_ratio = medium_data['processing_ratio']
    small_ratio = small_data['processing_ratio']
    speedup = medium_ratio / small_ratio
    speedup_pct = (1 - small_ratio / medium_ratio) * 100

    logger.info(f"Medium: {medium_ratio:.2f}x ({medium_data['processing_time']:.2f}s)")
    logger.info(f"Small:  {small_ratio:.2f}x ({small_data['processing_time']:.2f}s)")
    logger.info(f"Speedup: {speedup:.2f}x faster ({speedup_pct:+.1f}%)")

    performance_verdict = "✅ PASS" if speedup >= 1.3 else "⚠️ MARGINAL" if speedup >= 1.2 else "❌ FAIL"
    logger.info(f"Performance verdict: {performance_verdict}")

    # Accuracy comparison
    logger.info(f"\n--- ACCURACY ---")

    # Text comparison
    medium_text = medium_data['transcription']['full_text']
    small_text = small_data['transcription']['full_text']

    logger.info(f"\nMedium transcription ({medium_data['transcription']['word_count']} words):")
    logger.info(f'"{medium_text}"\n')

    logger.info(f"Small transcription ({small_data['transcription']['word_count']} words):")
    logger.info(f'"{small_text}"\n')

    # Word-level comparison
    medium_words = set(w['word'].lower() for w in medium_data['words'])
    small_words = set(w['word'].lower() for w in small_data['words'])

    common_words = medium_words & small_words
    medium_only = medium_words - small_words
    small_only = small_words - medium_words

    word_match_rate = len(common_words) / max(len(medium_words), len(small_words)) * 100

    logger.info(f"Word-level analysis:")
    logger.info(f"  Common words: {len(common_words)}")
    logger.info(f"  Medium-only: {len(medium_only)}")
    logger.info(f"  Small-only: {len(small_only)}")
    logger.info(f"  Match rate: {word_match_rate:.1f}%")

    if medium_only:
        logger.info(f"  Words in medium but not small: {list(medium_only)[:10]}")
    if small_only:
        logger.info(f"  Words in small but not medium: {list(small_only)[:10]}")

    # Confidence comparison
    medium_conf = medium_data['transcription']['avg_word_probability']
    small_conf = small_data['transcription']['avg_word_probability']
    conf_diff = small_conf - medium_conf
    conf_diff_pct = (small_conf / medium_conf - 1) * 100

    logger.info(f"\nConfidence comparison:")
    logger.info(f"  Medium: {medium_conf:.2%}")
    logger.info(f"  Small:  {small_conf:.2%}")
    logger.info(f"  Difference: {conf_diff_pct:+.1f}%")

    # Accented words comparison
    medium_accented = medium_data['analysis']['accented_words_count']
    small_accented = small_data['analysis']['accented_words_count']

    logger.info(f"\nAccented words (PT-BR):")
    logger.info(f"  Medium: {medium_accented}")
    logger.info(f"  Small:  {small_accented}")
    logger.info(f"  Difference: {small_accented - medium_accented:+d}")

    # Calculate accuracy degradation
    text_similarity = word_match_rate / 100
    conf_degradation = abs(conf_diff_pct) / 100

    # Weighted accuracy score (text similarity 70%, confidence 30%)
    accuracy_score = (text_similarity * 0.7) + ((1 - conf_degradation) * 0.3)
    degradation_pct = (1 - accuracy_score) * 100

    logger.info(f"\n--- ACCURACY VERDICT ---")
    logger.info(f"Text similarity: {text_similarity:.1%}")
    logger.info(f"Overall accuracy score: {accuracy_score:.1%}")
    logger.info(f"Estimated degradation: {degradation_pct:.1f}%")

    accuracy_verdict = "✅ PASS" if degradation_pct < 10 else "⚠️ MARGINAL" if degradation_pct < 15 else "❌ FAIL"
    logger.info(f"Accuracy verdict: {accuracy_verdict} (threshold: <10% degradation)")

    # Final decision
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL DECISION")
    logger.info(f"{'='*80}")

    go_decision = performance_verdict == "✅ PASS" and accuracy_verdict in ["✅ PASS", "⚠️ MARGINAL"]

    if go_decision:
        logger.info(f"✅ GO: Prosseguir com whisper-small")
        logger.info(f"   Speedup: {speedup:.2f}x ({speedup_pct:+.1f}%)")
        logger.info(f"   Accuracy: {degradation_pct:.1f}% degradation (acceptable)")
        decision = "GO"
        reason = f"Performance improvement ({speedup:.2f}x) with acceptable accuracy ({degradation_pct:.1f}% degradation)"
    else:
        logger.info(f"❌ NO-GO: Manter whisper-medium")
        if performance_verdict != "✅ PASS":
            reason = f"Insufficient speedup ({speedup:.2f}x, expected ≥1.3x)"
        else:
            reason = f"Unacceptable accuracy degradation ({degradation_pct:.1f}%, threshold <10%)"
        logger.info(f"   Reason: {reason}")
        decision = "NO-GO"

    logger.info(f"{'='*80}\n")

    return {
        "timestamp": datetime.now().isoformat(),
        "decision": decision,
        "reason": reason,
        "performance": {
            "medium_ratio": medium_ratio,
            "small_ratio": small_ratio,
            "speedup": speedup,
            "speedup_percentage": speedup_pct,
            "verdict": performance_verdict
        },
        "accuracy": {
            "word_match_rate": word_match_rate,
            "confidence_diff_pct": conf_diff_pct,
            "overall_score": accuracy_score * 100,
            "degradation_pct": degradation_pct,
            "verdict": accuracy_verdict
        },
        "medium_data": medium_data,
        "small_data": small_data
    }

async def main():
    """
    Executa A/B test completo.
    """
    audio_path = "data/inputs/q.speakers.wav"

    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        return

    logger.info("="*80)
    logger.info("A/B TEST: whisper-medium vs whisper-small")
    logger.info("="*80)
    logger.info(f"Audio: {audio_path}")
    logger.info(f"Duration: ~14.5s")
    logger.info(f"Expected time: ~30-40s total")
    logger.info("="*80 + "\n")

    # Test both models
    medium_data = await test_model(audio_path, "medium")

    if not medium_data:
        logger.error("Medium model test failed, aborting")
        return

    small_data = await test_model(audio_path, "small")

    if not small_data:
        logger.error("Small model test failed, aborting")
        return

    # Compare
    comparison = compare_models(medium_data, small_data)

    # Save report
    output_dir = Path("benchmarks")
    output_file = output_dir / f"ab_test_medium_vs_small_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    logger.info(f"Report saved: {output_file}")

    logger.info(f"\n{'='*80}")
    logger.info("NEXT STEPS")
    logger.info("="*80)

    if comparison['decision'] == "GO":
        logger.info("✅ Proceed with whisper-small implementation:")
        logger.info("   1. Create branch: feature/whisper-small-optimization")
        logger.info("   2. Update src/transcription.py (model_name='small')")
        logger.info("   3. Apply Level 1+2 PT-BR optimizations")
        logger.info("   4. Run full benchmark suite")
        logger.info("   5. Apply Level 4 PyAnnote enhancements")
    else:
        logger.info("❌ Keep whisper-medium, alternative optimizations:")
        logger.info("   1. Focus on VAD tuning for speed")
        logger.info("   2. Apply Level 4 PyAnnote for accuracy")
        logger.info("   3. Consider compute optimizations (threads, batch)")

    logger.info("="*80)

if __name__ == "__main__":
    asyncio.run(main())
