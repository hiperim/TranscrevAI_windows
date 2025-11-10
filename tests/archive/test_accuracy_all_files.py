"""
Teste de Accuracy: Todos os 4 Arquivos de Benchmark
Objetivo: Validar accuracy real do medium model (beam_size=5, best_of=5) atual
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

# Palavras-chave PT-BR importantes para verificar
PT_BR_KEYWORDS = [
    # Palavras com acentos
    "não", "está", "você", "já", "é", "até", "também",
    "só", "lá", "aí", "daí", "então", "porém", "além",

    # Palavras com til
    "ação", "ações", "informação", "informações", "opção", "opções",
    "irmão", "mãe", "mão", "pão", "cidadão",

    # Contrações comuns
    "pra", "pro", "ta", "to", "ce", "né",

    # Palavras técnicas comuns
    "português", "brasileiro", "conversa", "falar", "pessoa", "pessoas"
]

from typing import Optional

# ... (rest of imports)

async def test_accuracy_single_file(audio_path: str) -> Optional[dict]:
    """
    Testa accuracy de um arquivo específico.
    """
    filename = Path(audio_path).name
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING ACCURACY: {filename}")
    logger.info(f"{'='*80}")

    try:
        # Obter duração
        import librosa
        duration = librosa.get_duration(path=audio_path)
        logger.info(f"Audio duration: {duration:.2f}s")

        # Inicializar serviço de transcrição
        logger.info(f"Initializing medium model (beam_size=5, best_of=5)...")
        trans_service = TranscriptionService(model_name="medium", device="cpu")
        await trans_service.initialize()

        # Transcrever com word timestamps
        logger.info("Transcribing with word timestamps...")
        result = await trans_service.transcribe_with_enhancements(
            audio_path,
            word_timestamps=True
        )

        # Extrair texto completo
        full_text = result.text
        logger.info(f"\n{'='*60}")
        logger.info(f"FULL TRANSCRIPTION:")
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

        logger.info(f"Metrics:")
        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
        logger.info(f"  - Processing ratio: {result.processing_time/duration:.2f}x")
        logger.info(f"  - Word count: {len(all_words)}")
        logger.info(f"  - Segments: {len(result.segments)}")
        logger.info(f"  - Confidence (result): {result.confidence:.2%}")
        logger.info(f"  - Avg word probability: {avg_probability:.2%}")
        logger.info(f"  - Low confidence words (<70%): {len(low_confidence_words)}")
        logger.info(f"  - Accented PT-BR words: {len(accented_words)}")

        # Verificar palavras-chave PT-BR
        text_lower = full_text.lower()
        found_keywords = []
        missing_keywords = []

        for keyword in PT_BR_KEYWORDS:
            if keyword in text_lower:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

        logger.info(f"\nPT-BR Keyword Detection:")
        logger.info(f"  - Found: {len(found_keywords)}/{len(PT_BR_KEYWORDS)} ({len(found_keywords)/len(PT_BR_KEYWORDS)*100:.1f}%)")
        if found_keywords[:5]:
            logger.info(f"  - Sample found: {', '.join(found_keywords[:5])}")
        if missing_keywords[:5]:
            logger.info(f"  - Sample missing: {', '.join(missing_keywords[:5])}")

        # Listar palavras de baixa confiança
        if low_confidence_words:
            logger.info(f"\nLow Confidence Words (<70%):")
            for w in low_confidence_words[:10]:
                logger.info(f"  - \"{w['word']}\" ({w['probability']:.2%})")

        # Listar palavras acentuadas
        if accented_words:
            logger.info(f"\nAccented PT-BR Words:")
            for w in accented_words[:10]:
                logger.info(f"  - \"{w['word']}\" ({w['probability']:.2%})")

        return {
            "filename": filename,
            "duration": duration,
            "processing_time": result.processing_time,
            "processing_ratio": result.processing_time / duration,
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
                "accented_words": accented_words[:10],
                "pt_br_keywords_found": len(found_keywords),
                "pt_br_keywords_total": len(PT_BR_KEYWORDS),
                "pt_br_keywords_percentage": len(found_keywords) / len(PT_BR_KEYWORDS) * 100,
                "found_keywords": found_keywords,
                "missing_keywords": missing_keywords
            }
        }

    except Exception as e:
        logger.error(f"Error testing {filename}: {e}", exc_info=True)
        return None

async def main():
    """
    Testa accuracy em todos os 4 arquivos de benchmark.
    """
    logger.info("="*80)
    logger.info("ACCURACY TEST: All 4 Benchmark Files")
    logger.info("Model: whisper-medium (int8, beam_size=5, best_of=5)")
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

        result = await test_accuracy_single_file(audio_path)
        if result:
            results.append(result)

        # Pausa entre arquivos
        logger.info("\n" + "="*80 + "\n")
        await asyncio.sleep(2)

    # ===== SUMMARY ANALYSIS =====
    if results:
        logger.info("\n" + "="*80)
        logger.info("SUMMARY: Accuracy Across All Files")
        logger.info("="*80 + "\n")

        # Tabela formatada
        print(f"{'File':<20} {'Words':>8} {'Segments':>10} {'Confidence':>12} {'Avg Prob':>10} {'Low Conf':>10} {'Accented':>10}")
        print("-" * 100)

        for r in results:
            trans = r['transcription']
            analysis = r['analysis']
            print(f"{r['filename']:<20} {trans['word_count']:>8} {trans['segment_count']:>10} {trans['confidence']:>11.2%} {trans['avg_word_probability']:>9.2%} {analysis['low_confidence_count']:>10} {analysis['accented_words_count']:>10}")

        # Estatísticas gerais
        avg_confidence = sum(r['transcription']['confidence'] for r in results) / len(results)
        avg_word_prob = sum(r['transcription']['avg_word_probability'] for r in results) / len(results)
        total_words = sum(r['transcription']['word_count'] for r in results)
        total_low_conf = sum(r['analysis']['low_confidence_count'] for r in results)
        avg_pt_br_percentage = sum(r['analysis']['pt_br_keywords_percentage'] for r in results) / len(results)

        print("-" * 100)
        print(f"{'AVERAGE':<20} {total_words:>8} {'':<10} {avg_confidence:>11.2%} {avg_word_prob:>9.2%} {total_low_conf:>10} {'':<10}")

        logger.info(f"\n--- OVERALL ACCURACY ANALYSIS ---")
        logger.info(f"Average confidence (result): {avg_confidence:.2%}")
        logger.info(f"Average word probability: {avg_word_prob:.2%}")
        logger.info(f"Total words transcribed: {total_words}")
        logger.info(f"Total low confidence words: {total_low_conf} ({total_low_conf/total_words*100:.1f}% of all words)")
        logger.info(f"Average PT-BR keyword detection: {avg_pt_br_percentage:.1f}%")

        # Análise de padrões
        logger.info(f"\n--- PATTERN ANALYSIS ---")

        # Ordenar por confidence
        sorted_by_conf = sorted(results, key=lambda x: x['transcription']['avg_word_probability'], reverse=True)

        logger.info("\nFiles by accuracy (best to worst):")
        for i, r in enumerate(sorted_by_conf, 1):
            trans = r['transcription']
            status = "✅ GOOD" if trans['avg_word_probability'] >= 0.85 else "⚠️ FAIR" if trans['avg_word_probability'] >= 0.75 else "❌ POOR"
            logger.info(f"  {i}. {r['filename']}: {trans['avg_word_probability']:.2%} {status}")

        # Threshold analysis
        good_files = sum(1 for r in results if r['transcription']['avg_word_probability'] >= 0.85)
        fair_files = sum(1 for r in results if 0.75 <= r['transcription']['avg_word_probability'] < 0.85)
        poor_files = sum(1 for r in results if r['transcription']['avg_word_probability'] < 0.75)

        logger.info(f"\nAccuracy distribution:")
        logger.info(f"  - Good (≥85%): {good_files}/{len(results)} files")
        logger.info(f"  - Fair (75-85%): {fair_files}/{len(results)} files")
        logger.info(f"  - Poor (<75%): {poor_files}/{len(results)} files")

        # Decisão final
        logger.info(f"\n--- FINAL ASSESSMENT ---")
        if avg_word_prob >= 0.85:
            logger.info(f"✅ ACCURACY OK: {avg_word_prob:.2%} meets 85% threshold")
        elif avg_word_prob >= 0.75:
            logger.info(f"⚠️ ACCURACY MARGINAL: {avg_word_prob:.2%} below 85% target but acceptable")
        else:
            logger.info(f"❌ ACCURACY POOR: {avg_word_prob:.2%} significantly below 85% target")
            logger.info(f"   → This confirms int8 medium model has fundamental accuracy limitations")

        # Salvar relatório completo
        output_dir = Path("benchmarks")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"accuracy_test_all_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": "whisper-medium (int8, beam_size=5, best_of=5)",
            "summary": {
                "avg_confidence": avg_confidence,
                "avg_word_probability": avg_word_prob,
                "total_words": total_words,
                "total_low_confidence": total_low_conf,
                "avg_pt_br_percentage": avg_pt_br_percentage,
                "good_files": good_files,
                "fair_files": fair_files,
                "poor_files": poor_files
            },
            "files": results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"REPORT SAVED: {output_file}")
        logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())
