"""
Script de Captura de Accuracy Baseline
Objetivo: Documentar a transcrição do medium model antes de A/B test com small

Dia 1 - Tarde: Preparação para A/B Test
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

async def capture_baseline(audio_path: str, model_name: str = "medium"):
    """
    Captura a baseline de accuracy de um modelo específico.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"CAPTURING ACCURACY BASELINE")
    logger.info(f"Model: {model_name}")
    logger.info(f"Audio: {audio_path}")
    logger.info(f"{'='*80}\n")

    try:
        # Obter duração
        import librosa
        duration = librosa.get_duration(path=audio_path)
        logger.info(f"Audio duration: {duration:.2f}s")

        # Inicializar serviço de transcrição
        logger.info(f"\nInitializing {model_name} model...")
        trans_service = TranscriptionService(model_name=model_name, device="cpu")
        await trans_service.initialize()

        # Transcrever com word timestamps
        logger.info("Transcribing with word timestamps...")
        result = await trans_service.transcribe_with_enhancements(
            audio_path,
            word_timestamps=True
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"TRANSCRIPTION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Processing time: {result.processing_time:.2f}s")
        logger.info(f"Confidence: {result.confidence:.2%}")
        logger.info(f"Word count: {result.word_count}")
        logger.info(f"Segments: {len(result.segments)}")

        # Extrair texto completo
        full_text = result.text
        logger.info(f"\n{'='*60}")
        logger.info(f"FULL TRANSCRIPTION TEXT:")
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

        logger.info(f"Total words extracted: {len(all_words)}")

        # Verificar palavras-chave PT-BR
        logger.info(f"\n{'='*60}")
        logger.info(f"PT-BR KEYWORD DETECTION")
        logger.info(f"{'='*60}")

        text_lower = full_text.lower()
        found_keywords = []
        missing_keywords = []

        for keyword in PT_BR_KEYWORDS:
            if keyword in text_lower:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

        logger.info(f"\nKeywords found: {len(found_keywords)}/{len(PT_BR_KEYWORDS)}")
        if found_keywords:
            logger.info(f"Found: {', '.join(found_keywords[:10])}{'...' if len(found_keywords) > 10 else ''}")
        if missing_keywords:
            logger.info(f"Missing: {', '.join(missing_keywords[:10])}{'...' if len(missing_keywords) > 10 else ''}")

        # Análise de palavras com acentos/til
        accented_words = [w for w in all_words if any(c in w['word'] for c in 'áàâãéêíóôõúüç')]
        logger.info(f"\nWords with accents/til: {len(accented_words)}")
        if accented_words:
            logger.info("Sample accented words:")
            for w in accented_words[:10]:
                logger.info(f"  - {w['word']} (confidence: {w['probability']:.2%})")

        # Análise de confiança
        avg_probability = sum(w['probability'] for w in all_words) / len(all_words) if all_words else 0
        low_confidence_words = [w for w in all_words if w['probability'] < 0.7]

        logger.info(f"\nAverage word probability: {avg_probability:.2%}")
        logger.info(f"Low confidence words (<70%): {len(low_confidence_words)}")
        if low_confidence_words:
            logger.info("Sample low confidence words:")
            for w in low_confidence_words[:5]:
                logger.info(f"  - {w['word']} ({w['probability']:.2%})")

        # Montar baseline report
        baseline_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "audio_file": Path(audio_path).name,
            "audio_duration": duration,
            "processing_time": result.processing_time,
            "transcription": {
                "full_text": full_text,
                "word_count": result.word_count,
                "segment_count": len(result.segments),
                "confidence": result.confidence,
                "avg_word_probability": avg_probability
            },
            "words": all_words,
            "pt_br_analysis": {
                "keywords_found": len(found_keywords),
                "keywords_total": len(PT_BR_KEYWORDS),
                "keywords_percentage": len(found_keywords) / len(PT_BR_KEYWORDS) * 100,
                "found_list": found_keywords,
                "missing_list": missing_keywords,
                "accented_words_count": len(accented_words),
                "low_confidence_words_count": len(low_confidence_words)
            }
        }

        # Salvar baseline
        output_dir = Path("benchmarks")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"accuracy_baseline_{model_name}_{Path(audio_path).stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"BASELINE SAVED")
        logger.info(f"{'='*60}")
        logger.info(f"Output: {output_file}")
        logger.info(f"\nSummary:")
        logger.info(f"  - Full text: {result.word_count} words")
        logger.info(f"  - PT-BR keywords: {len(found_keywords)}/{len(PT_BR_KEYWORDS)} ({len(found_keywords)/len(PT_BR_KEYWORDS)*100:.1f}%)")
        logger.info(f"  - Accented words: {len(accented_words)}")
        logger.info(f"  - Avg confidence: {avg_probability:.2%}")
        logger.info(f"  - Low confidence: {len(low_confidence_words)} words")

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ BASELINE CAPTURE COMPLETE")
        logger.info(f"{'='*60}\n")

        return baseline_data

    except Exception as e:
        logger.error(f"Error capturing baseline: {e}", exc_info=True)
        return None

async def main():
    """
    Captura baseline do q.speakers.wav (arquivo com 4 speakers e boa variedade).
    """
    audio_path = "data/inputs/q.speakers.wav"

    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        return

    # Capturar baseline do medium model
    baseline = await capture_baseline(audio_path, model_name="medium")

    if baseline:
        logger.info("\n" + "="*80)
        logger.info("NEXT STEPS:")
        logger.info("="*80)
        logger.info("1. Review the baseline transcription above")
        logger.info("2. Note any specific words or phrases to verify")
        logger.info("3. Run A/B test with small model")
        logger.info("4. Compare transcriptions for accuracy degradation")
        logger.info("5. Decision: GO/NO-GO based on <10% degradation threshold")
        logger.info("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
