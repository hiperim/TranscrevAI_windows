# tests/test_fase2_preprocessing.py
"""
Teste da FASE 2: Audio Preprocessing

ULTRA ACCURACY (baseline) = 68.21% accuracy, 18.1s

FASE 2 adiciona:
  1. Normalização de volume (pydub)
  2. Redução de ruído (noisereduce)
  3. VAD adaptativo baseado em características do áudio

Ganho esperado: +1.5-5% accuracy
Tempo esperado: +1.5-4s (19-22s)
"""

import asyncio
import time
import sys
import re
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from tests.metrics import calculate_wer, calculate_cer

# Ground truth
GROUND_TRUTH = {
    "d.speakers.wav": {"text_file": "d_speakers.txt", "speakers": 2},
    "q.speakers.wav": {"text_file": "q_speakers.txt", "speakers": 4},
    "t.speakers.wav": {"text_file": "t_speakers.txt", "speakers": 3},
    "t2.speakers.wav": {"text_file": "t2_speakers.txt", "speakers": 3}
}

AUDIO_DIR = Path(__file__).parent.parent / "data" / "recordings"
TRUTH_DIR = Path(__file__).parent / "ground_truth"
REPORT_DIR = Path(__file__).parent.parent / ".claude" / "test_reports"

def normalize_text_for_wer(text: str) -> str:
    """Normaliza texto para comparação justa."""
    text = text.lower()
    text = re.sub(r'[^\w\sàáâãèéêìíîòóôõùúûç]', '', text)
    text = ' '.join(text.split())
    return text.strip()

async def test_fase2_preprocessing():
    """Testa ULTRA ACCURACY + FASE 2 preprocessing."""

    print("\n" + "="*70)
    print("TESTE FASE 2: AUDIO PREPROCESSING")
    print("Model: medium")
    print("Compute: int8")
    print("="*70)
    print("\nCONFIGURAÇÃO ULTRA ACCURACY + FASE 2:")
    print("  ✅ VAD: threshold 0.4, silence 1000ms (base)")
    print("  ✅ Beam search: 10/10")
    print("  ✅ PT-BR corrections: 12 regras seguras")
    print("  🆕 Normalização de volume (pydub)")
    print("  🆕 Redução de ruído (noisereduce)")
    print("  🆕 VAD adaptativo (analisa audio e ajusta)")
    print("")
    print("Baseline ULTRA ACCURACY: 68.21% accuracy, 18.1s")
    print("Ganho esperado FASE 2: +1.5-5% accuracy, +1.5-4s tempo")
    print("="*70)

    # Inicializar COM preprocessing habilitado
    service = TranscriptionService(
        model_name="medium",
        compute_type="int8",
        enable_preprocessing=True  # ATIVAR PREPROCESSAMENTO
    )
    await service.initialize()

    results = []

    for audio_file, truth_data in GROUND_TRUTH.items():
        audio_path = AUDIO_DIR / audio_file
        if not audio_path.exists():
            continue

        print(f"\n{audio_file}...", end=" ", flush=True)

        # Ground truth
        truth_path = TRUTH_DIR / truth_data["text_file"]
        expected_raw = truth_path.read_text(encoding="utf-8").strip()

        # Transcrever com ULTRA ACCURACY + FASE 2 (usa preprocessing)
        start = time.time()
        result = await service.transcribe_with_enhancements(str(audio_path))
        processing_time = time.time() - start

        actual_raw = result.text

        # Métricas
        expected_norm = normalize_text_for_wer(expected_raw)
        actual_norm = normalize_text_for_wer(actual_raw)

        wer = calculate_wer(expected_norm, actual_norm)
        cer = calculate_cer(expected_norm, actual_norm)
        accuracy = max(0, 1 - wer) * 100

        print(f"WER: {wer:.2%} | Acc: {accuracy:.1f}% | Time: {processing_time:.1f}s")

        results.append({
            "file": audio_file,
            "wer": wer,
            "cer": cer,
            "accuracy": accuracy,
            "processing_time": processing_time,
            "expected_raw": expected_raw,
            "actual_raw": actual_raw
        })

    await service.unload_model()

    # Resumo
    avg_wer = sum(r['wer'] for r in results) / len(results)
    avg_cer = sum(r['cer'] for r in results) / len(results)
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
    avg_time = sum(r['processing_time'] for r in results) / len(results)

    print("\n" + "="*70)
    print("RESULTADOS ULTRA ACCURACY + FASE 2 PREPROCESSING")
    print("="*70)
    print(f"Avg. Accuracy:  {avg_accuracy:.2f}%")
    print(f"Avg. WER:       {avg_wer:.2%}")
    print(f"Avg. CER:       {avg_cer:.2%}")
    print(f"Avg. Time:      {avg_time:.1f}s")
    print("="*70)

    print("\n" + "="*70)
    print("COMPARAÇÃO: ULTRA ACCURACY vs ULTRA + FASE 2")
    print("="*70)
    print("ULTRA ACCURACY (baseline):")
    print("  - Accuracy: 68.21%")
    print("  - WER: 31.79%")
    print("  - Time: 18.1s")
    print("")
    print(f"ULTRA + FASE 2 PREPROCESSING (atual):")
    print(f"  - Accuracy: {avg_accuracy:.2f}%")
    print(f"  - WER: {avg_wer:.2%}")
    print(f"  - Time: {avg_time:.1f}s")
    print("")

    diff_accuracy = avg_accuracy - 68.21
    diff_time = avg_time - 18.1
    print(f"GANHO ACCURACY: {diff_accuracy:+.2f} pontos percentuais")
    print(f"CUSTO TEMPO: {diff_time:+.1f}s ({(diff_time/18.1)*100:+.1f}%)")

    if diff_accuracy >= 3:
        print(f"✅ FASE 2 EXCELENTE! Ganho de {diff_accuracy:.2f}%")
    elif diff_accuracy >= 1.5:
        print(f"✅ FASE 2 BOM! Ganho de {diff_accuracy:.2f}%")
    elif diff_accuracy > 0:
        print(f"⚠️  FASE 2 OK. Pequeno ganho de {diff_accuracy:.2f}%")
    else:
        print(f"❌ FASE 2 FALHOU. Perda de {abs(diff_accuracy):.2f}%")

    print("="*70)

    # Salvar relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORT_DIR / f"fase2_preprocessing_{timestamp}.json"
    json_path.write_text(json.dumps({
        "timestamp": timestamp,
        "config": {
            "model": "medium",
            "compute_type": "int8",
            "base": "ULTRA ACCURACY",
            "preprocessing": [
                "Volume normalization (pydub)",
                "Noise reduction (noisereduce)",
                "Adaptive VAD (audio analysis)"
            ]
        },
        "avg_accuracy": avg_accuracy,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_time": avg_time,
        "comparison": {
            "baseline_accuracy": 68.21,
            "baseline_time": 18.1,
            "gain_accuracy": diff_accuracy,
            "cost_time": diff_time
        },
        "results": results
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"\n✅ Dados salvos: {json_path}")

    return {
        "avg_accuracy": avg_accuracy,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_time": avg_time,
        "gain_vs_ultra": diff_accuracy
    }

if __name__ == "__main__":
    asyncio.run(test_fase2_preprocessing())
