# tests/test_baseline_optimized.py
"""
Teste do baseline OpenAI Whisper medium int8 OTIMIZADO

Compara com resultados anteriores:
- Baseline ANTES das otimizações
- Baseline DEPOIS das 3 otimizações:
  1. Correções PT-BR reduzidas (295 → 12 regras)
  2. VAD otimizado (threshold 0.45, silence 1200ms)
  3. Beam search aumentado (5/5 → 7/7)
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

async def test_baseline_optimized():
    """Testa baseline int8 otimizado."""

    print("\n" + "="*70)
    print("TESTE BASELINE OTIMIZADO")
    print("Model: medium")
    print("Compute: int8")
    print("="*70)
    print("\nOTIMIZAÇÕES IMPLEMENTADAS:")
    print("  1. ✅ Correções PT-BR: 295 → 12 regras")
    print("  2. ✅ VAD: threshold 0.45, silence 1200ms (vs 2000ms)")
    print("  3. ✅ Beam search: 7/7 (vs 5/5)")
    print("="*70)

    # Inicializar
    service = TranscriptionService(model_name="medium", compute_type="int8")
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

        # Transcrever
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
    print("RESULTADOS OTIMIZADOS")
    print("="*70)
    print(f"Avg. Accuracy:  {avg_accuracy:.2f}%")
    print(f"Avg. WER:       {avg_wer:.2%}")
    print(f"Avg. CER:       {avg_cer:.2%}")
    print(f"Avg. Time:      {avg_time:.1f}s")
    print("="*70)

    print("\n" + "="*70)
    print("COMPARAÇÃO COM BASELINE ANTERIOR")
    print("="*70)
    print("Baseline ANTES (float32, sem otimizações):")
    print("  - Accuracy: 64.40%")
    print("  - WER: 35.60%")
    print("  - Time: 31.1s")
    print("")
    print(f"Baseline AGORA (int8, otimizado):")
    print(f"  - Accuracy: {avg_accuracy:.2f}%")
    print(f"  - WER: {avg_wer:.2%}")
    print(f"  - Time: {avg_time:.1f}s")
    print("")

    diff_accuracy = avg_accuracy - 64.40
    print(f"DIFERENÇA: {diff_accuracy:+.2f} pontos percentuais")

    if diff_accuracy > 0:
        print(f"✅ MELHORIA de {diff_accuracy:.2f}%!")
    elif diff_accuracy < -2:
        print(f"❌ REGRESSÃO de {abs(diff_accuracy):.2f}%")
    else:
        print("⚠️  Resultados similares")

    print("="*70)

    # Salvar relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORT_DIR / f"baseline_optimized_{timestamp}.json"
    json_path.write_text(json.dumps({
        "timestamp": timestamp,
        "config": {
            "model": "medium",
            "compute_type": "int8",
            "optimizations": [
                "PT-BR corrections: 12 rules (was 295)",
                "VAD: threshold 0.45, silence 1200ms",
                "Beam search: 7/7 (was 5/5)"
            ]
        },
        "avg_accuracy": avg_accuracy,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_time": avg_time,
        "results": results
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"\n✅ Dados salvos: {json_path}")

if __name__ == "__main__":
    asyncio.run(test_baseline_optimized())
