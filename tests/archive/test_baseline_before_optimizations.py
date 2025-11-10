# tests/test_baseline_before_optimizations.py
"""
Teste do baseline OpenAI Whisper medium int8 ANTES das otimizações

Parâmetros ORIGINAIS (antes das 3 otimizações):
  1. Correções PT-BR: 295 regras (vou desabilitar para baseline puro)
  2. VAD: threshold 0.5, min_speech_duration_ms 250, silence 2000ms
  3. Beam search: 5/5

Compara com baseline otimizado atual.
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

async def test_baseline_before_optimizations():
    """Testa baseline int8 com parâmetros ORIGINAIS (antes das otimizações)."""

    print("\n" + "="*70)
    print("TESTE BASELINE ANTES DAS OTIMIZAÇÕES")
    print("Model: medium")
    print("Compute: int8")
    print("="*70)
    print("\nPARÂMETROS ORIGINAIS (baseline antigo):")
    print("  1. ❌ Correções PT-BR: DESABILITADAS (baseline puro)")
    print("  2. ❌ VAD: threshold 0.5, silence 2000ms (original)")
    print("  3. ❌ Beam search: 5/5 (original)")
    print("="*70)

    # Inicializar
    service = TranscriptionService(model_name="medium", compute_type="int8")

    # HACK: Desabilitar correções PT-BR temporariamente
    original_corrections = service.ptbr_corrections.copy()
    service.ptbr_corrections = {}  # Desabilita todas as correções

    await service.initialize()

    results = []

    # Parâmetros ANTIGOS (antes das otimizações)
    old_vad_parameters = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 2000
    }
    old_beam_size = 5
    old_best_of = 5

    for audio_file, truth_data in GROUND_TRUTH.items():
        audio_path = AUDIO_DIR / audio_file
        if not audio_path.exists():
            continue

        print(f"\n{audio_file}...", end=" ", flush=True)

        # Ground truth
        truth_path = TRUTH_DIR / truth_data["text_file"]
        expected_raw = truth_path.read_text(encoding="utf-8").strip()

        # Transcrever com parâmetros ANTIGOS
        start = time.time()
        result = await service.transcribe_with_enhancements(
            str(audio_path),
            vad_parameters=old_vad_parameters,
            beam_size=old_beam_size,
            best_of=old_best_of
        )
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

    # Restaurar correções originais
    service.ptbr_corrections = original_corrections

    await service.unload_model()

    # Resumo
    avg_wer = sum(r['wer'] for r in results) / len(results)
    avg_cer = sum(r['cer'] for r in results) / len(results)
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
    avg_time = sum(r['processing_time'] for r in results) / len(results)

    print("\n" + "="*70)
    print("RESULTADOS BASELINE ANTIGO (SEM OTIMIZAÇÕES)")
    print("="*70)
    print(f"Avg. Accuracy:  {avg_accuracy:.2f}%")
    print(f"Avg. WER:       {avg_wer:.2%}")
    print(f"Avg. CER:       {avg_cer:.2%}")
    print(f"Avg. Time:      {avg_time:.1f}s")
    print("="*70)

    print("\n" + "="*70)
    print("COMPARAÇÃO: ANTES vs DEPOIS DAS OTIMIZAÇÕES")
    print("="*70)
    print("Baseline ANTES (int8, sem otimizações):")
    print(f"  - Accuracy: {avg_accuracy:.2f}%")
    print(f"  - WER: {avg_wer:.2%}")
    print(f"  - Time: {avg_time:.1f}s")
    print("")
    print("Baseline DEPOIS (int8, otimizado) - do teste anterior:")
    print("  - Accuracy: 64.90%")
    print("  - WER: 35.10%")
    print("  - Time: 16.5s")
    print("")

    diff_accuracy = 64.90 - avg_accuracy
    print(f"GANHO DAS OTIMIZAÇÕES: {diff_accuracy:+.2f} pontos percentuais")

    if diff_accuracy > 2:
        print(f"✅ OTIMIZAÇÕES MELHORARAM {diff_accuracy:.2f}%!")
    elif diff_accuracy > 0:
        print(f"✅ Pequena melhoria de {diff_accuracy:.2f}%")
    elif diff_accuracy < -2:
        print(f"❌ REGRESSÃO de {abs(diff_accuracy):.2f}%")
    else:
        print("⚠️  Resultados similares")

    print("="*70)

    # Salvar relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORT_DIR / f"baseline_before_optimizations_{timestamp}.json"
    json_path.write_text(json.dumps({
        "timestamp": timestamp,
        "config": {
            "model": "medium",
            "compute_type": "int8",
            "ptbr_corrections": "DISABLED (baseline puro)",
            "vad": {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 2000
            },
            "beam_search": {
                "beam_size": 5,
                "best_of": 5
            }
        },
        "avg_accuracy": avg_accuracy,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_time": avg_time,
        "results": results
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"\n✅ Dados salvos: {json_path}")

    return {
        "avg_accuracy": avg_accuracy,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_time": avg_time
    }

if __name__ == "__main__":
    asyncio.run(test_baseline_before_optimizations())
