# tests/test_abc_float32.py
"""
Teste ABC: Comparação de 3 modelos com faster-whisper float32

A. Baseline medium float32
B. jlondonobo-medium-ptbr-ct2-float32
C. pierreguillou-medium-ptbr-ct2-float32

Todos usando faster-whisper com float32 (comparação justa).
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

# ===========================================================
# CONFIGURAÇÃO DOS 3 MODELOS
# ===========================================================

MODELS = [
    {
        "name": "A_baseline",
        "model_name": "medium",
        "description": "Baseline OpenAI Whisper Medium"
    },
    {
        "name": "B_jlondonobo",
        "model_name": r"C:\TranscrevAI_windows\models\jlondonobo-medium-ptbr-ct2-float32",
        "description": "jlondonobo fine-tuned PT-BR"
    },
    {
        "name": "C_pierreguillou",
        "model_name": r"C:\TranscrevAI_windows\models\pierreguillou-medium-ptbr-ct2-float32",
        "description": "pierreguillou fine-tuned PT-BR"
    }
]

COMPUTE_TYPE = "float32"

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

# ===========================================================
# NORMALIZAÇÃO DE TEXTO
# ===========================================================

def normalize_text_for_wer(text: str) -> str:
    """Normaliza texto para comparação justa de WER."""
    text = text.lower()
    text = re.sub(r'[^\w\sàáâãèéêìíîòóôõùúûç]', '', text)
    text = ' '.join(text.split())
    return text.strip()

# ===========================================================
# TESTE
# ===========================================================

async def test_single_file(service, audio_path: Path, truth_data: dict, model_name: str) -> dict:
    """Testa um arquivo de áudio."""

    print(f"    {audio_path.name}...", end=" ", flush=True)

    # Ler ground truth
    truth_path = TRUTH_DIR / truth_data["text_file"]
    expected_raw = truth_path.read_text(encoding="utf-8").strip()

    # Transcrever
    start = time.time()
    result = await service.transcribe_with_enhancements(str(audio_path))
    processing_time = time.time() - start

    actual_raw = result.text

    # Normalizar e calcular métricas
    expected_norm = normalize_text_for_wer(expected_raw)
    actual_norm = normalize_text_for_wer(actual_raw)

    wer = calculate_wer(expected_norm, actual_norm)
    cer = calculate_cer(expected_norm, actual_norm)
    accuracy = max(0, 1 - wer) * 100

    print(f"WER: {wer:.2%} | Acc: {accuracy:.1f}% | Time: {processing_time:.1f}s")

    return {
        "file": audio_path.name,
        "model": model_name,
        "wer": wer,
        "cer": cer,
        "accuracy": accuracy,
        "processing_time": processing_time,
        "expected_raw": expected_raw,
        "actual_raw": actual_raw,
        "expected_norm": expected_norm,
        "actual_norm": actual_norm
    }

async def test_model(model_config: dict) -> list:
    """Testa um modelo em todos os arquivos."""

    print(f"\n{'='*70}")
    print(f"{model_config['name']}: {model_config['description']}")
    print(f"Model: {model_config['model_name']}")
    print(f"Compute: {COMPUTE_TYPE}")
    print(f"{'='*70}")

    # Inicializar serviço
    try:
        service = TranscriptionService(
            model_name=model_config['model_name'],
            compute_type=COMPUTE_TYPE
        )
        await service.initialize()
    except Exception as e:
        print(f"❌ ERRO ao inicializar: {e}")
        return []

    results = []

    for audio_file, truth_data in GROUND_TRUTH.items():
        audio_path = AUDIO_DIR / audio_file
        if not audio_path.exists():
            print(f"    ⚠️  Arquivo não encontrado: {audio_path}")
            continue

        try:
            result = await test_single_file(service, audio_path, truth_data, model_config['name'])
            results.append(result)
        except Exception as e:
            print(f"    ❌ ERRO: {e}")

    await service.unload_model()

    return results

# ===========================================================
# RELATÓRIO
# ===========================================================

def generate_report(all_results: dict) -> str:
    """Gera relatório comparativo."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calcular médias
    summaries = {}
    for model_name, results in all_results.items():
        if not results:
            continue

        summaries[model_name] = {
            "avg_wer": sum(r['wer'] for r in results) / len(results),
            "avg_cer": sum(r['cer'] for r in results) / len(results),
            "avg_accuracy": sum(r['accuracy'] for r in results) / len(results),
            "avg_time": sum(r['processing_time'] for r in results) / len(results),
            "num_files": len(results)
        }

    # Ordenar por accuracy
    sorted_models = sorted(summaries.items(), key=lambda x: x[1]['avg_accuracy'], reverse=True)

    report = f"""# Teste ABC - Comparação Float32 com faster-whisper

**Data:** {timestamp}
**Biblioteca:** faster-whisper (CTranslate2)
**Precisão:** {COMPUTE_TYPE}
**Arquivos testados:** {len(GROUND_TRUTH)}

---

## 📊 Resultados Resumidos

| Rank | Modelo | Avg. Accuracy | Avg. WER | Avg. CER | Avg. Time |
|------|--------|--------------|----------|----------|-----------|
"""

    for rank, (name, summary) in enumerate(sorted_models, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        report += f"| {emoji} | **{name}** | **{summary['avg_accuracy']:.2f}%** | {summary['avg_wer']:.2%} | {summary['avg_cer']:.2%} | {summary['avg_time']:.1f}s |\n"

    # Análise do vencedor
    if len(sorted_models) >= 2:
        winner = sorted_models[0]
        runner_up = sorted_models[1]
        diff = winner[1]['avg_accuracy'] - runner_up[1]['avg_accuracy']

        report += f"\n### 🏆 Vencedor: **{winner[0]}**\n\n"
        report += f"- **Accuracy:** {winner[1]['avg_accuracy']:.2f}%\n"
        report += f"- **WER:** {winner[1]['avg_wer']:.2%}\n"
        report += f"- **Vantagem sobre {runner_up[0]}:** {diff:+.2f} pontos percentuais\n\n"

        if diff >= 5.0:
            report += "✅ **MELHORIA SIGNIFICATIVA** - Diferença clara!\n\n"
        elif diff >= 2.0:
            report += "⚠️  **MELHORIA MODERADA** - Diferença perceptível.\n\n"
        elif diff >= 0.5:
            report += "⚠️  **MELHORIA MARGINAL** - Diferença pequena.\n\n"
        else:
            report += "⚠️  **ESTATISTICAMENTE SIMILAR** - Sem diferença significativa.\n\n"

    # Resultados por arquivo
    report += "\n## 📋 Resultados Detalhados por Arquivo\n\n"

    for audio_file in GROUND_TRUTH.keys():
        report += f"### {audio_file}\n\n"
        report += "| Modelo | WER | CER | Accuracy | Time |\n"
        report += "|--------|-----|-----|----------|------|\n"

        for model_name in all_results.keys():
            file_results = [r for r in all_results[model_name] if r['file'] == audio_file]
            if file_results:
                r = file_results[0]
                report += f"| {model_name} | {r['wer']:.2%} | {r['cer']:.2%} | {r['accuracy']:.1f}% | {r['processing_time']:.1f}s |\n"

        report += "\n"

    # Amostras de transcrição
    report += "\n## 📝 Amostras de Transcrição\n\n"

    for audio_file in GROUND_TRUTH.keys():
        report += f"### {audio_file}\n\n"

        # Expected
        truth_path = TRUTH_DIR / GROUND_TRUTH[audio_file]['text_file']
        expected = truth_path.read_text(encoding='utf-8').strip()
        report += f"**Ground Truth:**\n```\n{expected}\n```\n\n"

        # Cada modelo
        for model_name in all_results.keys():
            file_results = [r for r in all_results[model_name] if r['file'] == audio_file]
            if file_results:
                r = file_results[0]
                report += f"**{model_name}** (WER: {r['wer']:.2%}):\n```\n{r['actual_raw']}\n```\n\n"

        report += "---\n\n"

    # Recomendação final
    report += "\n## 🎯 Decisão Final\n\n"

    if sorted_models:
        winner = sorted_models[0]
        baseline_summary = summaries.get('A_baseline', None)

        if winner[0] == 'A_baseline':
            report += "### ✅ MANTER BASELINE\n\n"
            report += f"O modelo baseline alcançou a melhor accuracy ({winner[1]['avg_accuracy']:.2f}%).\n\n"
            report += "**Conclusão:** Modelos fine-tuned PT-BR NÃO melhoraram accuracy neste áudio real.\n\n"
        else:
            if baseline_summary:
                improvement = winner[1]['avg_accuracy'] - baseline_summary['avg_accuracy']
                report += f"### ✅ TROCAR PARA {winner[0].upper()}\n\n"
                report += f"Melhoria de **{improvement:+.2f} pontos percentuais** sobre baseline.\n\n"
                model_info = [m for m in MODELS if m['name'] == winner[0]][0]
                report += f"**Recomendação:** Usar `{model_info['model_name']}`\n\n"

    report += "\n---\n\n"
    report += f"**Teste concluído:** {timestamp}\n"
    report += f"**Precisão usada:** {COMPUTE_TYPE}\n"

    return report

# ===========================================================
# MAIN
# ===========================================================

async def main():
    """Executa teste ABC."""

    print("\n" + "="*70)
    print("TESTE ABC - COMPARAÇÃO FLOAT32")
    print("A: Baseline | B: jlondonobo | C: pierreguillou")
    print("Todos com faster-whisper float32")
    print("="*70)

    all_results = {}

    for model_config in MODELS:
        results = await test_model(model_config)
        all_results[model_config['name']] = results

    print("\n" + "="*70)
    print("GERANDO RELATÓRIO")
    print("="*70)

    report = generate_report(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / f"test_abc_float32_{timestamp}.md"
    report_path.write_text(report, encoding='utf-8')

    json_path = REPORT_DIR / f"test_abc_float32_{timestamp}.json"
    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"\n✅ Relatório: {report_path}")
    print(f"✅ Dados: {json_path}")

    print("\n" + "="*70)
    print("✅ TESTE COMPLETO")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
