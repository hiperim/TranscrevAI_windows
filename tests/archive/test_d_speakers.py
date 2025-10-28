# tests/test_d_speakers.py
"""
Script simplificado para testar o pipeline completo com d.speakers.wav
Mostra: transcrição, diarização, métricas (WER, speed ratio, speaker accuracy)
"""

import asyncio
import time
import sys
from pathlib import Path
import librosa

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from tests.metrics import calculate_wer

# ===========================================================
# CONFIGURAÇÃO
# ===========================================================

# Usar configuração atual do projeto
MODEL_NAME = "medium"  # Nome do modelo atual
COMPUTE_TYPE = "int8"  # Tipo de compute atual
DEVICE = "cpu"

# Arquivo de teste
TEST_AUDIO = "d.speakers.wav"
EXPECTED_SPEAKERS = 2

# Paths
AUDIO_DIR = Path(__file__).parent.parent / "data" / "recordings"
TRUTH_DIR = Path(__file__).parent / "ground_truth"

# ===========================================================

async def test_d_speakers():
    """Testa o pipeline completo com d.speakers.wav"""

    print("="*60)
    print("TESTE DE PIPELINE COMPLETO - d.speakers.wav")
    print("="*60)
    print(f"Modelo: {MODEL_NAME}")
    print(f"Compute Type: {COMPUTE_TYPE}")
    print(f"Device: {DEVICE}")
    print("="*60)

    # 1. Verificar se arquivos existem
    audio_path = AUDIO_DIR / TEST_AUDIO
    truth_path = TRUTH_DIR / "d_speakers.txt"

    if not audio_path.exists():
        print(f"❌ ERRO: Arquivo de áudio não encontrado: {audio_path}")
        return

    if not truth_path.exists():
        print(f"❌ ERRO: Ground truth não encontrado: {truth_path}")
        return

    # 2. Carregar ground truth
    expected_text = truth_path.read_text(encoding="utf-8").strip()
    print(f"\n📝 Ground Truth Text ({len(expected_text)} caracteres):")
    print(f"   {expected_text[:100]}..." if len(expected_text) > 100 else f"   {expected_text}")
    print(f"\n👥 Expected Speakers: {EXPECTED_SPEAKERS}")

    # 3. Obter duração do áudio
    audio_duration = librosa.get_duration(path=str(audio_path))
    print(f"\n⏱️  Audio Duration: {audio_duration:.2f}s")

    # 4. Inicializar serviços
    print("\n🔧 Inicializando serviços...")
    try:
        transcription_service = TranscriptionService(
            model_name=MODEL_NAME,
            device=DEVICE
        )
        await transcription_service.initialize()
        diarizer = PyannoteDiarizer()
        print("   ✅ Serviços inicializados com sucesso")
    except Exception as e:
        print(f"   ❌ Erro ao inicializar serviços: {e}")
        return

    # 5. Executar pipeline completo
    print("\n🎬 Executando pipeline completo...")
    start_time = time.time()

    try:
        # Transcrição
        print("   → Transcrevendo...")
        transcription_result = await transcription_service.transcribe_with_enhancements(
            str(audio_path),
            beam_size=5,
            best_of=5
        )

        # Diarização
        print("   → Diarizando...")
        diarization_result = await diarizer.diarize(
            str(audio_path),
            transcription_result.segments
        )

        end_time = time.time()
        print("   ✅ Pipeline concluído")

    except Exception as e:
        print(f"   ❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Calcular métricas
    processing_time = end_time - start_time
    processing_ratio = processing_time / audio_duration
    actual_text = transcription_result.text
    detected_speakers = diarization_result["num_speakers"]

    # WER (Word Error Rate)
    wer = calculate_wer(expected_text, actual_text)
    transcription_accuracy = max(0, (1 - wer) * 100)

    # Diarization accuracy
    diarization_accuracy = 100.0 if detected_speakers == EXPECTED_SPEAKERS else 0.0

    # 7. Mostrar resultados
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)

    print("\n📊 MÉTRICAS DE PERFORMANCE:")
    print(f"   Processing Time: {processing_time:.2f}s")
    print(f"   Speed Ratio: {processing_ratio:.2f}x")
    print(f"   Target: ≤2.0x (CPU-only)")
    if processing_ratio <= 2.0:
        print("   ✅ PASSOU - Dentro do target")
    else:
        print("   ⚠️  AVISO - Acima do target")

    print("\n📝 TRANSCRIÇÃO:")
    print(f"   Accuracy (1-WER): {transcription_accuracy:.2f}%")
    print(f"   WER: {wer:.4f}")
    print(f"   Target: ≥90%")
    if transcription_accuracy >= 90.0:
        print("   ✅ PASSOU - Accuracy adequada")
    else:
        print("   ⚠️  AVISO - Abaixo do target")

    print(f"\n   Texto obtido ({len(actual_text)} caracteres):")
    print(f"   {actual_text[:200]}..." if len(actual_text) > 200 else f"   {actual_text}")

    print("\n👥 DIARIZAÇÃO:")
    print(f"   Detected Speakers: {detected_speakers}")
    print(f"   Expected Speakers: {EXPECTED_SPEAKERS}")
    print(f"   Accuracy: {diarization_accuracy:.0f}%")
    if diarization_accuracy == 100.0:
        print("   ✅ PASSOU - Speaker count correto")
    else:
        print("   ❌ FALHOU - Speaker count incorreto")

    # Mostrar segmentos com speakers
    print("\n   Segmentos por speaker:")
    for seg in diarization_result.get("segments", [])[:5]:  # Primeiros 5 segmentos
        speaker = seg.get("speaker", "unknown")
        text = seg.get("text", "")[:50]
        print(f"   [{speaker}] {text}...")

    if len(diarization_result.get("segments", [])) > 5:
        print(f"   ... (mais {len(diarization_result['segments']) - 5} segmentos)")

    # 8. Resumo final
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)

    all_passed = (
        processing_ratio <= 2.0 and
        transcription_accuracy >= 90.0 and
        diarization_accuracy == 100.0
    )

    if all_passed:
        print("✅ TODOS OS TESTES PASSARAM")
    else:
        print("⚠️  ALGUNS TESTES FALHARAM - Revisar resultados acima")

    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_d_speakers())
