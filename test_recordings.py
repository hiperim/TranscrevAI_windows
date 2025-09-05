#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script para testar os últimos 5 arquivos de gravação com as melhorias implementadas
"""
import asyncio
import json
import websockets
import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Últimos 5 arquivos de gravação para teste
TEST_FILES = [
    "recording_1757012370.wav",
    "recording_1757012909.wav", 
    "recording_1757015324.wav",
    "recording_1757017997.wav",
    "recording_1757103553.wav"
]

RECORDINGS_DIR = Path("data/recordings")
SERVER_URL = "ws://localhost:8001/ws/test_session"

async def test_transcription(audio_file, language="pt"):
    """Testa a transcrição de um arquivo de áudio"""
    
    if not os.path.exists(audio_file):
        print(f"❌ Arquivo não encontrado: {audio_file}")
        return None
        
    print(f"\n🎤 Testando: {os.path.basename(audio_file)} (idioma: {language})")
    print(f"📁 Arquivo: {audio_file}")
    
    try:
        # Conecta ao WebSocket
        async with websockets.connect(SERVER_URL) as websocket:
            
            # Envia solicitação de transcrição
            request = {
                "type": "transcribe",
                "audio_file": str(audio_file),
                "language": language,
                "session_id": "test_session"
            }
            
            await websocket.send(json.dumps(request))
            print("📤 Solicitação enviada...")
            
            # Recebe e processa respostas
            transcription_result = None
            
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "progress":
                        transcription_progress = data.get("transcription", 0)
                        diarization_progress = data.get("diarization", 0)
                        print(f"⏳ Progresso - Transcrição: {transcription_progress}%, Diarização: {diarization_progress}%")
                    
                    elif data.get("type") == "result":
                        transcription_result = data
                        print("✅ Resultado recebido!")
                        break
                        
                    elif data.get("type") == "error":
                        print(f"❌ Erro: {data.get('message', 'Unknown error')}")
                        break
                        
                except asyncio.TimeoutError:
                    print("❌ Timeout aguardando resposta")
                    break
                except Exception as e:
                    print(f"❌ Erro na comunicação: {e}")
                    break
            
            return transcription_result
            
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        return None

def print_result_analysis(result, filename):
    """Analisa e exibe os resultados da transcrição"""
    if not result:
        print("❌ Sem resultado para analisar")
        return
        
    print(f"\n📊 ANÁLISE - {filename}")
    print("=" * 60)
    
    # Informações gerais
    method = result.get("method", "unknown")
    speakers_detected = result.get("speakers_detected", 0)
    processing_time = result.get("processing_time", 0)
    
    print(f"🔧 Método: {method}")
    print(f"👥 Speakers detectados: {speakers_detected}")
    print(f"⏱️ Tempo de processamento: {processing_time:.2f}s")
    
    # Transcrição
    transcription_data = result.get("transcription_data", [])
    print(f"📝 Segmentos de transcrição: {len(transcription_data)}")
    
    if transcription_data:
        print("\n📜 TEXTO TRANSCRITO:")
        full_text = ""
        for segment in transcription_data:
            text = segment.get("text", "").strip()
            if text:
                full_text += text + " "
                
        print(f"'{full_text.strip()}'")
        
        print("\n🔍 DETALHES DOS SEGMENTOS:")
        for i, segment in enumerate(transcription_data[:5]):  # Primeiros 5 segmentos
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            confidence = segment.get("confidence", 0)
            print(f"  [{i+1}] {start:.2f}s-{end:.2f}s: '{text}' (conf: {confidence:.3f})")
    
    # Diarização
    diarization_segments = result.get("diarization_segments", [])
    print(f"\n👥 Segmentos de diarização: {len(diarization_segments)}")
    
    if diarization_segments:
        print("🗣️ DETALHES DOS SPEAKERS:")
        for i, segment in enumerate(diarization_segments[:5]):  # Primeiros 5 segmentos
            start = segment.get("start", 0)
            end = segment.get("end", 0) 
            speaker = segment.get("speaker", "Unknown")
            confidence = segment.get("confidence", 0)
            print(f"  [{i+1}] {start:.2f}s-{end:.2f}s: {speaker} (conf: {confidence:.3f})")

async def main():
    """Função principal do teste"""
    print("🚀 INICIANDO TESTES DE TRANSCRIÇÃO")
    print("=" * 60)
    print("🔧 Testando correções implementadas:")
    print("   ✅ Fix PyAudioAnalysis array ambiguity")
    print("   ✅ Fix Portuguese transcription accuracy") 
    print("   ✅ Optimized Whisper config for PT/ES")
    print("   ✅ Enhanced diagnostic logging")
    print("=" * 60)
    
    results = []
    
    for audio_file in TEST_FILES:
        audio_path = RECORDINGS_DIR / audio_file
        result = await test_transcription(audio_path, language="pt")
        
        if result:
            results.append((audio_file, result))
            print_result_analysis(result, audio_file)
        else:
            print(f"❌ Falha no teste: {audio_file}")
            
        print("\n" + "─" * 60)
        await asyncio.sleep(1)  # Pausa entre testes
    
    # Resumo final
    print(f"\n🎯 RESUMO DOS TESTES")
    print("=" * 60)
    print(f"📊 Total de arquivos testados: {len(TEST_FILES)}")
    print(f"✅ Sucessos: {len(results)}")
    print(f"❌ Falhas: {len(TEST_FILES) - len(results)}")
    
    if results:
        avg_time = sum(r[1].get("processing_time", 0) for r in results) / len(results)
        avg_speakers = sum(r[1].get("speakers_detected", 0) for r in results) / len(results)
        print(f"⏱️ Tempo médio de processamento: {avg_time:.2f}s")
        print(f"👥 Média de speakers detectados: {avg_speakers:.1f}")
        
        # Verificar se o problema português foi resolvido
        portuguese_words_detected = False
        for filename, result in results:
            transcription_data = result.get("transcription_data", [])
            full_text = " ".join(segment.get("text", "") for segment in transcription_data).lower()
            if any(word in full_text for word in ["televisão", "régua", "copo", "teste"]):
                portuguese_words_detected = True
                break
        
        if portuguese_words_detected:
            print("🇧🇷 ✅ Palavras portuguesas detectadas corretamente!")
        else:
            print("🇧🇷 ❓ Nenhuma palavra portuguesa reconhecida nos testes")

if __name__ == "__main__":
    asyncio.run(main())