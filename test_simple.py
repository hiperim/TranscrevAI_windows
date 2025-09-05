#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste simples para verificar as correções implementadas
"""
import asyncio
import json
import websockets
import os
from pathlib import Path

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

async def test_single_file(audio_file, language="pt"):
    """Testa a transcrição de um arquivo"""
    
    audio_path = RECORDINGS_DIR / audio_file
    if not audio_path.exists():
        print(f"ERRO: Arquivo não encontrado: {audio_path}")
        return None
        
    print(f"\n--- Testando: {audio_file} (idioma: {language}) ---")
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            
            # Envia solicitação
            request = {
                "type": "transcribe",
                "audio_file": str(audio_path),
                "language": language,
                "session_id": "test_session"
            }
            
            await websocket.send(json.dumps(request))
            print("Solicitação enviada...")
            
            # Recebe respostas
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=120.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "progress":
                        transcription = data.get("transcription", 0)
                        diarization = data.get("diarization", 0)
                        print(f"Progresso - Transcrição: {transcription}%, Diarização: {diarization}%")
                    
                    elif data.get("type") == "result":
                        print("Resultado recebido!")
                        return data
                        
                    elif data.get("type") == "error":
                        print(f"ERRO: {data.get('message', 'Erro desconhecido')}")
                        return None
                        
                except asyncio.TimeoutError:
                    print("TIMEOUT: Nenhuma resposta em 120s")
                    return None
                except Exception as e:
                    print(f"ERRO na comunicação: {e}")
                    return None
            
    except Exception as e:
        print(f"ERRO na conexão: {e}")
        return None

def analyze_result(result, filename):
    """Analisa o resultado"""
    if not result:
        print("SEM RESULTADO PARA ANALISAR")
        return
        
    print(f"\nANALISE - {filename}")
    print("=" * 50)
    
    method = result.get("method", "unknown")
    speakers = result.get("speakers_detected", 0)
    time_taken = result.get("processing_time", 0)
    
    print(f"Método: {method}")
    print(f"Speakers detectados: {speakers}")
    print(f"Tempo: {time_taken:.2f}s")
    
    # Transcrição
    transcription_data = result.get("transcription_data", [])
    print(f"Segmentos transcrição: {len(transcription_data)}")
    
    if transcription_data:
        full_text = ""
        for segment in transcription_data:
            text = segment.get("text", "").strip()
            if text:
                full_text += text + " "
                
        print(f"TEXTO: '{full_text.strip()}'")
        
        # Verificar palavras portuguesas
        portuguese_indicators = ["teste", "televisão", "régua", "copo", "português", "brasileiro"]
        text_lower = full_text.lower()
        found_pt_words = [word for word in portuguese_indicators if word in text_lower]
        if found_pt_words:
            print(f"Palavras PT detectadas: {found_pt_words}")
        
    # Diarização
    diar_segments = result.get("diarization_segments", [])
    print(f"Segmentos diarização: {len(diar_segments)}")

async def main():
    """Função principal"""
    print("INICIANDO TESTES DE TRANSCRICAO")
    print("Correções testadas:")
    print("- Fix PyAudioAnalysis array ambiguity")
    print("- Fix Portuguese transcription accuracy") 
    print("- Optimized Whisper config for PT/ES")
    print("- Enhanced diagnostic logging")
    print("=" * 50)
    
    results = []
    
    for audio_file in TEST_FILES:
        result = await test_single_file(audio_file, language="pt")
        
        if result:
            results.append((audio_file, result))
            analyze_result(result, audio_file)
        else:
            print(f"FALHA: {audio_file}")
            
        print("-" * 50)
        await asyncio.sleep(2)  # Pausa entre testes
    
    # Resumo
    print(f"\nRESUMO DOS TESTES")
    print("=" * 50)
    print(f"Total testados: {len(TEST_FILES)}")
    print(f"Sucessos: {len(results)}")
    print(f"Falhas: {len(TEST_FILES) - len(results)}")
    
    if results:
        avg_time = sum(r[1].get("processing_time", 0) for r in results) / len(results)
        print(f"Tempo médio: {avg_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())