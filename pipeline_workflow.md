# Fluxo do Pipeline do TranscrevAI

Este documento fornece uma representação visual do pipeline de processamento de áudio na aplicação TranscrevAI.

## Fluxo de Alto Nível

O processo pode ser visualizado como uma série de estágios sequenciais, começando pela entrada de áudio e terminando com a saída final de transcrição e diarização.

```
(1) Entrada de Áudio      --> (2) Criação da Sessão      --> (3) Processamento de Áudio
(Upload/Gravação)           (Conexão WebSocket)          (Normalização, VAD)
      |                             |                            |
      |                             v                            v
      |                     (6) Saída Final              <-- (5) Diarização
      |                     (Exibição na UI, SRT/MP4)      (Atribui Locutores)
      |                             ^                            ^
      |                             |                            |
      +------------------------> (4) Transcrição ---------------+
                                 (Modelo Whisper)
```

## Detalhamento Passo a Passo

1.  **Entrada de Áudio:** O usuário faz o upload de um arquivo de áudio existente ou grava um novo através da interface web.

2.  **Criação da Sessão:** Uma sessão única é criada para o usuário através de uma conexão WebSocket, que gerencia o estado e a comunicação durante todo o processo.

3.  **Processamento de Áudio:** O áudio bruto é preparado para a transcrição. Isso envolve:
    - **Normalização:** Padronização do volume do áudio.
    - **Detecção de Atividade de Voz (VAD):** Identificação de segmentos do áudio que contêm fala, removendo o silêncio para melhorar a eficiência.

4.  **Transcrição:** Os segmentos de áudio processados são enviados ao modelo `faster-whisper`, que transcreve a fala em texto.

5.  **Diarização:** O modelo `pyannote.audio` analisa o áudio para distinguir entre diferentes locutores, atribuindo um ID único a cada um. O texto da transcrição é então alinhado com esses IDs de locutor.

6.  **Saída Final:** A transcrição final com diarização (texto com rótulos de locutor) é enviada de volta para a interface web do usuário via WebSocket. O usuário então tem a opção de baixar os resultados como um arquivo de legenda (`.srt`) ou um vídeo MP4.

---
---

# TranscrevAI Pipeline Workflow (English)

This document provides a visual representation of the audio processing pipeline within the TranscrevAI application.

## High-Level Workflow

The process can be visualized as a series of sequential stages, starting from the audio input and ending with the final transcription and diarization output.

```
(1) Audio Input            --> (2) Session Creation         --> (3) Audio Processing
(User Upload/Recording)        (WebSocket Connection)           (Normalization, VAD)
      |                                |                              |
      |                                v                              v
      |                        (6) Final Output             <-- (5) Diarization
      |                        (Display in UI, SRT/MP4)         (Assigns Speakers)
      |                                ^                              ^
      |                                |                              |
      +---------------------------> (4) Transcription ---------------+
                                  (Whisper Model)
```

## Step-by-Step Breakdown

1.  **Audio Input:** The user either uploads a pre-existing audio file or records a new one through the web interface.

2.  **Session Creation:** A unique session is created for the user via a WebSocket connection, which manages the state and communication for the duration of the process.

3.  **Audio Processing:** The raw audio is prepared for transcription. This involves:
    - **Normalization:** Standardizing the audio volume.
    - **Voice Activity Detection (VAD):** Identifying segments of the audio that contain speech and trimming silence to improve efficiency.

4.  **Transcription:** The processed audio segments are passed to the `faster-whisper` model, which transcribes the speech into text.

5.  **Diarização:** The `pyannote.audio` model analyzes the audio to distinguish between different speakers, assigning a unique ID to each one. The transcription text is then aligned with these speaker IDs.

6.  **Final Output:** The final, diarized transcription (text with speaker labels) is sent back to the user's web interface via WebSocket. The user is then given the option to download the results as a subtitle file (`.srt`) or an MP4 video.
---
