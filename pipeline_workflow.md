# Fluxo do Pipeline do TranscrevAI

Este documento fornece uma representação visual e detalhada do pipeline de processamento de áudio na aplicação TranscrevAI, incorporando os componentes de arquitetura modular mais recentes.

## Fluxo de Alto Nível

O processo pode ser visualizado como uma série de estágios sequenciais e paralelos, com gestão centralizada de sessões e execução de tarefas intensivas em background.

```
(1) Entrada de Áudio      --> (2) Criação/Gestão da Sessão      --> (3) Pré-Processamento de Áudio
(Upload/Gravação)           (WebSocket, SessionManager)        (LiveAudioProcessor, Normalização)
      |                             |                                  |
      |                             v                                  v
      |                     (7) Saída Final                      <-- (5) Diarização
      |                     (Exibição na UI, SRT/MP4)              (PyannoteDiarizer, Atribui Locutores)
      |                             ^                                  ^
      |                             |                                  |
      +------------------------> (4) Transcrição ---------------------+
                                 (faster-whisper, em Worker Thread)
                                        |
                                        v
                                (6) Pós-Processamento
                                (VAD, Correções PT-BR)
```

## Detalhamento Passo a Passo

1.  **Entrada de Áudio:**
    *   **Upload:** O utilizador faz o upload de um ficheiro de áudio existente (ex: .wav, .mp3, .webm) através da interface web.
    *   **Gravação ao Vivo:** O utilizador inicia uma gravação diretamente no navegador. O áudio é enviado em tempo real via WebSocket.

2.  **Criação/Gestão da Sessão (`SessionManager`):**
    *   Para cada interação (upload ou gravação), uma sessão única é criada e gerida pelo `SessionManager`.
    *   O `SessionManager` monitoriza o estado da sessão, a atividade, e é responsável pela limpeza de sessões expiradas ou completas e pelos seus recursos associados.
    *   É estabelecida uma conexão WebSocket para comunicação contínua e em tempo real com o utilizador.

3.  **Pré-Processamento de Áudio (`LiveAudioProcessor` & Outros):**
    *   **Gravação ao Vivo:** O `LiveAudioProcessor` recebe os chunks de áudio via WebSocket, armazena-os temporariamente em disco (disk buffering) e converte-os para um formato standard (.wav) quando a gravação termina.
    *   **Upload:** O ficheiro de áudio é recebido e armazenado temporariamente.
    *   **Normalização:** O áudio é normalizado para um volume padrão.
    *   **Análise de Qualidade:** O `AudioQualityAnalyzer` avalia a qualidade do áudio e fornece avisos, se necessário.

4.  **Transcrição (`TranscriptionService` - em Worker Thread):**
    *   O serviço `TranscriptionService` (utilizando `faster-whisper`) recebe o áudio pré-processado.
    *   Esta tarefa intensiva em CPU é descarregada para um _worker thread_ separado para não bloquear o loop de eventos principal do FastAPI, mantendo a aplicação responsiva.
    *   O `faster-whisper` transcreve a fala em texto.

5.  **Diarização (`PyannoteDiarizer` - em Worker Thread):**
    *   O serviço `PyannoteDiarizer` (utilizando `pyannote.audio`) analisa o áudio e a transcrição para distinguir entre diferentes locutores, atribuindo um ID único a cada um.
    *   Esta tarefa também pode ser executada num _worker thread_ para otimização de performance.

6.  **Pós-Processamento:**
    *   **VAD (Voice Activity Detection):** Os segmentos de fala são identificados (se não feito antes) e o silêncio é removido para melhorar a precisão da diarização e transcrição final.
    *   **Correções PT-BR:** Aplicação de correções ortográficas e gramaticais específicas para o português do Brasil.

7.  **Saída Final:**
    *   A transcrição final com diarização (texto com rótulos de locutor e timestamps) é enviada de volta para a interface web do utilizador via WebSocket.
    *   O utilizador pode então fazer o download dos resultados como um ficheiro de legenda (`.srt`) ou um vídeo MP4 com as legendas embutidas.

---
---

# TranscrevAI Pipeline Workflow (English)

This document provides a visual and detailed representation of the audio processing pipeline in the TranscrevAI application, incorporating the latest modular architecture components.

## High-Level Workflow

The process can be visualized as a series of sequential and parallel stages, with centralized session management and background execution of intensive tasks.

```
(1) Audio Input            --> (2) Session Creation/Management   --> (3) Audio Pre-processing
(Upload/Recording)         (WebSocket, SessionManager)         (LiveAudioProcessor, Normalization)
      |                              |                                   |
      |                              v                                   v
      |                      (7) Final Output                      <-- (5) Diarization
      |                      (Display in UI, SRT/MP4)              (PyannoteDiarizer, Assigns Speakers)
      |                              ^                                   ^
      |                              |                                   |
      +-------------------------> (4) Transcription ------------------+
                                  (faster-whisper, in Worker Thread)
                                         |
                                         v
                                 (6) Post-processing
                                 (VAD, PT-BR Corrections)
```

## Step-by-Step Breakdown

1.  **Audio Input:**
    *   **Upload:** The user uploads an existing audio file (e.g., .wav, .mp3, .webm) through the web interface.
    *   **Live Recording:** The user starts a recording directly in the browser. The audio is sent in real-time via WebSocket.

2.  **Session Creation/Management (`SessionManager`):**
    *   For each interaction (upload or recording), a unique session is created and managed by the `SessionManager`.
    *   The `SessionManager` monitors the session state, activity, and is responsible for cleaning up expired or completed sessions and their associated resources.
    *   A WebSocket connection is established for continuous, real-time communication with the user.

3.  **Audio Pre-processing (`LiveAudioProcessor` & Others):**
    *   **Live Recording:** The `LiveAudioProcessor` receives audio chunks via WebSocket, temporarily stores them on disk (disk buffering), and converts them to a standard format (.wav) when the recording ends.
    *   **Upload:** The audio file is received and stored temporarily.
    *   **Normalization:** The audio is normalized to a standard volume.
    *   **Quality Analysis:** The `AudioQualityAnalyzer` assesses the audio quality and provides warnings if necessary.

4.  **Transcription (`TranscriptionService` - in Worker Thread):**
    *   The `TranscriptionService` (using `faster-whisper`) receives the pre-processed audio.
    *   This CPU-intensive task is offloaded to a separate _worker thread_ to avoid blocking the main FastAPI event loop, keeping the application responsive.
    *   `faster-whisper` transcribes the speech into text.

5.  **Diarization (`PyannoteDiarizer` - in Worker Thread):**
    *   The `PyannoteDiarizer` service (using `pyannote.audio`) analyzes the audio and transcription to distinguish between different speakers, assigning a unique ID to each one.
    *   This task can also be executed in a _worker thread_ for performance optimization.

6.  **Post-processing:**
    *   **VAD (Voice Activity Detection):** Speech segments are identified (if not done before) and silence is trimmed to improve the accuracy of the final transcription and diarization.
    *   **PT-BR Corrections:** Application of spelling and grammatical corrections specific to Brazilian Portuguese.

7.  **Final Output:**
    *   The final, diarized transcription (text with speaker labels and timestamps) is sent back to the user's web interface via WebSocket.
    *   The user can then download the results as a subtitle file (`.srt`) or an MP4 video with embedded subtitles.