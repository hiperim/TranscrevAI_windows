# TranscrevAI

## Visão Geral
O TranscrevAI é uma aplicação de alto desempenho para transcrição de áudio e diarização de locutores. Ele recebe um áudio como entrada e fornece uma transcrição completa, identificando quem disse o quê e quando. Foi projetado para ser uma ferramenta poderosa para quem precisa de transcrições rápidas e precisas de conversas, reuniões ou gravações.

Toda a transcrição ocorre localmente na máquina onde o servidor está rodando, sem o uso de nenhuma API externa, garantindo a privacidade dos dados. Esta arquitetura offline alinha-se fortemente com os princípios de segurança e minimização de dados expostos da Lei Geral de Proteção de Dados brasileira (LGPD).

## Funcionalidades
- **Transcrição de Alto Desempenho:** Utiliza o modelo faster-whisper para transcrição local e rápida (implementação otimizada do Whisper da OpenAI para CPU);
- **Diarização de Locutores:** Identifica diferentes locutores no áudio usando pyannote.audio;
- **Gravação ao Vivo:** Permite a gravação de áudio diretamente no navegador, com buffering em disco para suportar gravações longas sem consumir excesso de RAM;
- **Upload de Ficheiros:** Suporta o upload de ficheiros de áudio pré-gravados;
- **Geração de Legendas .srt:** Cria arquivos de legenda para vídeos;
- **Geração de Vídeos .mp4:** Produz vídeos com legendas embutidas sobre um fundo preto;
- **Atualizações de Progresso na UI:** Possui uma interface WebSocket com interação e monitoramento do progresso pelo usuário.

## Tecnologias Utilizadas
- **Backend:** Python 3.11, FastAPI
- **Modelos de IA/ML:**
    - **Transcrição:** faster-whisper
    - **Diarização:** pyannote.audio
- **Biblioteca Principal de ML:** PyTorch
- **Comunicação em Tempo Real:** WebSockets
- **Processamento de Áudio/Vídeo:** FFmpeg, Pydub

## Arquitetura
A aplicação é construída sobre o FastAPI e segue uma arquitetura moderna baseada em Injeção de Dependência (DI), garantindo que os componentes sejam modulares, estáveis e geridos de forma eficiente.

   - Serviços Modulares: Cada funcionalidade principal é encapsulada num serviço (TranscriptionService, PyannoteDiarizer, LiveAudioProcessor).
   - Gestão de Sessões ('SessionManager'): Gere o ciclo de vida de cada sessão de utilizador (upload ou gravação ao vivo), incluindo a limpeza automática de sessões antigas para libertar recursos.
   - Processamento em Background: Tarefas pesadas (transcrição, diarização) são executadas num worker thread separado para não bloquear o servidor principal, garantindo que a interface permaneça responsiva.
   - Buffering em Disco: Durante a gravação ao vivo, os pedaços de áudio são armazenados temporariamente em disco, permitindo gravações longas com baixo consumo de memória.
   
## Performance
O TranscrevAI inclui funcionalidade de Otimização Adaptativa de Threads. No arranque, a aplicação deteta o hardware da máquina (núcleos de CPU, RAM) e aloca dinamicamente o número ideal de threads para as tarefas de transcrição e diarização, otimizando o desempenho para a sua configuração específica.

## Instalação

1.  **Clone o repositório:**
    ```bash
    git clone <repository-url>
    cd transcrevai_windows
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Instale as dependências de produção:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Para desenvolvimento, instale as dependências de desenvolvimento:**
    ```bash
    pip install -r requirements-dev.txt
    ```

5. **Descarregue os modelos de IA/ML:**
    Execute o seguinte script para descarregar os modelos de transcrição e diarização para a pasta local models/.cache.

    ```bash
    python setups_certs_SSL_ModelsCache/download_models.py
    ```

6. **(Ambiente de Desenvolvimento) Configuração Local com '.env':**
    Para configurações personalizadas, como usar HTTPS localmente, crie um ficheiro .env na raiz do projeto. Copie o exemplo abaixo e ajuste se
    necessário.

    `.env.example`
    ```
    # Exemplo de configuração para HTTPS local
    # TRANSCREVAI_SSL_CERT=certs/localhost+2.pem
    # TRANSCREVAI_SSL_KEY=certs/localhost+2-key.pem
    ```

## Como Usar

Para executar a aplicação, use o seguinte comando na raiz do projeto:
```bash
python main.py
```
O servidor irá iniciar e o log indicará se está a ser executado em `http` ou `https` e em que endereço (normalmente `http://localhost:8000`).

## Desenvolvimento

Para executar os testes, use `pytest`:
```bash
pytest
```

---
---

# TranscrevAI (English)

## Overview
TranscrevAI is a high-performance application for audio transcription and speaker diarization. It takes an audio input and provides a complete transcription, identifying who said what and when. It is designed to be a powerful tool for anyone who needs fast and accurate transcriptions of conversations, meetings, or recordings.

All transcription occurs locally on the machine where the server is running, without the use of any external APIs, ensuring data privacy. This offline architecture strongly aligns with the security and data minimization principles of the brazilian General Data Protection Law (LGPD).

## Features
- **High-Performance Transcription:** Uses the faster-whisper model for fast, local transcription (a CPU-optimized implementation of OpenAI's Whisper).
- **Speaker Diarization:** Identifies different speakers in the audio using pyannote.audio.
- **Live Recording:** Allows audio recording directly in the browser, with disk buffering to support long recordings without consuming excess RAM.
- **File Upload:** Supports uploading pre-recorded audio files.
- **.srt Subtitle Generation:** Creates subtitle files for videos.
- **.mp4 Video Generation:** Produces videos with embedded subtitles over a black background.
- **UI Progress Updates:** Features a WebSocket interface for user interaction and progress monitoring.

## Tech Stack
- **Backend:** Python 3.11, FastAPI
- **AI/ML Models:**
    - **Transcription:** faster-whisper
    - **Diarization:** pyannote.audio
- **Core ML Library:** PyTorch
- **Real-time Communication:** WebSockets
- **Audio/Video Processing:** FFmpeg, Pydub

## Architecture
The application is built on FastAPI and follows a modern Dependency Injection (DI) based architecture, ensuring that components are modular, stable, and efficiently managed.

   - **Modular Services:** Each core functionality is encapsulated in a service (TranscriptionService, PyannoteDiarizer, LiveAudioProcessor).
   - **Session Management ('SessionManager'):** Manages the lifecycle of each user session (upload or live recording), including automatic cleanup of old sessions to free up resources.
   - **Background Processing:** Heavy tasks (transcription, diarization) are executed in a separate worker thread to avoid blocking the main server, ensuring the interface remains responsive.
   - **Disk Buffering:** During live recording, audio chunks are temporarily stored on disk, allowing for long recordings with low memory consumption.
   
## Performance
TranscrevAI includes an Adaptive Thread Optimization feature. On startup, the application detects the machine's hardware (CPU cores, RAM) and dynamically allocates the optimal number of threads for transcription and diarization tasks, optimizing performance for your specific setup.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd transcrevai_windows
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install production dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **For development, install development dependencies:**
    ```bash
    pip install -r requirements-dev.txt
    ```

5. **Download the AI/ML models:**
    Run the following script to download the transcription and diarization models to the local `models/.cache` folder.

    ```bash
    python setups_certs_SSL_ModelsCache/download_models.py
    ```

6. **(Development Environment) Local Configuration with '.env':
    For custom settings, such as using HTTPS locally, create a `.env` file in the project root. Copy the example below and adjust as needed.

    `.env.example`
    ```
    # Example configuration for local HTTPS
    # TRANSCREVAI_SSL_CERT=certs/localhost+2.pem
    # TRANSCREVAI_SSL_KEY=certs/localhost+2-key.pem
    ```

## Usage

To run the application, use the following command from the project root:
```bash
python main.py
```
The server will start, and the log will indicate whether it is running on `http` or `https` and at what address (usually `http://localhost:8000`).

## Development

To run the tests, use `pytest`:
```bash
pytest
```
