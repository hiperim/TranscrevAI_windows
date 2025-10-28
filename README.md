# TranscrevAI

## Visão Geral
O TranscrevAI é uma aplicação de alto desempenho para transcrição de áudio e diarização de locutores. Ele recebe um áudio como entrada e fornece uma transcrição completa, identificando quem disse o quê e quando. Foi projetado para ser uma ferramenta poderosa para quem precisa de transcrições rápidas e precisas de conversas, reuniões ou gravações.

## Funcionalidades
- **Transcrição de Alto Desempenho:** Utiliza o modelo Whisper da OpenAI para transcrição local e rápida.
- **Diarização de Locutores:** Identifica diferentes locutores no áudio.
- **Geração de Legendas SRT:** Cria arquivos de legenda para vídeos.
- **Geração de Vídeos MP4:** Produz vídeos com legendas embutidas sobre um fundo preto.
- **Atualizações de Progresso em Tempo Real:** Possui uma interface WebSocket para interação e monitoramento do progresso.

## Tecnologias Utilizadas
- **Backend:** Python 3.11, FastAPI
- **Modelos de IA/ML:**
    - **Transcrição:** OpenAI's Whisper (via `faster-whisper` para otimização de CPU)
    - **Diarização:** `pyannote.audio`
- **Biblioteca Principal de ML:** PyTorch
- **Comunicação em Tempo Real:** WebSockets
- **Processamento de Áudio/Vídeo:** FFmpeg, Pydub

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

## Como Usar

Para executar a aplicação, use o seguinte comando na raiz do projeto:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
A aplicação estará disponível em `http://localhost:8000`. Para HTTPS, siga as instruções em `SSL_SETUP.md`.

## Desenvolvimento

Para executar os testes, use `pytest`:
```bash
pytest
```

## Arquitetura
O sistema é construído em um backend FastAPI usando uma arquitetura assíncrona para lidar com tarefas de transcrição longas sem bloqueios. WebSockets são usados para comunicação em tempo real com o cliente. Tarefas de machine learning que são intensivas em CPU são descarregadas para um pool de threads para evitar o bloqueio do loop de eventos principal.

Um diagrama de arquitetura mais detalhado será fornecido em `ARCHITECTURE.md`.

## Licença
Este projeto ainda não possui uma licença.

---
---

# TranscrevAI (English)

## Overview
TranscrevAI is a high-performance audio transcription and speaker diarization application. It takes an audio input and provides a complete transcription, identifying who said what and when. It's designed to be a powerful tool for anyone needing fast and accurate transcriptions of conversations, meetings, or recordings.

## Features
- **High-Performance Transcription:** Utilizes OpenAI's Whisper model for fast, local transcription.
- **Speaker Diarization:** Identifies different speakers in the audio.
- **SRT Subtitle Generation:** Creates subtitle files for videos.
- **MP4 Video Generation:** Produces videos with embedded subtitles over a black background.
- **Live Progress Updates:** Features a WebSocket interface for interaction and progress monitoring.

## Tech Stack
- **Backend:** Python 3.11, FastAPI
- **AI/ML Models:**
    - **Transcription:** OpenAI's Whisper (via `faster-whisper` for CPU optimization)
    - **Diarization:** `pyannote.audio`
- **Core ML Library:** PyTorch
- **Real-time Communication:** WebSockets
- **Audio/Video Processing:** FFmpeg, Pydub

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

## Usage

To run the application, use the following command from the project root:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
The application will be available at `http://localhost:8000`. For HTTPS, follow the instructions in `SSL_SETUP.md`.

## Development

To run the test suite, use pytest:
```bash
pytest
```

## Architecture
The system is built on a FastAPI backend using an asynchronous architecture to handle long-running transcription tasks without blocking. WebSockets are used for real-time communication with the client. CPU-bound machine learning tasks are offloaded to a thread pool to prevent blocking the main event loop.

A more detailed architecture diagram will be provided in `ARCHITECTURE.md`.

## License
This project is not yet licensed.

---