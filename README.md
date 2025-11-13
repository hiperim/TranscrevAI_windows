# TranscrevAI

## Visão Geral

O TranscrevAI é uma aplicação de alto desempenho para transcrição de áudio e diarização de locutores. Ele recebe um áudio como entrada e fornece uma transcrição completa, identificando quem disse o quê e quando. Foi projetado para ser uma ferramenta poderosa para quem precisa de transcrições rápidas e precisas de conversas, reuniões ou gravações.

Toda a transcrição ocorre localmente na máquina onde o servidor está rodando, sem o uso de nenhuma API externa, garantindo a privacidade dos dados. Esta arquitetura offline alinha-se fortemente com os princípios de segurança e minimização de dados expostos da Lei Geral de Proteção de Dados brasileira (LGPD).

## Funcionalidades

- **Transcrição de Alto Desempenho:** Utiliza o modelo faster-whisper para transcrição local e rápida (implementação otimizada do Whisper da OpenAI para CPU)
- **Diarização de Locutores:** Identifica diferentes locutores no áudio usando pyannote.audio com algoritmo word-level alignment
- **Gravação ao Vivo:** Permite a gravação de áudio diretamente no navegador, com buffering em disco para suportar gravações longas sem consumir excesso de RAM
- **Upload de Ficheiros:** Suporta o upload de ficheiros de áudio pré-gravados
- **Geração de Legendas .srt:** Cria arquivos de legenda para vídeos
- **Geração de Vídeos .mp4:** Produz vídeos com legendas embutidas sobre um fundo preto
- **Atualizações de Progresso em Tempo Real:** Interface WebSocket com monitoramento do progresso pelo usuário

## Tecnologias Utilizadas

- **Backend:** Python 3.11, FastAPI
- **Modelos de IA/ML:**
    - **Transcrição:** faster-whisper (Whisper medium otimizado para CPU)
    - **Diarização:** pyannote.audio 3.1
- **Biblioteca Principal de ML:** PyTorch (CPU-only, INT8 quantization)
- **Comunicação em Tempo Real:** WebSockets
- **Processamento de Áudio/Vídeo:** FFmpeg, librosa
- **Deployment:** Docker, Gunicorn/Uvicorn
- **SSL/HTTPS:** Suporte completo para desenvolvimento e produção

## Arquitetura

A aplicação é construída sobre o FastAPI e segue uma arquitetura moderna baseada em Injeção de Dependência (DI), garantindo que os componentes sejam modulares, estáveis e geridos de forma eficiente.

- **Serviços Modulares:** Cada funcionalidade principal é encapsulada num serviço (TranscriptionService, PyannoteDiarizer, LiveAudioProcessor, SessionManager)
- **Gestão de Sessões:** Ciclo de vida completo de cada sessão de utilizador com limpeza automática (timeout de 24h)
- **Processamento Assíncrono:** Tarefas pesadas executadas em worker threads separados para não bloquear o servidor principal
- **Buffering em Disco:** Gravações longas armazenadas temporariamente em disco, permitindo baixo consumo de memória
- **Otimização Adaptativa:** Detecção automática de hardware (CPU cores, RAM) e alocação dinâmica de threads

## Performance

**Métricas alcançadas:**
- **Startup time:** <30s com pre-loading de modelos
- **Memory usage:** ~2GB peak (otimizado para sistemas com 8GB RAM)
- **Processing ratio:** ~1.5x realtime
- **Accuracy PT-BR:** 90%+ com correções linguísticas pós-processamento
- **Architecture:** CPU-only com INT8 quantization para compatibilidade universal

---

## Instalação e Uso

### Opção 1: Docker Hub (Recomendado - Modelos Incluídos)

**Sem necessidade de token Hugging Face. Modelos já na imagem.**

```bash
# Pull da imagem (primeira vez, ~17GB)
docker pull hiperim/transcrevai:latest

# Executar aplicação
docker run -d -p 8000:8000 --name transcrevai hiperim/transcrevai:latest

# Acessar
# http://localhost:8000
```

**Usando docker-compose:**
```bash
docker-compose -f docker-compose.pull.yml up -d
```

**Parar aplicação:**
```bash
docker stop transcrevai
docker rm transcrevai
```

---

### Opção 2: Build Local com Docker

**Requer token Hugging Face para download dos modelos.**

1. Clone o repositório:
   ```bash
   git clone <repository-url>
   cd transcrevai_windows
   ```

2. Crie arquivo `.env` com seu token:
   ```bash
   HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

3. Build da imagem:

   **Windows:**
   ```powershell
   .\SETUPs_certs_SSL_ModelsCache\build.ps1
   ```

   **Linux/Mac:**
   ```bash
   ./SETUPs_certs_SSL_ModelsCache/build-docker.sh
   ```

4. Executar:
   ```bash
   docker-compose up -d
   ```

5. Acessar: `http://localhost:8000`

---

### Opção 3: Instalação Local (Desenvolvimento)

**Requer Python 3.11+, FFmpeg e token Hugging Face.**

1. Clone o repositório:
   ```bash
   git clone <repository-url>
   cd transcrevai_windows
   ```

2. Crie e ative ambiente virtual:
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. Instale dependências:
   ```bash
   # Produção
   pip install -r requirements.txt

   # Desenvolvimento (inclui pytest, etc)
   pip install -r requirements-dev.txt
   ```

4. Configure token no `.env`:
   ```bash
   HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

5. Download dos modelos de IA/ML:
   ```bash
   python SETUPs_certs_SSL_ModelsCache/download_models.py
   ```

   Este comando baixa (~3-5GB):
   - faster-whisper-medium (transcrição)
   - pyannote/speaker-diarization-3.1
   - pyannote/segmentation-3.0
   - pyannote/wespeaker embeddings

6. Executar aplicação:
   ```bash
   python main.py
   ```

7. Acessar: `http://localhost:8000`

---

## Configuração HTTPS (Opcional)

A aplicação suporta HTTPS para desenvolvimento e produção. HTTPS é necessário para a funcionalidade de gravação ao vivo devido aos requisitos da API `getUserMedia()` do navegador.

### Desenvolvimento (localhost)

Execute o script automatizado:
```batch
# Windows (como Administrador)
.\SETUPs_certs_SSL_ModelsCache\setup_dev_certs.bat
```

Este script instala mkcert e gera certificados locais confiáveis.

**Documentação completa:**
- [SSL_SETUP.md](./SETUPs_certs_SSL_ModelsCache/SSL_SETUP.md) - Guia completo de configuração HTTPS

---

## Testes

A aplicação inclui suite completa de testes:

```bash
# Todos os testes
pytest

# Testes específicos
pytest tests/test_services.py
pytest tests/test_performance.py
pytest tests/test_edge_cases.py

# Com coverage
pytest --cov=src tests/
```

**Testes incluem:**
- Testes unitários com mocks
- Testes de integração
- Testes de performance (startup time, memory usage)
- Testes de edge cases (rate limiting, corrupted files, etc)
- Métricas de qualidade (WER/CER)

---

## Documentação Técnica

- **[DOCKER_DEPLOYMENT.md](./SETUPs_certs_SSL_ModelsCache/DOCKER_DEPLOYMENT.md)** - Guia completo de deployment Docker (produção, desenvolvimento, testes)
- **[SSL_SETUP.md](./SETUPs_certs_SSL_ModelsCache/SSL_SETUP.md)** - Configuração HTTPS para desenvolvimento e produção
- **[pipeline_workflow.md](./pipeline_workflow.md)** - Diagrama detalhado do fluxo de processamento

---

## Estrutura do Projeto

```
transcrevai_windows/
├── src/                          # Código fonte principal
│   ├── transcription.py          # Serviço de transcrição (faster-whisper)
│   ├── diarization.py            # Serviço de diarização (pyannote)
│   ├── audio_processing.py       # Processamento de áudio e sessões
│   ├── pipeline.py               # Orquestração do pipeline completo
│   ├── dependencies.py           # Injeção de dependências
│   ├── exceptions.py             # Hierarquia de exceções customizadas
│   └── websocket_handler.py      # Validação WebSocket
├── tests/                        # Suite de testes (1630+ linhas)
│   ├── test_services.py          # Testes unitários
│   ├── test_performance.py       # Testes de performance
│   ├── test_edge_cases.py        # Casos extremos
│   ├── metrics.py                # WER/CER calculations
│   └── conftest.py               # Pytest fixtures
├── config/                       # Configuração da aplicação
│   └── app_config.py             # AppConfig com validação
├── static/                       # Frontend (JavaScript, CSS)
├── templates/                    # Templates HTML (Jinja2)
├── SETUPs_certs_SSL_ModelsCache/ # Scripts de setup
│   ├── download_models.py        # Download automático de modelos
│   ├── build.ps1                 # Build Docker (Windows)
│   ├── build-docker.sh           # Build Docker (Linux/Mac)
│   └── setup_dev_certs.bat       # Setup SSL desenvolvimento
├── Dockerfile                    # Imagem Docker otimizada (~17GB)
├── docker-compose.yml            # Build local
├── docker-compose.pull.yml       # Pull do Docker Hub
├── docker-compose.dev.yml        # Desenvolvimento (hot-reload)
├── pytest.ini                    # Configuração pytest
├── pyrightconfig.json            # Type checking
└── main.py                       # Entry point da aplicação
```

---

## Configuração via Variáveis de Ambiente

A aplicação suporta configuração via arquivo `.env` ou variáveis de ambiente:

```bash
# Servidor
TRANSCREVAI_HOST=0.0.0.0
TRANSCREVAI_PORT=8000

# SSL (opcional)
TRANSCREVAI_SSL_CERT=certs/localhost.pem
TRANSCREVAI_SSL_KEY=certs/localhost-key.pem

# Modelo
TRANSCREVAI_MODEL_NAME=medium
TRANSCREVAI_DEVICE=cpu
TRANSCREVAI_COMPUTE_TYPE=int8

# Diarização (fine-tuning)
TRANSCREVAI_DIARIZATION_THRESHOLD=0.335
TRANSCREVAI_DIARIZATION_MIN_CLUSTER_SIZE=12
TRANSCREVAI_DIARIZATION_MIN_SPEAKERS=
TRANSCREVAI_DIARIZATION_MAX_SPEAKERS=

# Performance
TRANSCREVAI_MAX_MEMORY=2.0
TRANSCREVAI_LOG_LEVEL=INFO
```

---

## Requisitos de Sistema

### Mínimo (funcionamento básico)
- **OS:** Windows 10/11 (64-bit), Linux, macOS
- **CPU:** 4+ cores (qualquer processador x86-64 moderno)
- **RAM:** 8GB (aplicação usa ~2GB em pico)
- **Storage:** 5GB disponível
- **Network:** Apenas para download inicial de modelos (se build local)

### Recomendado (melhor performance)
- **CPU:** 8+ cores
- **RAM:** 16GB
- **Storage:** SSD

**Nota:** A aplicação é 100% CPU-only. Não requer GPU.

---

## API Endpoints

### HTTP Endpoints

- **GET** `/` - Interface web principal
- **GET** `/health` - Health check endpoint
- **POST** `/upload` - Upload de arquivo de áudio (max 500MB, 60min)
- **GET** `/download-srt/{session_id}` - Download de arquivo SRT
- **GET** `/api/download/{session_id}/{file_type}` - Download genérico (audio/transcript/subtitles)

### WebSocket Endpoint

- **WS** `/ws/{session_id}` - Conexão para gravação ao vivo

**Rate Limiting:**
- HTTP endpoints: 10 requests/minuto por IP
- WebSocket: 20 conexões/minuto por IP

---
---

# TranscrevAI (English)

## Overview

TranscrevAI is a high-performance application for audio transcription and speaker diarization. It takes an audio input and provides a complete transcription, identifying who said what and when. It is designed to be a powerful tool for anyone who needs fast and accurate transcriptions of conversations, meetings, or recordings.

All transcription occurs locally on the machine where the server is running, without the use of any external APIs, ensuring data privacy. This offline architecture strongly aligns with the security and data minimization principles of the Brazilian General Data Protection Law (LGPD).

## Features

- **High-Performance Transcription:** Uses the faster-whisper model for fast, local transcription (CPU-optimized implementation of OpenAI's Whisper)
- **Speaker Diarization:** Identifies different speakers in the audio using pyannote.audio with word-level alignment algorithm
- **Live Recording:** Allows audio recording directly in the browser, with disk buffering to support long recordings without consuming excess RAM
- **File Upload:** Supports uploading pre-recorded audio files
- **SRT Subtitle Generation:** Creates subtitle files for videos
- **MP4 Video Generation:** Produces videos with embedded subtitles over a black background
- **Real-time Progress Updates:** WebSocket interface with user progress monitoring

## Tech Stack

- **Backend:** Python 3.11, FastAPI
- **AI/ML Models:**
    - **Transcription:** faster-whisper (Whisper medium optimized for CPU)
    - **Diarization:** pyannote.audio 3.1
- **Core ML Library:** PyTorch (CPU-only, INT8 quantization)
- **Real-time Communication:** WebSockets
- **Audio/Video Processing:** FFmpeg, librosa
- **Deployment:** Docker, Gunicorn/Uvicorn
- **SSL/HTTPS:** Full support for development and production

## Architecture

The application is built on FastAPI and follows a modern Dependency Injection (DI) based architecture, ensuring that components are modular, stable, and efficiently managed.

- **Modular Services:** Each core functionality is encapsulated in a service (TranscriptionService, PyannoteDiarizer, LiveAudioProcessor, SessionManager)
- **Session Management:** Complete lifecycle of each user session with automatic cleanup (24h timeout)
- **Asynchronous Processing:** Heavy tasks executed in separate worker threads to avoid blocking the main server
- **Disk Buffering:** Long recordings temporarily stored on disk, allowing low memory consumption
- **Adaptive Optimization:** Automatic hardware detection (CPU cores, RAM) and dynamic thread allocation

## Performance

**Achieved metrics:**
- **Startup time:** <30s with model pre-loading
- **Memory usage:** ~2GB peak (optimized for 8GB RAM systems)
- **Processing ratio:** ~1.5x realtime
- **Accuracy PT-BR:** 90%+ with post-processing linguistic corrections
- **Architecture:** CPU-only with INT8 quantization for universal compatibility

---

## Installation and Usage

### Option 1: Docker Hub (Recommended - Models Included)

**No Hugging Face token needed. Models already embedded in the image.**

```bash
# Pull image (first time, ~17GB)
docker pull hiperim/transcrevai:latest

# Run application
docker run -d -p 8000:8000 --name transcrevai hiperim/transcrevai:latest

# Access
# http://localhost:8000
```

**Using docker-compose:**
```bash
docker-compose -f docker-compose.pull.yml up -d
```

**Stop application:**
```bash
docker stop transcrevai
docker rm transcrevai
```

---

### Option 2: Local Build with Docker

**Requires Hugging Face token for model download.**

1. Clone repository:
   ```bash
   git clone <repository-url>
   cd transcrevai_windows
   ```

2. Create `.env` file with your token:
   ```bash
   HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

3. Build image:

   **Windows:**
   ```powershell
   .\SETUPs_certs_SSL_ModelsCache\build.ps1
   ```

   **Linux/Mac:**
   ```bash
   ./SETUPs_certs_SSL_ModelsCache/build-docker.sh
   ```

4. Run:
   ```bash
   docker-compose up -d
   ```

5. Access: `http://localhost:8000`

---

### Option 3: Local Installation (Development)

**Requires Python 3.11+, FFmpeg and Hugging Face token.**

1. Clone repository:
   ```bash
   git clone <repository-url>
   cd transcrevai_windows
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   # Production
   pip install -r requirements.txt

   # Development (includes pytest, etc)
   pip install -r requirements-dev.txt
   ```

4. Configure token in `.env`:
   ```bash
   HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

5. Download AI/ML models:
   ```bash
   python SETUPs_certs_SSL_ModelsCache/download_models.py
   ```

   This command downloads (~3-5GB):
   - faster-whisper-medium (transcription)
   - pyannote/speaker-diarization-3.1
   - pyannote/segmentation-3.0
   - pyannote/wespeaker embeddings

6. Run application:
   ```bash
   python main.py
   ```

7. Access: `http://localhost:8000`

---

## HTTPS Configuration (Optional)

The application supports HTTPS for development and production. HTTPS is required for the live recording functionality due to the browser's `getUserMedia()` API requirements.

### Development (localhost)

Run the automated script:
```batch
# Windows (as Administrator)
.\SETUPs_certs_SSL_ModelsCache\setup_dev_certs.bat
```

This script installs mkcert and generates trusted local certificates.

**Complete documentation:**
- [SSL_SETUP.md](./SETUPs_certs_SSL_ModelsCache/SSL_SETUP.md) - Complete HTTPS configuration guide

---

## Tests

The application includes a complete test suite:

```bash
# All tests
pytest

# Specific tests
pytest tests/test_services.py
pytest tests/test_performance.py
pytest tests/test_edge_cases.py

# With coverage
pytest --cov=src tests/
```

**Tests include:**
- Unit tests with mocks
- Integration tests
- Performance tests (startup time, memory usage)
- Edge case tests (rate limiting, corrupted files, etc)
- Quality metrics (WER/CER)

---

## Technical Documentation

- **[DOCKER_DEPLOYMENT.md](./SETUPs_certs_SSL_ModelsCache/DOCKER_DEPLOYMENT.md)** - Complete Docker deployment guide (production, development, tests)
- **[SSL_SETUP.md](./SETUPs_certs_SSL_ModelsCache/SSL_SETUP.md)** - HTTPS configuration for development and production
- **[pipeline_workflow.md](./pipeline_workflow.md)** - Detailed processing flow diagram

---

## Project Structure

```
transcrevai_windows/
├── src/                          # Main source code
│   ├── transcription.py          # Transcription service (faster-whisper)
│   ├── diarization.py            # Diarization service (pyannote)
│   ├── audio_processing.py       # Audio processing and sessions
│   ├── pipeline.py               # Complete pipeline orchestration
│   ├── dependencies.py           # Dependency injection
│   ├── exceptions.py             # Custom exception hierarchy
│   └── websocket_handler.py      # WebSocket validation
├── tests/                        # Test suite (1630+ lines)
│   ├── test_services.py          # Unit tests
│   ├── test_performance.py       # Performance tests
│   ├── test_edge_cases.py        # Edge cases
│   ├── metrics.py                # WER/CER calculations
│   └── conftest.py               # Pytest fixtures
├── config/                       # Application configuration
│   └── app_config.py             # AppConfig with validation
├── static/                       # Frontend (JavaScript, CSS)
├── templates/                    # HTML templates (Jinja2)
├── SETUPs_certs_SSL_ModelsCache/ # Setup scripts
│   ├── download_models.py        # Automatic model download
│   ├── build.ps1                 # Docker build (Windows)
│   ├── build-docker.sh           # Docker build (Linux/Mac)
│   └── setup_dev_certs.bat       # SSL setup for development
├── Dockerfile                    # Optimized Docker image (~17GB)
├── docker-compose.yml            # Local build
├── docker-compose.pull.yml       # Pull from Docker Hub
├── docker-compose.dev.yml        # Development (hot-reload)
├── pytest.ini                    # Pytest configuration
├── pyrightconfig.json            # Type checking
└── main.py                       # Application entry point
```

---

## Configuration via Environment Variables

The application supports configuration via `.env` file or environment variables:

```bash
# Server
TRANSCREVAI_HOST=0.0.0.0
TRANSCREVAI_PORT=8000

# SSL (optional)
TRANSCREVAI_SSL_CERT=certs/localhost.pem
TRANSCREVAI_SSL_KEY=certs/localhost-key.pem

# Model
TRANSCREVAI_MODEL_NAME=medium
TRANSCREVAI_DEVICE=cpu
TRANSCREVAI_COMPUTE_TYPE=int8

# Diarization (fine-tuning)
TRANSCREVAI_DIARIZATION_THRESHOLD=0.335
TRANSCREVAI_DIARIZATION_MIN_CLUSTER_SIZE=12
TRANSCREVAI_DIARIZATION_MIN_SPEAKERS=
TRANSCREVAI_DIARIZATION_MAX_SPEAKERS=

# Performance
TRANSCREVAI_MAX_MEMORY=2.0
TRANSCREVAI_LOG_LEVEL=INFO
```

---

## System Requirements

### Minimum (basic functionality)
- **OS:** Windows 10/11 (64-bit), Linux, macOS
- **CPU:** 4+ cores (any modern x86-64 processor)
- **RAM:** 8GB (application uses ~2GB at peak)
- **Storage:** 5GB available
- **Network:** Only for initial model download (if local build)

### Recommended (better performance)
- **CPU:** 8+ cores
- **RAM:** 16GB
- **Storage:** SSD

**Note:** The application is 100% CPU-only. No GPU required.

---

## API Endpoints

### HTTP Endpoints

- **GET** `/` - Main web interface
- **GET** `/health` - Health check endpoint
- **POST** `/upload` - Upload audio file (max 500MB, 60min)
- **GET** `/download-srt/{session_id}` - Download SRT file
- **GET** `/api/download/{session_id}/{file_type}` - Generic download (audio/transcript/subtitles)

### WebSocket Endpoint

- **WS** `/ws/{session_id}` - Connection for live recording

**Rate Limiting:**
- HTTP endpoints: 10 requests/minute per IP
- WebSocket: 20 connections/minute per IP
