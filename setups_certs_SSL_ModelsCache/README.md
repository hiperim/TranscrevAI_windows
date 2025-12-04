# Setup Scripts Directory

This directory contains scripts for building Docker images, downloading models, and setting up SSL certificates.

---

## Quick Start (Usuarios)

```bash
git clone https://github.com/hiperim/transcrevai.git
cd transcrevai
docker compose -f docker-compose.pull.yml up
```

Access: http://localhost:8000

- Image (~20GB) downloads automatically from Docker Hub
- All ML models embedded - no token needed
- Hardware auto-detected for optimal performance

---

## Para Desenvolvedores

### Docker Build Scripts (Multi-Architecture)

Scripts para build de imagens que funcionam em Intel/AMD (x86_64) e Apple Silicon (ARM64):

| Script | Plataforma |
|--------|------------|
| `build-multiarch.ps1` | Windows PowerShell |
| `build-multiarch.sh` | Linux/macOS |

**Requer:** Docker Desktop, token Hugging Face em `.env`

```bash
# Windows
.\build-multiarch.ps1

# Linux/Mac
chmod +x ./build-multiarch.sh
./build-multiarch.sh
```

### SSL Certificates (Desenvolvimento)

**`setup_dev_certs.bat`** - Gera certificados SSL locais para HTTPS

```batch
# Windows (como Administrador)
.\setup_dev_certs.bat
```

### Download de Modelos (Build Local)

**`download_models.py`** - Baixa modelos Whisper e Pyannote

```bash
# Requer HUGGING_FACE_HUB_TOKEN em .env
python download_models.py
```

---

## Documentacao

- **[DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md)** - Guia de deployment Docker
- **[SSL_SETUP.md](./SSL_SETUP.md)** - Configuracao HTTPS
- **[ARM_COMPATIBILITY.md](./ARM_COMPATIBILITY.md)** - Compatibilidade ARM/Apple Silicon

---

For questions or issues, see the main project [README.md](../README.md).
