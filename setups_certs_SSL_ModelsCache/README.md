# Setup Scripts Directory

This directory contains scripts for building Docker images, downloading models, and setting up SSL certificates.

## 📦 Docker Build Scripts

### Multi-Architecture Build (Recommended)

**Use these scripts to build images that work on both Intel/AMD (x86_64) and Apple Silicon (ARM64):**

- **`build-multiarch.ps1`** - Windows PowerShell script
- **`build-multiarch.sh`** - Linux/macOS bash script

**Features:**
- ✅ Builds for AMD64 (Intel/AMD) and ARM64 (Apple Silicon)
- ✅ Creates single Docker Hub tag that works on all architectures
- ✅ Automatically pushes to Docker Hub
- ✅ Requires Docker Desktop (includes buildx)

**Usage:**

```bash
# Windows
.\build-multiarch.ps1

# Linux/Mac
chmod +x ./build-multiarch.sh
./build-multiarch.sh
```

**See also:** [ARM_COMPATIBILITY.md](./ARM_COMPATIBILITY.md) for detailed guide.

---

## 🔐 SSL Certificate Setup

### Development SSL Certificates

**`setup_dev_certs.bat`** - Generates trusted local SSL certificates for HTTPS development

**Usage:**
```batch
# Windows (run as Administrator)
.\setup_dev_certs.bat
```

**See also:** [SSL_SETUP.md](./SSL_SETUP.md) for complete SSL configuration guide.

---

## 📥 Model Download

### Download ML Models

**`download_models.py`** - Downloads Whisper and Pyannote models to local cache

**Usage:**
```bash
# Make sure HUGGING_FACE_HUB_TOKEN is set in .env
python download_models.py
```

**What it downloads:**
- Whisper medium model (~1.5GB)
- Pyannote speaker diarization models (~2GB)
- Pyannote segmentation model (~300MB)
- Pyannote embedding model (~200MB)

**Total:** ~4GB

---

## 📚 Documentation

- **[ARM_COMPATIBILITY.md](./ARM_COMPATIBILITY.md)** - ARM/Apple Silicon support guide
- **[DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md)** - Complete Docker deployment guide
- **[SSL_SETUP.md](./SSL_SETUP.md)** - HTTPS/SSL configuration guide

---

## 🗑️ Deprecated Files

The following files are **no longer used** and kept only for reference:

- ~~`build.ps1`~~ - Replaced by `build-multiarch.ps1`
- ~~`build-docker.sh`~~ - Replaced by `build-multiarch.sh`

**Please use the multi-arch scripts instead for all new builds.**

---

## 🚀 Quick Start

**For first-time setup:**

1. **Download models:**
   ```bash
   python download_models.py
   ```

2. **Build multi-arch image:**
   ```bash
   # Windows
   .\build-multiarch.ps1

   # Linux/Mac
   ./build-multiarch.sh
   ```

3. **Run application:**
   ```bash
   docker pull hiperim/transcrevai:latest
   docker run -d -p 8000:8000 hiperim/transcrevai:latest
   ```

4. **Access:** http://localhost:8000

---

## ❓ Which Script Should I Use?

| Scenario | Script to Use |
|----------|---------------|
| Building for Docker Hub (production) | `build-multiarch.sh` or `build-multiarch.ps1` |
| Building for local testing only | `docker-compose build` (uses Dockerfile.multiarch) |
| Downloading models before build | `download_models.py` |
| Setting up HTTPS for development | `setup_dev_certs.bat` |
| Understanding ARM compatibility | Read `ARM_COMPATIBILITY.md` |
| Full deployment instructions | Read `DOCKER_DEPLOYMENT.md` |

---

## 💡 Tips

- **Multi-arch builds take longer** (~2x time) because they build for 2 architectures
- **Use `--no-cache`** flag if you're troubleshooting build issues
- **Docker Hub login required** for multi-arch builds (they auto-push)
- **ARM64 builds on x86_64** use QEMU emulation (slower but works)

---

For questions or issues, see the main project [README.md](../README.md).
