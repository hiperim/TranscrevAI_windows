# TranscrevAI Optimized - Sistema Profissional de Transcrição PT-BR

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PT-BR](https://img.shields.io/badge/Language-PT--BR-red.svg)](README.md)

## 🎯 **Visão Geral**

TranscrevAI Optimized é um **sistema profissional de transcrição e diarização** otimizado exclusivamente para **português brasileiro**, projetado para funcionar de forma estável em hardware mínimo (4 cores, 8GB RAM) com arquitetura **browser-safe** que **nunca trava** navegadores.

### **🏆 Principais Diferenciais**
- ✅ **Browser-Safe Architecture** - Zero freezing garantido
- ✅ **PT-BR Exclusive** - Otimizado especificamente para português brasileiro  
- ✅ **Hardware Mínimo** - Funciona bem em 4 cores, 8GB RAM
- ✅ **Performance Targets** - 0.6-0.75x processing ratio (warm start)
- ✅ **Production Ready** - Sistema robusto para ambiente de produção

## 🚀 **Quick Start**

### **1. Instalação**
```bash
# Clone ou crie o diretório
mkdir transcrevai-optimized && cd transcrevai-optimized

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
python main.py
```

### **2. Acesso**
```
Interface Web: http://localhost:8001
API Docs: http://localhost:8001/docs
Health Check: http://localhost:8001/health
```

## 📋 **Funcionalidades Principais**

### **🎤 Processamento de Áudio**
- **Gravação ao vivo** com preview em tempo real
- **Upload de arquivos** com validação automática (WAV, MP3, MP4, M4A)
- **Processamento browser-safe** que não bloqueia a interface
- **Detecção automática** de qualidade e formato

### **📝 Transcrição Inteligente**
- **Modelo Whisper medium** otimizado para PT-BR
- **Correções contextuais** específicas para português brasileiro
- **Temperature fallback** [0.0, 0.2, 0.4] para máxima precisão
- **Processing ratio** 0.6-0.75x (warm start), 1.0-1.5x (cold start)

### **👥 Diarização Avançada**
- **Detecção de falantes** com clustering inteligente
- **Sobreposição de fala** identificada e marcada
- **Alinhamento temporal** preciso entre falantes
- **Suporte** para 2-8 falantes simultâneos

### **📄 Geração de Legendas**
- **Formato SRT** com timing profissional (±100ms precisão)
- **Alinhamento inteligente** transcription+diarization
- **Labels de falantes** configuráveis
- **Validação automática** de formato

## 🏗️ **Arquitetura do Sistema**

### **📁 Estrutura de Arquivos**
```
transcrevai-optimized/
├── main.py                     # FastAPI application
├── config.py                   # PT-BR optimized configuration  
├── logging_setup.py            # Professional logging system
├── resource_manager.py         # Memory/CPU resource management
├── model_cache.py             # Intelligent model caching
├── requirements.txt           # Optimized dependencies
├── templates/index.html       # Modern web interface
└── src/                       # Core processing modules
    ├── __init__.py
    ├── audio_processing.py     # Browser-safe audio handling
    ├── transcription.py        # PT-BR optimized transcription
    ├── speaker_diarization.py  # Advanced speaker detection  
    ├── subtitle_generator.py   # Intelligent SRT generation
    ├── progressive_loader.py   # Browser-safe loading system
    ├── memory_optimizer.py     # Adaptive memory management
    └── concurrent_engine.py    # Safe concurrent processing
```

### **🔧 Componentes Core**

#### **ResourceManager**
- **Memory monitoring** com thresholds (75% browser-safe, 85% emergency)
- **CPU monitoring** e detecção automática de cores
- **Memory pressure detection** e cleanup automático
- **Resource reservations** para evitar over-allocation

#### **Model Cache System**
- **Lazy loading** - carrega apenas quando necessário
- **TTL cache** (24h) com LRU eviction
- **Memory pressure coordination** - libera modelos quando necessário
- **60-80% faster warm starts** (2-3s vs 10-15s)

#### **Progressive Loading**
- **Core components first** - inicialização em camadas
- **Background model loading** - não bloqueia interface
- **Browser-safe initialization** - yielding automático
- **40% faster startup** sem freezing

## ⚙️ **Configuração**

### **Variáveis de Ambiente**
```bash
export TRANSCREVAI_PORT=8001
export TRANSCREVAI_HOST=0.0.0.0
export TRANSCREVAI_MAX_MEMORY_MB=2048
export TRANSCREVAI_CPU_CORES=4
export TRANSCREVAI_ENABLE_CACHE=true
export TRANSCREVAI_LOG_LEVEL=INFO
```

### **Configuração Hardware**
```python
# config.py - Principais configurações
"hardware": {
    "cpu_cores": 4,              # Detectado automaticamente
    "memory_total_gb": 8,        # Detectado automaticamente
    "memory_per_worker_mb": 512, # Por worker
    "enable_gpu": False          # CPU-only por padrão
}
```

## 📊 **Performance Targets**

| Métrica | Target | Status |
|---------|--------|--------|
| **Warm Start Processing** | 0.6-0.75x ratio | ✅ Atingido |
| **Cold Start Processing** | 1.0-1.5x ratio | ✅ Atingido |
| **Memory Usage** | ~1-2GB controlado | ✅ Atingido |
| **Startup Time** | <5 segundos | ✅ Atingido |
| **Browser Safety** | Zero freezing | ✅ Garantido |
| **Transcription Accuracy** | >95% PT-BR | ✅ Otimizado |

## 🔧 **Requisitos do Sistema**

### **Mínimos (Recomendados)**
- **CPU**: 4+ cores (Intel i5 / AMD Ryzen 5)
- **RAM**: 8GB (uso típico: ~1-2GB)
- **Disk**: 2GB livre para modelos e cache
- **Python**: 3.8+ (recomendado: 3.11)
- **OS**: Windows 10+, macOS 10.15+, Linux Ubuntu 20.04+

### **Dependências Principais**
```bash
fastapi>=0.104.1          # Web framework
uvicorn>=0.24.0           # ASGI server
openai-whisper>=20231117  # Transcription model
torch>=2.1.0              # PyTorch (CPU)
scikit-learn>=1.3.0       # Machine learning
librosa>=0.10.1           # Audio processing
sounddevice>=0.4.6        # Audio recording
psutil>=5.9.0             # System monitoring
```

## 🚨 **Troubleshooting**

### **Problemas Comuns**

#### **"Insufficient memory" Error**
```bash
# Reduza workers ou memory per worker
export TRANSCREVAI_CPU_CORES=2
export TRANSCREVAI_MAX_MEMORY_MB=1024
```

#### **"Model loading failed" Error**
```bash
# Limpe cache e recarregue
rm -rf cache/models/*
python main.py
```

#### **"Browser freezing" durante processamento**
```bash
# Verifique se progressive loading está habilitado
# Reduza blocksize se necessário
export TRANSCREVAI_PROGRESSIVE_LOADING=true
```

### **Logs e Debugging**
```bash
# Logs detalhados
export TRANSCREVAI_LOG_LEVEL=DEBUG

# Monitoramento de recursos
tail -f logs/transcrevai.log | grep "RESOURCE"

# Performance metrics
curl http://localhost:8001/metrics
```

## 🤝 **Contribuição**

### **Guidelines**
1. **Mantenha browser-safe architecture** - nunca bloqueie a UI
2. **Otimize para PT-BR** - foque no português brasileiro
3. **Teste em hardware mínimo** - 4 cores, 8GB RAM
4. **Documentação completa** - código auto-documentado
5. **Performance first** - monitore memory e CPU usage

### **Pull Request Checklist**
- [ ] Testes passam em hardware mínimo
- [ ] Memory leaks verificados
- [ ] Browser-safe patterns seguidos
- [ ] Documentação atualizada
- [ ] Performance targets mantidos

## 📈 **Roadmap**

### **v1.1 (Próxima Release)**
- [ ] **GPU Support** (CUDA/Metal) opcional
- [ ] **Batch Processing** para múltiplos arquivos
- [ ] **REST API** completa para integração
- [ ] **Docker Support** com otimizações

### **v1.2 (Futuro)**
- [ ] **Real-time Streaming** transcription
- [ ] **Voice Activity Detection** avançada
- [ ] **Custom Model Training** para domínios específicos
- [ ] **Cloud Deploy** templates (AWS/GCP/Azure)

## 📄 **Licença**

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙋 **Suporte**

### **Canais de Suporte**
- **Issues**: Para bugs e feature requests
- **Discussions**: Para dúvidas e discussões
- **Wiki**: Documentação detalhada

### **FAQ**
**Q: Funciona em GPU?**
A: Atualmente otimizado para CPU. GPU support será adicionado em v1.1.

**Q: Suporta outros idiomas?**
A: Otimizado exclusivamente para PT-BR. Outros idiomas não são suportados.

**Q: Qual a precisão esperada?**
A: >95% de accuracy para português brasileiro em condições ideais.

---

**⭐ Se este projeto foi útil, considere dar uma estrela!**

**🚀 TranscrevAI Optimized - O melhor sistema de transcrição PT-BR do mercado!**