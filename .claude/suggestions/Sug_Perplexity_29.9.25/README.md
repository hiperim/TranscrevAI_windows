# TranscrevAI Optimized - PT-BR Exclusive

Sistema de transcrição e diarização de áudio otimizado exclusivamente para português brasileiro, com arquitetura browser-safe e performance otimizada para hardware mínimo.

## 🎯 Características Principais

- **PT-BR Exclusivo**: Otimizado especificamente para português brasileiro
- **Browser-Safe**: Arquitetura que não trava navegadores web
- **Hardware Mínimo**: Otimizado para 4 cores, 8GB RAM
- **Performance**: 0.6-0.75x processing ratio (warm start), 1x-1.5x (cold start)
- **Diarização Profissional**: Detecção avançada de falantes com overlapping
- **Interface Moderna**: WebSocket com tracking de progresso em tempo real

## 🚀 Funcionalidades

### Core Features
- ✅ Gravação ao vivo (start/pause/resume/stop)
- ✅ Upload de múltiplos formatos de áudio
- ✅ Transcrição automática com Whisper medium PT-BR
- ✅ Diarização avançada com detecção de sobreposição
- ✅ Geração de legendas SRT precisas
- ✅ Interface web responsiva

### Optimizations
- ✅ Resource Manager com memory pressure detection
- ✅ Model Cache com lazy loading (warm starts 60-80% faster)
- ✅ Progressive Loading (prevent browser freezing)
- ✅ Adaptive Memory Cleanup (prevent crashes)
- ✅ Hardware-aware CPU optimization
- ✅ PT-BR specific contextual corrections

## 📋 Requisitos

### Hardware Mínimo
- **CPU**: 4 cores (Intel i5 / AMD Ryzen 5 equivalent)
- **RAM**: 8GB (target usage: ~2GB peak, ~1GB normal)
- **Disco**: 2GB livre para modelos e cache
- **Sistema**: Windows 10+, Ubuntu 20.04+, macOS 10.15+

### Software
- Python 3.8+
- FFmpeg (incluído via static-ffmpeg)

## 🔧 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/transcrevai-optimized.git
cd transcrevai-optimized
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute o aplicativo
```bash
python main.py
```

### 4. Acesse a interface
Abra seu navegador em: `http://localhost:8001`

## 🏗️ Arquitetura

```
src/
├── audio_processing.py      # Gravação e processamento de áudio
├── transcription.py         # Transcrição Whisper PT-BR otimizada
├── speaker_diarization.py   # Diarização avançada com overlapping
├── subtitle_generator.py    # Geração de SRT com timestamps precisos
├── resource_manager.py      # Gerenciamento de recursos e memória
├── model_cache.py          # Cache inteligente de modelos
├── memory_optimizer.py     # Otimização de memória e cleanup
├── hardware_optimizer.py   # Detecção e otimização de hardware
├── concurrent_engine.py    # Engine de processamento concorrente
├── file_manager.py         # Gerenciamento de arquivos
└── logging_setup.py        # Sistema de logging

templates/
└── index.html              # Interface web moderna

config/
├── config.py               # Configurações PT-BR otimizadas
└── compliance_rules.py     # Regras de compliance e qualidade
```

## ⚡ Performance Targets

### Processing Speed
- **Warm Start**: 0.6-0.75x processing ratio
- **Cold Start**: 1.0-1.5x processing ratio
- **Startup Time**: <5s (com progressive loading)
- **Memory Usage**: ~1GB normal, ~2GB peak

### Accuracy Targets
- **Transcrição**: >95% accuracy para PT-BR
- **Diarização**: >90% speaker detection accuracy
- **Timestamps**: ±100ms precision em SRT

## 🛡️ Compliance Framework

O sistema segue 20 regras rigorosas de compliance para garantir:
- Estabilidade em navegadores web
- Uso eficiente de recursos
- Qualidade consistente de output
- Recovery automático de erros

## 🔍 Debugging e Monitoramento

### Logs Estruturados
```bash
# Logs principais
tail -f logs/transcrevai.log

# Performance monitoring
tail -f logs/performance.log

# Resource monitoring
tail -f logs/resources.log
```

### Health Check
```bash
curl http://localhost:8001/health
```

## 🚀 Docker Support

```dockerfile
# Build
docker build -t transcrevai-optimized .

# Run
docker run -p 8001:8001 transcrevai-optimized
```

## 🎛️ Configuração Avançada

### Environment Variables
```bash
export TRANSCREVAI_PORT=8001
export TRANSCREVAI_HOST=0.0.0.0
export TRANSCREVAI_MAX_MEMORY_MB=2048
export TRANSCREVAI_CPU_CORES=4
export TRANSCREVAI_ENABLE_CACHE=true
export TRANSCREVAI_LOG_LEVEL=INFO
```

### Custom Configuration
Edite `config/config.py` para ajustes específicos:
- Thresholds de memória
- Parâmetros de transcrição
- Configurações de diarização
- Limites de performance

## 🧪 Testes

```bash
# Testes unitários
python -m pytest tests/

# Teste de performance
python tests/performance_test.py

# Teste de stress
python tests/stress_test.py
```

## 📈 Otimizações Implementadas

### Memory Management
- Resource Controller com thresholds automáticos
- Adaptive Memory Cleanup baseado em pressure
- Model caching com LRU eviction
- Emergency cleanup em situações críticas

### Performance Optimizations
- Lazy loading de modelos pesados
- Progressive initialization (browser-safe)
- CPU-aware threading
- Hardware-specific optimizations

### PT-BR Specific
- Correções contextuais específicas
- Parâmetros Whisper otimizados para português
- Dicionário de correções expandido
- Acoustic model fine-tuning

## 🐛 Troubleshooting

### Browser Travando
- Verifique se o Progressive Loading está ativo
- Monitore uso de memória (deve ficar <75%)
- Verifique logs de resource_manager

### Performance Baixa
- Confirme se model cache está funcionando
- Verifique disponibilidade de CPU cores
- Monitore memory pressure
- Execute teste de hardware: `python -m src.hardware_optimizer --test`

### Accuracy Baixa
- Verifique se correções PT-BR estão ativas
- Confirme configurações de modelo medium
- Verifique qualidade do áudio de entrada
- Teste com diferentes thresholds de confidence

## 🤝 Contribuição

1. Fork o projeto
2. Crie sua feature branch: `git checkout -b feature/nova-feature`
3. Commit suas mudanças: `git commit -am 'Add nova feature'`
4. Push para branch: `git push origin feature/nova-feature`
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🆘 Suporte

Para suporte, abra uma issue no GitHub ou entre em contato:
- **Issues**: https://github.com/seu-usuario/transcrevai-optimized/issues
- **Email**: seu-email@exemplo.com
- **Discord**: Servidor da Comunidade

---

**TranscrevAI Optimized** - Performance meets reliability for Brazilian Portuguese transcription.