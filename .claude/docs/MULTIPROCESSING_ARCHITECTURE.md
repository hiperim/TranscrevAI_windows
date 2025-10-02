# Arquitetura Multiprocessing CPU-only TranscrevAI

## Visão Geral

Esta documentação descreve a arquitetura multiprocessing completa para o TranscrevAI, otimizada para processamento CPU-only em sistemas Windows 10/11 com 4+ cores e 8GB RAM, mantendo uso de memória em 750MB e alcançando ratio de processamento de 0.4-0.6x.

## Arquitetura de Componentes

### 1. ProcessManager Central (`MultiProcessingTranscrevAI`)

**Responsabilidades:**
- Coordenação central de todos os processos
- Gerenciamento de ciclo de vida dos processos
- Integração com WebSocket existente
- Monitoramento de recursos do sistema

**Características principais:**
- `max_cores = psutil.cpu_count() - 2` (implementação explícita)
- Meta de 750MB uso total de memória
- Restart automático em caso de falha
- Afinidade CPU otimizada por processo

### 2. Processos Especializados

#### AudioCaptureProcess
- **Core alocado:** 1 (primeiro core)
- **Responsabilidades:**
  - Captura de áudio em tempo real
  - Suporte a WAV e MP4
  - Conversão FFmpeg quando necessário
  - Buffering inteligente para processamento contínuo

#### TranscriptionProcess (com INT8 Quantização)
- **Cores alocados:** `max_cores // 2`
- **Responsabilidades:**
  - Transcrição usando Whisper medium PT-BR
  - Quantização INT8 automática para redução de memória (75%)
  - Cache inteligente de modelos por idioma
  - Processamento paralelo de segmentos de áudio

#### DiarizationProcess
- **Cores alocados:** `max_cores // 2`
- **Responsabilidades:**
  - Separação de speakers com múltiplos algoritmos
  - Seleção adaptativa de método baseada em análise de áudio
  - Clustering, análise espectral e métodos híbridos

### 3. Infraestrutura de Comunicação

#### SharedMemoryManager
- **Buffers compartilhados:** Audio, Transcription, Diarization
- **Limite por buffer:** 50 itens (controle de memória)
- **Thread-safe:** Locks nomeados por buffer
- **Auto-limpeza:** Remoção automática de itens antigos

#### QueueManager
- **Filas especializadas:**
  - `audio_queue`: Comandos de gravação
  - `transcription_queue`: Solicitações de transcrição
  - `diarization_queue`: Solicitações de diarização
  - `websocket_queue`: Updates em tempo real
  - `control_queue`: Controle global
  - `status_queue`: Monitoramento de status

#### ProcessMonitor
- **Monitoramento contínuo:** Verifica saúde a cada 2 segundos
- **Restart automático:** Até 5 tentativas para audio, 3 para processamento
- **Métricas de recursos:** CPU e memória por processo
- **Detecção de zombies:** Identificação e recuperação automática

## Otimizações CPU Específicas

### 1. Quantização INT8 do Whisper

```python
class INT8QuantizedWhisper:
    def _apply_int8_quantization(self):
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv1d},
            dtype=torch.qint8,
            inplace=False
        )
```

**Benefícios:**
- Redução de 75% no uso de memória
- Manutenção de 95%+ da precisão
- Velocidade otimizada para CPU x86-64
- Fallback automático para FP32 se quantização falhar

### 2. Gerenciamento de Cores CPU

```python
core_allocation = {
    ProcessType.AUDIO_CAPTURE: 1,      # Core 0
    ProcessType.TRANSCRIPTION: max_cores // 2,  # Cores 2+
    ProcessType.DIARIZATION: max_cores // 2,    # Cores restantes
    ProcessType.WEBSOCKET: 1           # Core 1
}
```

### 3. Configurações de Threading

- **Torch threads:** `torch.set_num_threads(core_count)`
- **Interop threads:** `torch.set_num_interop_threads(1)`
- **Afinidade CPU:** Configuração automática por processo
- **Prioridade:** Audio em alta prioridade, outros em normal

## Targets de Performance

### Requisitos Mínimos do Sistema
- **OS:** Windows 10/11 (64-bit)
- **CPU:** 4+ cores (qualquer x86-64 moderno)
- **RAM:** 8GB (750MB de uso da aplicação)
- **Storage:** 5GB disponível
- **Network:** Conexão padrão de internet

### Metas de Performance
- **Startup:** <5s com modelos INT8 pré-carregados
- **Uso de memória:** 750MB pico total
- **Ratio de processamento:** 0.4-0.6x (24min áudio em 10-15min)
- **Utilização CPU:** `max_cores = psutil.cpu_count() - 2`
- **Precisão:** 95%+ transcrição PT-BR

## Resistência a Falhas

### 1. Isolamento de Processos
- **Crash containment:** Falha de um processo não afeta outros
- **Memory isolation:** Vazamentos contidos por processo
- **Resource limits:** Limites por processo previnem exaustão

### 2. Recuperação Automática
```python
def _handle_dead_process(self, process_type: ProcessType, info: ProcessInfo):
    if info.restart_count < self.restart_limits.get(process_type, 3):
        info.restart_count += 1
        # Automatic restart logic
        self.queue_manager.send_control_message({
            "action": "restart_process",
            "process_type": process_type.value
        })
```

### 3. Graceful Degradation
- **Fallback modes:** Modelo original se quantização falhar
- **Method switching:** Diarização simples se algoritmos avançados falharem
- **Error reporting:** Notificação clara de limitações ativas

## Integração WebSocket

### 1. Real-time Updates
```python
def _forward_to_websocket(self, message: Dict[str, Any]):
    source = message.get("source")
    if source == "audio_capture":
        self._handle_audio_websocket_message(msg_data)
    elif source == "transcription":
        self._handle_transcription_websocket_message(msg_data)
    elif source == "diarization":
        self._handle_diarization_websocket_message(msg_data)
```

### 2. Progress Tracking
- **Audio levels:** Updates de nível de áudio em tempo real
- **Processing progress:** Atualizações de 0-100% por componente
- **Error notifications:** Notificação imediata de erros
- **Completion events:** Resultados finais estruturados

## Compatibilidade Windows

### 1. Método Spawn
```python
if sys.platform.startswith('win'):
    mp.set_start_method('spawn', force=True)
```

### 2. Process Management
- **Windows process priorities:** `psutil.HIGH_PRIORITY_CLASS` para audio
- **CPU affinity:** Configuração específica por core
- **Memory limits:** Controle via `psutil.rlimit` quando disponível
- **FFmpeg integration:** Detecção e uso automático

## Uso e Integração

### 1. Integração com Sistema Existente

```python
# Substituir função de processamento existente
from src.multiprocessing_integration import enhanced_process_audio_concurrent_multiprocessing

# No main.py, substituir:
# enhanced_process_audio_concurrent(...)
# Por:
enhanced_process_audio_concurrent_multiprocessing(
    session_id, audio_file, language, audio_input_type,
    processing_profile, format_type, websocket_manager
)
```

### 2. Uso Standalone

```python
async with EnhancedTranscrevAIWithMultiprocessing() as transcrevai:
    # Processar arquivo
    result = await transcrevai.process_audio_file("audio.wav", language="pt")

    # Ou sessão ao vivo
    await transcrevai.start_recording_session("session_1", language="pt")
    # ... gravação ...
    await transcrevai.stop_recording_session("session_1")
```

### 3. Monitoramento

```python
status = transcrevai.get_system_status()
print(f"CPU: {status['system_resources']['cpu_percent']:.1f}%")
print(f"Memória: {status['system_resources']['memory_used_gb']:.1f}GB")
print(f"Processos ativos: {len(status['processes'])}")
```

## Estrutura de Arquivos

```
src/
├── multiprocessing_architecture.py    # Componentes base
├── audio_capture_process.py           # Processo de captura
├── transcription_process.py           # Processo de transcrição + INT8
├── diarization_process.py            # Processo de diarização
├── multiprocessing_manager.py        # Gerenciador central
└── multiprocessing_integration.py    # Integração com sistema existente
```

## Benefícios da Arquitetura

### 1. Performance
- **Paralelização real:** Processamento simultâneo de audio, transcrição e diarização
- **CPU optimization:** Uso eficiente de todos os cores disponíveis
- **Memory efficiency:** Quantização INT8 + buffers limitados
- **Cache intelligence:** Modelos reutilizados entre sessões

### 2. Estabilidade
- **Process isolation:** Falhas isoladas e contenção de vazamentos
- **Auto-recovery:** Restart automático sem intervenção manual
- **Resource monitoring:** Prevenção de exaustão de recursos
- **Graceful shutdown:** Terminação limpa de todos os componentes

### 3. Escalabilidade
- **Core-aware:** Adaptação automática ao hardware disponível
- **Memory-conscious:** Operação dentro de limites definidos
- **Session management:** Múltiplas sessões simultâneas
- **Load balancing:** Distribuição inteligente de trabalho

### 4. Integração
- **Backward compatible:** Integração transparente com sistema existente
- **WebSocket ready:** Updates em tempo real mantidos
- **API consistency:** Interface familiar para desenvolvedores
- **Configuration flexible:** Ajustes por perfil de uso

## Validação e Testes

### 1. Performance Benchmarks
- **Startup time:** Verificação de <5s inicialização
- **Memory usage:** Monitoramento contínuo de 750MB limite
- **Processing ratio:** Validação de 0.4-0.6x target
- **CPU utilization:** Confirmação de uso otimizado de cores

### 2. Stability Tests
- **Long-running:** Operação 24/7 sem vazamentos
- **Crash recovery:** Restart automático após falhas simuladas
- **Resource exhaustion:** Comportamento sob stress de memória/CPU
- **Network interruption:** Robustez a problemas de conectividade

### 3. Compatibility Validation
- **Windows versions:** Teste em 10/11 diferentes
- **Hardware variety:** CPUs Intel/AMD, diferentes contagens de core
- **Memory configurations:** 8GB, 16GB, 32GB sistemas
- **Storage types:** HDD, SSD, NVMe performance

Esta arquitetura garante que o TranscrevAI atinja máxima compatibilidade e performance em sistemas Windows CPU-only, mantendo estabilidade e facilidade de uso em produção.