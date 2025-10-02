# SOLUÇÃO PICKLE ISSUE - TranscrevAI Multiprocessing

**Data:** 2025-09-30
**Issue:** `TypeError: cannot pickle 'weakref.ReferenceType' object`

---

## WEB RESEARCH SUMMARY (Triple Resume)

### Search 1: Weakref Error Fix
- **Causa:** Process objects contêm weakref.ref que não podem ser serializados
- **Solução:** Evitar passar Process objects ou Manager proxies como argumentos

### Search 2: Manager Queue Pickle Issue
- **Causa:** Manager().Queue() cria proxies com weakref callbacks
- **Solução:** Usar mp.Queue() direto ou criar Manager localmente no worker

### Search 3: Spawn Audio Processing
- **Causa:** 'spawn' method precisa serializar todos argumentos
- **Solução:** Passar dados primitivos, file paths, ou usar shared memory arrays

### CONSISTÊNCIA: 3/3 ✅
**Todas searches concordam:** NÃO passar Manager objects entre processos

---

## PROBLEMA IDENTIFICADO

**Localização:** `performance_optimizer.py:2050`

```python
# ANTES (ERRO)
process = mp.Process(
    target=worker_func,
    args=(os.getpid(), self.queue_manager, self.shared_memory, self.manual_mode),
    # ^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^
    # QueueManager e SharedMemoryManager contêm Manager() objects com weakref!
)
```

**Erro:**
```
ERROR - Erro iniciando processo transcription: cannot pickle 'weakref.ReferenceType' object
```

---

## SOLUÇÃO IMPLEMENTADA

### Correção 1: Audio Capture Worker ✅

**Problema:** `audio_capture_worker` não usa queue/shared_memory, mas recebe como arg

**Solução:**
```python
if process_type == ProcessType.AUDIO_CAPTURE:
    # audio_capture_worker só precisa de parent_pid
    process_args = (os.getpid(), self.manual_mode)
else:
    # outros workers
    process_args = (os.getpid(), self.queue_manager, self.shared_memory, self.manual_mode)
```

**Resultado:** ✅ Audio capture agora inicia sem erro

---

### Correção 2 (PENDENTE): Transcription/Diarization Workers

**Opções de solução:**

#### Opção A: Usar mp.Queue() Direto (Simples)
```python
class QueueManager:
    def __init__(self):
        # Usar mp.Queue() direto ao invés de Manager().Queue()
        self.audio_queue = mp.Queue(maxsize=10)
        self.transcription_queue = mp.Queue(maxsize=10)
        # ...
```
✅ **Vantagem:** mp.Queue() é picklable
❌ **Limitação:** Não pode ser compartilhado via Manager proxy

#### Opção B: Passar Apenas Queue Objects (Médio)
```python
# Passar queues individuais, não QueueManager object
process_args = (
    os.getpid(),
    self.queue_manager.transcription_queue,  # Queue direta
    self.shared_memory.shared_dict,  # DictProxy direta
    self.manual_mode
)
```
✅ **Vantagem:** Queues e DictProxy são picklable
⚠️ **Mudança:** Worker signatures precisam mudar

#### Opção C: Manager Server com Address (Complexo)
```python
# Criar Manager Server
manager = mp.Manager()
manager_address = manager.address

# No worker, conectar ao Manager
from multiprocessing.managers import BaseManager
BaseManager.register('get_queue')
m = BaseManager(address=manager_address, authkey=...)
m.connect()
```
✅ **Vantagem:** Arquitetura Manager correta
❌ **Desvantagem:** Requer refactor grande

---

## RECOMENDAÇÃO

### **Opção B (Híbrida):**
1. Manter QueueManager/SharedMemoryManager como wrappers
2. Passar objetos específicos (queues, dicts) diretamente
3. Modificar worker signatures para receber objetos diretos

**Implementação:**
```python
# performance_optimizer.py
def _start_process(self, process_type, worker_func):
    if process_type == ProcessType.TRANSCRIPTION:
        process_args = (
            os.getpid(),
            self.queue_manager.transcription_queue,
            self.shared_memory.shared_dict,
            self.manual_mode
        )
    elif process_type == ProcessType.DIARIZATION:
        process_args = (
            os.getpid(),
            self.queue_manager.diarization_queue,
            self.shared_memory.shared_dict,
            self.manual_mode
        )
```

```python
# Worker signature
def transcription_worker(parent_pid: int,
                        transcription_queue: mp.Queue,
                        shared_dict: DictProxy,
                        manual_mode: bool = True):
    # Usar queue diretamente
    cmd = transcription_queue.get()
    # ...
```

---

## PRÓXIMOS PASSOS

1. **Modificar worker signatures** (transcription_worker, diarization_worker)
2. **Atualizar _start_process** para passar objetos específicos
3. **Testar** que workers iniciam sem pickle error
4. **Validar** que comunicação entre processos funciona

---

## ARQUIVOS AFETADOS

- `src/performance_optimizer.py:2041-2076` (_start_process)
- `src/performance_optimizer.py:74-173` (diarization_worker)
- `src/performance_optimizer.py:185-282` (transcription_worker)

---

**STATUS ATUAL:**
- ✅ Audio capture: CORRIGIDO
- ⏳ Transcription: EM PROGRESSO
- ⏳ Diarization: EM PROGRESSO

**ESTIMATIVA:** 20-30 minutos para implementar Opção B completa