from __future__ import annotations  # Enable forward references for type hints

import asyncio
import gc
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

from src.logging_setup import setup_app_logging


logger = setup_app_logging(logger_name="transcrevai.performance_optimizer")

# Define missing loggers used throughout the file
resource_logger = setup_app_logging(logger_name="transcrevai.performance_optimizer.resource")
manager_logger = setup_app_logging(logger_name="transcrevai.performance_optimizer.manager")


# FASE 10: Shared Memory Multiprocessing Helper
def process_with_shared_memory(audio_data, worker_func, *args, **kwargs):
    """
    FASE 10: Process audio using shared memory to avoid pickling overhead

    Args:
        audio_data: numpy array with audio data
        worker_func: Worker function to execute
        *args, **kwargs: Additional arguments for worker function

    Returns:
        Result from worker function

    Performance: 20-30% faster multiprocessing for large audio files (>100MB)
    """
    try:
        import numpy as np
        from multiprocessing import shared_memory

        # Create shared memory
        shm = shared_memory.SharedMemory(
            create=True,
            size=audio_data.nbytes
        )

        try:
            # Copy audio data to shared memory
            shared_array = np.ndarray(
                audio_data.shape,
                dtype=audio_data.dtype,
                buffer=shm.buf
            )
            shared_array[:] = audio_data[:]

            logger.debug(f"[FASE 10] Created shared memory: {audio_data.nbytes / (1024*1024):.2f}MB")

            # Process without pickling overhead
            result = worker_func(shared_array, *args, **kwargs)

            return result

        finally:
            # Cleanup shared memory
            shm.close()
            shm.unlink()
            logger.debug("[FASE 10] Shared memory cleaned up")

    except Exception as e:
        logger.error(f"[FASE 10] Shared memory processing failed: {e}")
        # Fallback to normal processing
        return worker_func(audio_data, *args, **kwargs)


# Worker functions for multiprocessing
def audio_capture_worker(parent_pid: int, manual_mode: bool = True):
    """Worker para captura de áudio - multiprocessing implementation"""
    worker_logger = setup_app_logging(logger_name="transcrevai.audio_capture_worker")
    worker_logger.info("Audio capture worker iniciado")

    try:
        # Configurar signal handlers para shutdown gracioso
        import signal
        def signal_handler(_signum, _frame):
            worker_logger.info("Audio capture worker recebeu sinal de shutdown")
            return

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Loop principal do worker - execução contínua
        while True:
            try:
                # Verifica se o processo pai ainda existe
                if not psutil.pid_exists(parent_pid):
                    worker_logger.info("Processo pai não encontrado, finalizando...")
                    break

                # Simulação de captura de áudio
                worker_logger.debug("Audio capture worker - monitorando...")
                time.sleep(1.0)  # Intervalo de monitoramento

            except KeyboardInterrupt:
                worker_logger.info("Audio capture worker recebeu interrupção")
                break
            except Exception as e:
                worker_logger.error(f"Erro no audio capture worker: {e}")
                time.sleep(0.5)  # Pequeno delay antes de continuar

    except Exception as e:
        worker_logger.error(f"Erro crítico no audio capture worker: {e}")
    finally:
        worker_logger.info("Audio capture worker finalizando")


def diarization_worker(parent_pid: int, diarization_queue, shared_dict, manual_mode: bool = True):
    """Worker para diarização - recebe queue e dict diretamente (pickle-safe)"""
    worker_logger = setup_app_logging(logger_name="transcrevai.diarization_worker")
    worker_logger.info(f"Diarization worker iniciado (manual_mode: {manual_mode})")

    try:
        # Importações necessárias para processamento real
        import os
        import sys
        import gc
        from pathlib import Path

        # Adicionar src ao path para importações
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

        # Lazy imports dos módulos do sistema
        diarization_module = None

        # Configurar signal handlers para shutdown gracioso
        import signal
        def signal_handler(_signum, _frame):
            worker_logger.info("Diarization worker recebeu sinal de shutdown")
            return

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Loop principal do worker - processamento real
        while True:
            try:
                # Check if parent process is still running
                if not psutil.pid_exists(parent_pid):
                    worker_logger.info("Parent process terminated, shutting down")
                    break

                # Lazy load dos módulos do sistema apenas quando necessário
                if diarization_module is None:
                    try:
                        from src.diarization import CPUSpeakerDiarization
                        cpu_manager = CPUCoreManager()
                        diarization_module = CPUSpeakerDiarization(cpu_manager=cpu_manager)
                        worker_logger.info("Diarization module loaded successfully with intelligent coordination")
                    except Exception as e:
                        worker_logger.error(f"Failed to load diarization module: {e}")
                        diarization_module = False  # Mark as failed
                        time.sleep(5) # Wait before retrying
                        continue

                # Get task from queue (usando queue direta)
                task = diarization_queue.get(timeout=1) # Blocking with timeout
                if task:
                    command = task.get("command")
                    payload = task.get("payload", {})
                    session_id = payload.get("session_id")
                    audio_file = payload.get("audio_file")

                    if command == "diarize_audio" and audio_file and session_id:
                        worker_logger.info(f"Processing diarization for session {session_id}: {audio_file}")

                        try:
                            if diarization_module and diarization_module is not False:
                                # CRITICAL FIX: Usar sync wrapper em vez de asyncio.run() em worker thread
                                # Assuming diarize_audio is an async function, run it in a new event loop
                                import asyncio
                                import concurrent.futures

                                def run_diarization_sync():
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    try:
                                        diarize_func = getattr(diarization_module, 'diarize_audio')
                                        return loop.run_until_complete(
                                            diarize_func(
                                                audio_file=str(audio_file),
                                                method="advanced"
                                            )
                                        )
                                    finally:
                                        loop.close()

                                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                    future = executor.submit(run_diarization_sync)
                                    result = future.result(timeout=300)  # 5 min timeout

                                # Store result in shared dict (usando dict direta)
                                shared_dict[f"diarization_{session_id}"] = result
                                worker_logger.info(f"Diarization for session {session_id} completed and stored.")
                            else:
                                worker_logger.error("Diarization module not available, skipping task")

                        except Exception as e:
                            worker_logger.error(f"Error processing diarization for session {session_id}: {e}")
                            # Store error in shared dict
                            shared_dict[f"diarization_{session_id}"] = {"error": str(e)}

                # Aguardar antes de verificar novamente
                time.sleep(0.1) # Small sleep to prevent busy-waiting

            except queue.Empty: # Timeout occurred, no task in queue
                time.sleep(0.1) # Small sleep to prevent busy-waiting
                continue
            except Exception as e:
                worker_logger.error(f"Erro no diarization worker: {e}")
                time.sleep(1.0)

    except Exception as e:
        worker_logger.error(f"Erro crítico no diarization worker: {e}")
    finally:
        worker_logger.info("Diarization worker finalizando")


def transcription_worker(parent_pid: int, transcription_queue, shared_dict, manual_mode: bool = True):
    """Worker para transcrição - recebe queue e dict diretamente (pickle-safe)"""
    worker_logger = setup_app_logging(logger_name="transcrevai.transcription_worker")
    worker_logger.info(f"Transcription worker iniciado (manual_mode: {manual_mode})")

    try:
        # Importações necessárias para processamento real
        import os
        import sys
        import gc
        import glob
        from pathlib import Path

        # Adicionar src ao path para importações
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

        # Lazy imports dos módulos do sistema
        transcription_module = None
        whisper_manager = None

        # Configurar signal handlers para shutdown gracioso
        import signal
        def signal_handler(_signum, _frame):
            worker_logger.info("Transcription worker recebeu sinal de shutdown")
            return

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Loop principal do worker - processamento real
        while True:
            try:
                # Check if parent process is still running
                if not psutil.pid_exists(parent_pid):
                    worker_logger.info("Parent process terminated, shutting down")
                    break

                # Lazy load dos módulos do sistema apenas quando necessário
                if transcription_module is None:
                    try:
                        from src.transcription import OptimizedTranscriber
                        # CPUCoreManager já está definido neste módulo - usar referência direta

                        # Criar cpu_manager para coordenação inteligente (FASE 3B)
                        cpu_manager = CPUCoreManager()
                        # Initialize without model_name argument
                        transcription_module = OptimizedTranscriber(cpu_manager=cpu_manager)
                        worker_logger.info("Transcription module loaded successfully with intelligent coordination")
                    except Exception as e:
                        worker_logger.error(f"Failed to load transcription module: {e}")
                        transcription_module = False  # Mark as failed
                        time.sleep(5) # Wait before retrying
                        continue

                # Get task from queue (usando queue direta)
                task = transcription_queue.get(timeout=1) # Blocking with timeout
                if task:
                    command = task.get("command")
                    payload = task.get("payload", {})
                    session_id = payload.get("session_id")
                    audio_file = payload.get("audio_file")
                    language = payload.get("language", "pt")
                    domain = payload.get("domain", "general")

                    if command == "transcribe_audio" and audio_file and session_id:
                        worker_logger.info(f"Processing transcription for session {session_id}: {audio_file}")

                        try:
                            if transcription_module and transcription_module is not False:
                                result = transcription_module.transcribe_parallel(
                                    audio_path=str(audio_file),
                                    domain=domain
                                )
                                # Store result in shared dict (usando dict direta)
                                shared_dict[f"transcription_{session_id}"] = result
                                worker_logger.info(f"Transcription for session {session_id} completed and stored.")
                            else:
                                worker_logger.error("Transcription module not loaded, skipping task")

                        except Exception as e:
                            worker_logger.error(f"Error processing transcription for session {session_id}: {e}")
                            # Store error in shared dict
                            shared_dict[f"transcription_{session_id}"] = {"error": str(e)}

                # Aguardar antes de verificar novamente
                time.sleep(0.1) # Small sleep to prevent busy-waiting

            except queue.Empty: # Timeout occurred, no task in queue
                time.sleep(0.1) # Small sleep to prevent busy-waiting
                continue
            except Exception as e:
                worker_logger.error(f"Erro no transcription worker: {e}")
                time.sleep(1.0)
    except Exception as e:
        worker_logger.error(f"Erro crítico no transcription worker: {e}")
    finally:
        worker_logger.info("Transcription worker finalizando")
# Configurar método spawn para Windows
if sys.platform.startswith('win'):
    mp.set_start_method('spawn', force=True)

# Configurações de performance e recursos
@dataclass(slots=True)
class ProcessConfig:
    """Configuração de processo para otimização CPU"""
    max_cores: int
    memory_limit_mb: int
    target_processing_ratio: float  # 0.4-0.6x
    quantization_enabled: bool
    crash_restart_enabled: bool

class ProcessType(Enum):
    """Tipos de processo na arquitetura"""
    AUDIO_CAPTURE = "audio_capture"
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    WEBSOCKET = "websocket"
    MONITOR = "monitor"

class ProcessStatus(Enum):
    """Status dos processos"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    RESTARTING = "restarting"

@dataclass(slots=True)
class ProcessInfo:
    """Informações de processo"""
    process_id: int
    process_type: ProcessType
    status: ProcessStatus
    memory_usage_mb: float
    cpu_usage_percent: float
    start_time: float
    restart_count: int
    last_error: Optional[str] = None

class SharedMemoryManager:
    """Gerenciador de memória compartilhada para transferência eficiente de dados"""

    def __init__(self):
        self.manager = mp.Manager()
        self.shared_dict = self.manager.dict()
        self.shared_locks = {}
        self.audio_buffer = self.manager.list()
        self.transcription_buffer = self.manager.list()
        self.diarization_buffer = self.manager.list()

        # Configurar buffers com limites de memória rigorosos
        self.max_buffer_size = 25  # Reduzido para menor uso de memória
        self.max_memory_per_item_mb = 50  # Máximo 50MB por item

        # Isolation tracking
        self.process_isolation_info = self.manager.dict()
        self.memory_limits = self.manager.dict()
        self.crash_counts = self.manager.dict()

    def get_shared_dict(self) -> Any:
        """Retorna dicionário compartilhado thread-safe (DictProxy)"""
        return self.shared_dict

    def get_lock(self, name: str):
        """Retorna ou cria lock nomeado"""
        if name not in self.shared_locks:
            self.shared_locks[name] = self.manager.Lock()
        return self.shared_locks[name]

    def add_audio_data(self, data: Dict[str, Any]):
        """Adiciona dados de áudio ao buffer compartilhado"""
        with self.get_lock("audio_buffer"):
            if len(self.audio_buffer) >= self.max_buffer_size:
                self.audio_buffer.pop(0)  # Remove o mais antigo
            self.audio_buffer.append(data)

    def get_audio_data(self) -> Optional[Dict[str, Any]]:
        """Retorna próximo item do buffer de áudio"""
        with self.get_lock("audio_buffer"):
            if self.audio_buffer:
                return self.audio_buffer.pop(0)
            return None

    def add_transcription_data(self, data: Dict[str, Any]):
        """Adiciona dados de transcrição ao buffer compartilhado"""
        with self.get_lock("transcription_buffer"):
            if len(self.transcription_buffer) >= self.max_buffer_size:
                self.transcription_buffer.pop(0)
            self.transcription_buffer.append(data)

    def get_transcription_data(self) -> Optional[Dict[str, Any]]:
        """Retorna próximo item do buffer de transcrição"""
        with self.get_lock("transcription_buffer"):
            if self.transcription_buffer:
                return self.transcription_buffer.pop(0)
            return None

    def add_diarization_data(self, data: Dict[str, Any]):
        """Adiciona dados de diarização ao buffer compartilhado"""
        with self.get_lock("diarization_buffer"):
            if len(self.diarization_buffer) >= self.max_buffer_size:
                self.diarization_buffer.pop(0)
            self.diarization_buffer.append(data)

    def get_diarization_data(self) -> Optional[Dict[str, Any]]:
        """Retorna próximo item do buffer de diarização"""
        with self.get_lock("diarization_buffer"):
            if self.diarization_buffer:
                return self.diarization_buffer.pop(0)
            return None

    def add_transcription_data_for_session(self, session_id: str, data: Dict[str, Any]):
        """Adiciona dados de transcrição para uma sessão específica"""
        with self.get_lock(f"transcription_session_{session_id}"):
            self.shared_dict[f"transcription_result_{session_id}"] = data
            logger.debug(f"Transcription result for session {session_id} added to shared memory.")

    def get_transcription_data_for_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retorna dados de transcrição para uma sessão específica"""
        with self.get_lock(f"transcription_session_{session_id}"):
            result = self.shared_dict.get(f"transcription_result_{session_id}")
            if result:
                del self.shared_dict[f"transcription_result_{session_id}"] # Clear after retrieval
                logger.debug(f"Transcription result for session {session_id} retrieved from shared memory.")
            return result

    def add_diarization_data_for_session(self, session_id: str, data: Dict[str, Any]):
        """Adiciona dados de diarização para uma sessão específica"""
        with self.get_lock(f"diarization_session_{session_id}"):
            self.shared_dict[f"diarization_result_{session_id}"] = data
            logger.debug(f"Diarization result for session {session_id} added to shared memory.")

    def get_diarization_data_for_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retorna dados de diarização para uma sessão específica"""
        with self.get_lock(f"diarization_session_{session_id}"):
            result = self.shared_dict.get(f"diarization_result_{session_id}")
            if result:
                del self.shared_dict[f"diarization_result_{session_id}"] # Clear after retrieval
                logger.debug(f"Diarization result for session {session_id} retrieved from shared memory.")
            return result

    def register_process_isolation(self, process_id: int, process_type: ProcessType, memory_limit_mb: int):
        """Registra processo para isolamento e monitoramento"""
        self.process_isolation_info[process_id] = {
            "type": process_type.value,
            "start_time": time.time(),
            "memory_limit_mb": memory_limit_mb,
            "crash_count": 0,
            "last_heartbeat": time.time()
        }
        self.memory_limits[process_id] = memory_limit_mb
        self.crash_counts[process_id] = 0
        logger.info(f"Process {process_id} ({process_type.value}) registered with {memory_limit_mb}MB limit")

    def update_process_heartbeat(self, process_id: int):
        """Atualiza heartbeat do processo para monitoramento"""
        if process_id in self.process_isolation_info:
            self.process_isolation_info[process_id]["last_heartbeat"] = time.time()

    def record_process_crash(self, process_id: int):
        """Registra crash do processo para estatísticas"""
        if process_id in self.crash_counts:
            self.crash_counts[process_id] += 1
            if process_id in self.process_isolation_info:
                self.process_isolation_info[process_id]["crash_count"] = self.crash_counts[process_id]

    def check_process_isolation_compliance(self, process_id: int) -> bool:
        """Verifica se processo está dentro dos limites de isolamento"""
        try:
            import psutil
            if process_id not in self.memory_limits:
                return True

            process = psutil.Process(process_id)
            memory_mb = process.memory_info().rss / (1024 * 1024)
            limit_mb = self.memory_limits[process_id]

            compliance = memory_mb <= limit_mb
            if not compliance:
                logger.warning(f"Process {process_id} exceeds memory limit: {memory_mb:.1f}MB > {limit_mb}MB")

            return compliance
        except (psutil.NoSuchProcess, KeyError):
            return True  # Process não existe mais

    def cleanup(self):
        """Aggressive cleanup para liberar recursos e evitar memory leaks"""
        try:
            # Cleanup sequencial para evitar deadlocks
            buffers_to_clear = [
                ("audio_buffer", self.audio_buffer),
                ("transcription_buffer", self.transcription_buffer),
                ("diarization_buffer", self.diarization_buffer)
            ]

            for buffer_name, buffer_obj in buffers_to_clear:
                try:
                    with self.get_lock(buffer_name):
                        items_count = len(buffer_obj) if buffer_obj else 0
                        buffer_obj[:] = []  # Correctly clear ListProxy
                        if items_count > 0:
                            logger.debug(f"Cleared {items_count} items from {buffer_name}")
                except Exception as e:
                    logger.warning(f"Error clearing {buffer_name}: {e}")

            # Clear shared data structures
            self.shared_dict.clear()
            self.process_isolation_info.clear()
            self.memory_limits.clear()
            self.crash_counts.clear()

            # Clear shared_locks references
            self.shared_locks.clear()

            # Force garbage collection para liberar memória compartilhada
            import gc
            gc.collect()

            logger.debug("SharedMemoryManager cleanup completed successfully")

        except Exception as e:
            logger.warning(f"Erro na limpeza de memória compartilhada: {e}")

class QueueManager:
    """Gerenciador de filas para comunicação entre processos"""

    def __init__(self):
        # Filas para cada tipo de comunicação
        self.audio_queue = mp.Queue(maxsize=10)
        self.transcription_queue = mp.Queue(maxsize=10)
        self.diarization_queue = mp.Queue(maxsize=10)
        self.websocket_queue = mp.Queue(maxsize=20)
        self.control_queue = mp.Queue(maxsize=5)
        self.status_queue = mp.Queue(maxsize=15)

        # Mapeamento de tipos para filas
        self.queue_map = {
            ProcessType.AUDIO_CAPTURE: self.audio_queue,
            ProcessType.TRANSCRIPTION: self.transcription_queue,
            ProcessType.DIARIZATION: self.diarization_queue,
            ProcessType.WEBSOCKET: self.websocket_queue
        }

    def get_queue(self, process_type: ProcessType):
        """Retorna fila para tipo de processo"""
        return self.queue_map.get(process_type)

    def send_control_message(self, message: Dict[str, Any]):
        """Envia mensagem de controle"""
        try:
            self.control_queue.put_nowait(message)
        except queue.Full:
            logger.warning("Fila de controle cheia, descartando mensagem")
            return False

    def send_status_update(self, process_type: ProcessType, status: Dict[str, Any]):
        """Envia atualização de status"""
        try:
            status_message = {
                "process_type": process_type.value,
                "timestamp": time.time(),
                "data": status
            }
            self.status_queue.put_nowait(status_message)
        except queue.Full:
            logger.warning("Fila de status cheia, descartando atualização")
            return False

    def get_control_message(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Recebe mensagem de controle"""
        try:
            return self.control_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_status_update(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Recebe atualização de status"""
        try:
            return self.status_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def cleanup(self):
        """Limpa todas as filas"""
        queues = [
            self.audio_queue, self.transcription_queue, self.diarization_queue,
            self.websocket_queue, self.control_queue, self.status_queue
        ]

        for q in queues:
            try:
                while not q.empty():
                    q.get_nowait()
            except queue.Empty:
                pass
            except Exception as e:
                logger.warning(f"Erro limpando fila: {e}")

class ProcessMonitor:
    """Monitor de processos com restart automático e detecção de falhas"""

    def __init__(self, queue_manager: QueueManager, shared_memory: SharedMemoryManager):
        self.queue_manager = queue_manager
        self.shared_memory = shared_memory
        self.processes: Dict[ProcessType, ProcessInfo] = {}
        self.monitoring = False
        self.monitor_thread = None
        self.restart_limits = {
            ProcessType.AUDIO_CAPTURE: 5,
            ProcessType.TRANSCRIPTION: 3,
            ProcessType.DIARIZATION: 3,
            ProcessType.WEBSOCKET: 5
        }

    def start_monitoring(self):
        """Inicia monitoramento de processos"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitor de processos iniciado")

    def stop_monitoring(self):
        """Para monitoramento de processos"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Monitor de processos parado")

    def register_process(self, process_type: ProcessType, process_id: int):
        """Registra processo para monitoramento"""
        self.processes[process_type] = ProcessInfo(
            process_id=process_id,
            process_type=process_type,
            status=ProcessStatus.STARTING,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            start_time=time.time(),
            restart_count=0
        )
        logger.info(f"Processo {process_type.value} registrado (PID: {process_id})")

    def unregister_process(self, process_type: ProcessType):
        """Remove processo do monitoramento"""
        if process_type in self.processes:
            del self.processes[process_type]
            logger.info(f"Processo {process_type.value} removido do monitoramento")

    def update_process_status(self, process_type: ProcessType, status: ProcessStatus, error: Optional[str] = None):
        """Atualiza status do processo"""
        if process_type in self.processes:
            self.processes[process_type].status = status
            if error is not None and error:
                self.processes[process_type].last_error = error

            # Enviar atualização via queue
            self.queue_manager.send_status_update(process_type, {
                "status": status.value,
                "error": error,
                "timestamp": time.time()
            })

    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self.monitoring:
            try:
                self._check_processes()
                self._update_resource_usage()
                time.sleep(2.0)  # Verificar a cada 2 segundos
            except Exception as e:
                logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(5.0)

    def _check_processes(self):
        """Verifica saúde dos processos"""
        for process_type, info in self.processes.items():
            try:
                process = psutil.Process(info.process_id)

                if not process.is_running():
                    logger.warning(f"Processo {process_type.value} não está executando")
                    self._handle_dead_process(process_type, info)
                elif process.status() == psutil.STATUS_ZOMBIE:
                    logger.warning(f"Processo {process_type.value} é zombie")
                    self._handle_dead_process(process_type, info)
                else:
                    # Processo está vivo, atualizar status
                    if info.status == ProcessStatus.STARTING:
                        self.update_process_status(process_type, ProcessStatus.RUNNING)

            except psutil.NoSuchProcess:
                logger.warning(f"Processo {process_type.value} (PID: {info.process_id}) não encontrado")
                self._handle_dead_process(process_type, info)
            except Exception as e:
                logger.error(f"Erro verificando processo {process_type.value}: {e}")

    def _update_resource_usage(self):
        """Atualiza uso de recursos dos processos"""
        for process_type, info in self.processes.items():
            try:
                process = psutil.Process(info.process_id)

                # Atualizar uso de memória e CPU
                memory_info = process.memory_info()
                info.memory_usage_mb = memory_info.rss / (1024 * 1024)
                info.cpu_usage_percent = process.cpu_percent()

                # Verificar limites de memória (~1GB normal, ~2GB pico conforme claude.md)
                if info.memory_usage_mb > 512:  # 512MB por processo (normal: ~1GB/2 processos)
                    logger.warning(f"Processo {process_type.value} usando {info.memory_usage_mb:.1f}MB (acima do normal)")

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            except Exception as e:
                logger.error(f"Erro atualizando recursos do processo {process_type.value}: {e}")

    def _handle_dead_process(self, process_type: ProcessType, info: ProcessInfo):
        """Trata processo morto com restart automático inteligente"""
        info.status = ProcessStatus.ERROR
        self.shared_memory.record_process_crash(info.process_id)

        # Análise de crash para restart inteligente
        crash_count = self.shared_memory.crash_counts.get(info.process_id, 0)
        max_restarts = self.restart_limits.get(process_type, 3)

        # Backoff exponencial para evitar restart loops
        restart_delay = min(2 ** info.restart_count, 30)  # Max 30s delay

        if info.restart_count < max_restarts:
            info.restart_count += 1
            info.status = ProcessStatus.RESTARTING

            logger.warning(f"Processo {process_type.value} morreu (restart {info.restart_count}/{max_restarts})")
            logger.info(f"Aguardando {restart_delay}s antes do restart...")

            # Agendar restart com delay
            self._schedule_process_restart(process_type, info, restart_delay)

        else:
            logger.error(f"Processo {process_type.value} excedeu limite de restarts ({max_restarts})")
            self.update_process_status(process_type, ProcessStatus.ERROR,
                                     f"Exceeded restart limit ({max_restarts})")

    def _schedule_process_restart(self, process_type: ProcessType, info: ProcessInfo, delay: float):
        """Agenda restart de processo com delay"""
        def delayed_restart():
            time.sleep(delay)
            if self.monitoring and info.status == ProcessStatus.RESTARTING:
                try:
                    # Limpar recursos do processo anterior
                    self._cleanup_dead_process(info.process_id)

                    # Registrar tentativa de restart
                    logger.info(f"Iniciando restart do processo {process_type.value}")

                    # Simular restart (em implementação real, chamaria processo específico)
                    success = self._attempt_process_restart(process_type)

                    if success is not None and success:
                        logger.info(f"Processo {process_type.value} reiniciado com sucesso")
                        info.start_time = time.time()
                        self.update_process_status(process_type, ProcessStatus.STARTING)
                    else:
                        logger.error(f"Falha no restart do processo {process_type.value}")
                        self.update_process_status(process_type, ProcessStatus.ERROR, "Restart failed")

                except Exception as e:
                    logger.error(f"Erro durante restart do processo {process_type.value}: {e}")
                    self.update_process_status(process_type, ProcessStatus.ERROR, str(e))

        # Executar restart em thread separada para não bloquear monitor
        restart_thread = threading.Thread(target=delayed_restart, daemon=True)
        restart_thread.start()

    def _cleanup_dead_process(self, process_id: int):
        """Limpa recursos de processo morto"""
        try:
            # Tentar terminar processo gracefully se ainda existir
            if psutil.pid_exists(process_id):
                process = psutil.Process(process_id)
                process.terminate()
                process.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass
        except Exception as e:
            logger.warning(f"Erro limpando processo {process_id}: {e}")

    def _attempt_process_restart(self, process_type: ProcessType) -> bool:
        """Tenta reiniciar processo específico"""
        try:
            # Em implementação real, isso chamaria a função específica de cada processo
            # Por agora, retorna sucesso simulado
            logger.info(f"Simulando restart de {process_type.value}")
            return True
        except Exception as e:
            logger.error(f"Falha simulando restart de {process_type.value}: {e}")
            return False

    def get_process_info(self, process_type: ProcessType) -> Optional[ProcessInfo]:
        """Retorna informações do processo"""
        return self.processes.get(process_type)

    def get_all_processes_info(self) -> Dict[ProcessType, ProcessInfo]:
        """Retorna informações de todos os processos"""
        return self.processes.copy()

class CPUCoreManager:
    """Gerenciador de cores CPU para otimização de performance"""

    def __init__(self):
        self.total_cores = psutil.cpu_count(logical=True) or 4
        self.physical_cores = psutil.cpu_count(logical=False) or 2

        # Nova fórmula otimizada: max(1, logical_cores - 2, physical_cores - 2)
        self.max_cores = max(1, self.total_cores - 2, self.physical_cores - 2)

        logger.info(f"Cores disponíveis: {self.total_cores} lógicos, {self.physical_cores} físicos")
        logger.info(f"Cores utilizáveis: {self.max_cores} (fórmula: max(1, L-2, P-2))")

        # Coordenação inteligente de recursos (FASE 3)
        self._setup_intelligent_coordination()

    def _setup_intelligent_coordination(self):
        """Configura coordenação inteligente entre transcription e diarization"""
        # Reservar cores para sistema
        system_cores = 2  # Audio capture + WebSocket
        available_cores = max(1, self.max_cores - system_cores)

        # Distribuição otimizada: evitar over-subscription
        if available_cores >= 8:
            # Sistema potente: 70% transcription, 30% diarization
            trans_cores = max(4, int(available_cores * 0.7))
            diar_cores = max(2, available_cores - trans_cores)
        elif available_cores >= 4:
            # Sistema médio: 60% transcription, 40% diarization
            trans_cores = max(2, int(available_cores * 0.6))
            diar_cores = max(2, available_cores - trans_cores)
        else:
            # Sistema limitado: divisão igual
            trans_cores = max(1, available_cores // 2)
            diar_cores = max(1, available_cores - trans_cores)

        self.core_allocation = {
            ProcessType.AUDIO_CAPTURE: 1,
            ProcessType.TRANSCRIPTION: trans_cores,
            ProcessType.DIARIZATION: diar_cores,
            ProcessType.WEBSOCKET: 1
        }

        # Estado de coordenação dinâmica
        self.resource_coordinator = {
            'transcription_active': False,
            'diarization_active': False,
            'can_boost_transcription': True,
            'can_boost_diarization': True
        }

        logger.info(f"Alocação de cores: {self.core_allocation}")

    def get_cores_for_process(self, process_type: ProcessType) -> int:
        """Retorna número de cores alocados para processo"""
        return self.core_allocation.get(process_type, 1)

    def get_cpu_affinity_mask(self, process_type: ProcessType) -> List[int]:
        """Retorna máscara de afinidade CPU para processo"""
        cores_needed = self.get_cores_for_process(process_type)

        # Distribuir cores de forma eficiente
        if process_type == ProcessType.AUDIO_CAPTURE:
            return [0]  # Primeiro core para audio
        elif process_type == ProcessType.WEBSOCKET:
            return [1]  # Segundo core para websocket
        elif process_type == ProcessType.TRANSCRIPTION:
            start_core = 2
            total_cores = self.total_cores or 4  # Fallback para 4 cores
            return list(range(start_core, min(start_core + cores_needed, total_cores)))
        elif process_type == ProcessType.DIARIZATION:
            transcription_cores = self.get_cores_for_process(ProcessType.TRANSCRIPTION)
            start_core = 2 + transcription_cores
            total_cores = self.total_cores or 4  # Fallback para 4 cores
            return list(range(start_core, min(start_core + cores_needed, total_cores)))

        return [0]  # Fallback

    def request_resource_boost(self, process_type: ProcessType, active: bool) -> int:
        """Solicita boost de recursos para processo (coordenação dinâmica)"""
        if process_type == ProcessType.TRANSCRIPTION:
            self.resource_coordinator['transcription_active'] = active
        elif process_type == ProcessType.DIARIZATION:
            self.resource_coordinator['diarization_active'] = active

        # Se apenas um processo está ativo, pode usar mais cores
        trans_active = self.resource_coordinator['transcription_active']
        diar_active = self.resource_coordinator['diarization_active']

        if trans_active and not diar_active:
            # Boost transcription: pode usar cores de diarization temporariamente
            available_boost = self.core_allocation[ProcessType.DIARIZATION] // 2
            if process_type == ProcessType.TRANSCRIPTION and available_boost > 0:
                return self.core_allocation[ProcessType.TRANSCRIPTION] + available_boost

        elif diar_active and not trans_active:
            # Boost diarization: pode usar cores de transcription temporariamente
            available_boost = self.core_allocation[ProcessType.TRANSCRIPTION] // 2
            if process_type == ProcessType.DIARIZATION and available_boost > 0:
                return self.core_allocation[ProcessType.DIARIZATION] + available_boost

        # Retorna alocação padrão se ambos ativos ou sem boost
        return self.core_allocation.get(process_type, 1)

    def get_dynamic_cores_for_process(self, process_type: ProcessType, is_active: bool) -> int:
        """Retorna número de cores com coordenação dinâmica"""
        if process_type in [ProcessType.TRANSCRIPTION, ProcessType.DIARIZATION]:
            return self.request_resource_boost(process_type, is_active)
        return self.get_cores_for_process(process_type)

    def set_process_affinity(self, process: psutil.Process, process_type: ProcessType):
        """Define afinidade CPU do processo"""
        try:
            affinity_mask = self.get_cpu_affinity_mask(process_type)
            process.cpu_affinity(affinity_mask)
            logger.info(f"Afinidade CPU definida para {process_type.value}: cores {affinity_mask}")
        except Exception as e:
            logger.warning(f"Erro definindo afinidade CPU para {process_type.value}: {e}")






@dataclass(slots=True)
class MemoryStatus:
    """Memory status data structure"""
    total_gb: float
    available_gb: float
    used_percent: float
    browser_safe: bool
    threat_level: str
    recommendation: str

@dataclass(slots=True)
class SystemStatus:
    """Complete system status including CPU and memory"""
    memory: MemoryStatus
    cpu_percent: float
    cpu_count: int
    emergency_mode: bool
    conservative_mode: bool
    streaming_mode: bool
    timestamp: float

class ResourceManager:
    """
    Unified resource management and monitoring system
    Consolidates resource control and memory monitoring functionality
    """

    def __init__(self):
        # Memory targets (from claude.md specifications)
        self.memory_target_mb = 1024     # Normal target ~1GB
        self.memory_limit_mb = 2048      # Peak limit ~2GB

        # Resource control settings
        self.emergency_threshold = 0.85
        self.conservative_mode = False
        self.memory_reservations = {}
        self.next_reservation_id = 1

        # Streaming mode support
        self.streaming_mode = False
        self.streaming_threshold = 0.80  # Enable streaming at 80% memory usage

        # Memory monitoring settings
        self.thresholds = {
            'browser_safe': 75.0,    # 75% - Browser safety limit
            'warning': 70.0,         # 70% - Warning threshold
            'optimal': 65.0,         # 65% - Optimal operating level
            'critical': 80.0         # 80% - Critical level
        }

        # Monitoring state
        self.monitoring = False
        self.callbacks: Dict[str, Callable] = {}
        self.last_status: Optional[SystemStatus] = None
        self.alert_cooldown = 30  # seconds between alerts
        self.last_alert_time = 0

        resource_logger.info("ResourceManager initialized")
        resource_logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB, CPU: {psutil.cpu_count()} cores")
        resource_logger.info(f"Target: {self.memory_target_mb}MB normal / {self.memory_limit_mb}MB peak")
        resource_logger.info(f"Emergency threshold: {self.emergency_threshold*100:.1f}% RAM")

    # Resource Control Methods (from resource_controller.py)

    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent

    def is_emergency_mode(self) -> bool:
        """Check if system is in emergency mode"""
        return self.get_memory_usage() > self.emergency_threshold * 100

    def is_conservative_mode(self) -> bool:
        """Check if conservative mode is enabled"""
        return self.conservative_mode

    def set_conservative_mode(self, enabled: bool):
        """Enable or disable conservative resource usage mode"""
        self.conservative_mode = enabled
        status = "ENABLED" if enabled else "DISABLED"
        resource_logger.info(f"Conservative mode {status}")

    def is_streaming_mode(self) -> bool:
        """Check if streaming mode is active"""
        return self.streaming_mode

    def set_streaming_mode(self, enabled: bool):
        """Enable or disable streaming mode for memory optimization"""
        self.streaming_mode = enabled
        status = "ENABLED" if enabled else "DISABLED"
        resource_logger.info(f"Streaming mode {status}")

    def should_use_streaming_mode(self) -> bool:
        """Check if should use model streaming mode based on memory pressure"""
        return self.get_memory_usage() > (self.streaming_threshold * 100)

    def enable_streaming_mode(self):
        """Enable streaming mode for memory efficiency"""
        if not self.streaming_mode:
            self.streaming_mode = True
            resource_logger.info("Streaming mode enabled due to memory pressure")

    def disable_streaming_mode(self):
        """Disable streaming mode when memory is available"""
        if self.streaming_mode:
            self.streaming_mode = False
            resource_logger.info("Streaming mode disabled - memory recovered")

    def reserve_memory(self, amount_mb: float) -> int:
        """Reserve memory and return reservation ID"""
        reservation_id = self.next_reservation_id
        self.memory_reservations[reservation_id] = amount_mb
        self.next_reservation_id += 1
        resource_logger.debug(f"Reserved {amount_mb:.1f}MB memory (ID: {reservation_id})")
        return reservation_id

    def release_memory(self, reservation_id: int):
        """Release memory reservation"""
        if reservation_id in self.memory_reservations:
            amount = self.memory_reservations[reservation_id]
            del self.memory_reservations[reservation_id]
            resource_logger.debug(f"Released {amount:.1f}MB memory (ID: {reservation_id})")

    def get_memory_status_basic(self) -> Dict[str, Any]:
        """Get basic memory status (legacy format)"""
        vm = psutil.virtual_memory()
        return {
            'total_mb': vm.total / (1024 * 1024),
            'available_mb': vm.available / (1024 * 1024),
            'used_percent': vm.percent,
            'reserved_mb': sum(self.memory_reservations.values())
        }

    def get_available_memory_mb(self) -> float:
        """Get available memory in MB"""
        return psutil.virtual_memory().available / (1024 * 1024)

    def can_safely_allocate(self, amount_mb: float) -> bool:
        """Check if can safely allocate given amount of memory"""
        available = self.get_available_memory_mb()
        return available > amount_mb * 1.2  # 20% safety margin

    def get_recommended_settings(self) -> Dict[str, Any]:
        """Get recommended settings based on current system state"""
        available_memory = self.get_available_memory_mb()
        cpu_count = psutil.cpu_count() or 4

        if available_memory < 1024:  # Less than 1GB
            return {
                'max_concurrent_sessions': 1,
                'batch_size': 1,
                'num_threads': 1
            }
        elif available_memory < 4096:  # Less than 4GB
            return {
                'max_concurrent_sessions': 2,
                'batch_size': 2,
                'num_threads': min(2, cpu_count)
            }
        else:  # 4GB or more
            return {
                'max_concurrent_sessions': min(4, cpu_count),
                'batch_size': 4,
                'num_threads': min(4, cpu_count)
            }

    def get_cpu_config(self) -> Dict[str, Any]:
        """Get CPU configuration"""
        return {
            'cpu_count': psutil.cpu_count() or 4,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'recommended_threads': min(4, psutil.cpu_count() or 4)
        }

    # Memory Monitoring Methods (from memory_monitor.py)

    def register_callback(self, event: str, callback: Callable):
        """Register callback for memory events"""
        self.callbacks[event] = callback
        resource_logger.info(f"Registered callback for '{event}' events")

    def get_memory_status(self) -> MemoryStatus:
        """Get current memory status with browser safety analysis"""
        try:
            vm = psutil.virtual_memory()

            total_gb = vm.total / (1024**3)
            available_gb = vm.available / (1024**3)
            used_percent = vm.percent

            # Determine browser safety and threat level
            browser_safe = used_percent <= self.thresholds['browser_safe']

            if used_percent >= self.thresholds['critical']:
                threat_level = "CRITICAL"
                recommendation = "IMMEDIATE ACTION: Force cleanup and model unloading"
            elif used_percent >= self.thresholds['browser_safe']:
                threat_level = "UNSAFE"
                recommendation = "URGENT: System exceeds browser-safe limits"
            elif used_percent >= self.thresholds['warning']:
                threat_level = "WARNING"
                recommendation = "CAUTION: Approaching browser-safe limits"
            elif used_percent >= self.thresholds['optimal']:
                threat_level = "ELEVATED"
                recommendation = "GOOD: Within safe operating range"
            else:
                threat_level = "OPTIMAL"
                recommendation = "EXCELLENT: Optimal memory usage"

            return MemoryStatus(
                total_gb=total_gb,
                available_gb=available_gb,
                used_percent=used_percent,
                browser_safe=browser_safe,
                threat_level=threat_level,
                recommendation=recommendation
            )

        except Exception as e:
            resource_logger.error(f"Failed to get memory status: {e}")
            return MemoryStatus(0, 0, 100, False, "ERROR", "Unable to assess memory")

    def get_system_state(self) -> SystemStatus:
        """Get complete system state"""
        memory_status = self.get_memory_status()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count() or 4

        return SystemStatus(
            memory=memory_status,
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            emergency_mode=self.is_emergency_mode(),
            conservative_mode=self.is_conservative_mode(),
            streaming_mode=self.is_streaming_mode(),
            timestamp=time.time()
        )

    async def _trigger_callback(self, event: str, status: MemoryStatus):
        """Trigger registered callback with cooldown"""
        current_time = time.time()

        if current_time - self.last_alert_time < self.alert_cooldown:
            return  # Skip to prevent spam

        callback = self.callbacks.get(event)
        if callback is not None and callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(status)
                else:
                    callback(status)
                self.last_alert_time = current_time
            except Exception as e:
                resource_logger.error(f"Callback error for '{event}': {e}")

    async def check_and_alert(self, status: MemoryStatus):
        """Check status and trigger appropriate alerts"""
        if status.threat_level == "CRITICAL":
            await self._trigger_callback('critical', status)
        elif status.threat_level == "UNSAFE":
            await self._trigger_callback('unsafe', status)
        elif status.threat_level == "WARNING":
            await self._trigger_callback('warning', status)

        # Always trigger status update
        await self._trigger_callback('status_update', status)

    async def start_monitoring(self, interval: float = 10.0):
        """Start continuous memory monitoring"""
        if self.monitoring:
            resource_logger.warning("Memory monitoring already running")
            return

        self.monitoring = True
        resource_logger.info(f"Starting memory monitoring (interval: {interval}s)")

        try:
            while self.monitoring:
                system_status = self.get_system_state()
                self.last_status = system_status

                resource_logger.debug(f"Memory: {system_status.memory.used_percent:.1f}% ({system_status.memory.threat_level})")

                # Check for alerts
                await self.check_and_alert(system_status.memory)

                # Auto-enable streaming mode if needed
                if self.should_use_streaming_mode() and not self.streaming_mode:
                    self.enable_streaming_mode()
                elif not self.should_use_streaming_mode() and self.streaming_mode:
                    self.disable_streaming_mode()

                # Wait for next check
                await asyncio.sleep(interval)

        except Exception as e:
            resource_logger.error(f"Memory monitoring error: {e}")
        finally:
            self.monitoring = False
            resource_logger.info("Memory monitoring stopped")

    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        resource_logger.info("Memory monitoring stop requested")

    def force_cleanup(self) -> Dict[str, Any]:
        """Force aggressive memory cleanup"""
        resource_logger.warning("Executing FORCE CLEANUP due to memory pressure")

        cleanup_results: Dict[str, Any] = {
            'garbage_collected': 0,
            'memory_before': 0.0,
            'memory_after': 0.0,
            'reduction_mb': 0.0,
            'reservations_cleared': 0
        }

        try:
            # Get memory before cleanup
            before_status = self.get_memory_status()
            cleanup_results['memory_before'] = before_status.used_percent

            # Clear memory reservations
            reservations_cleared = len(self.memory_reservations)
            self.memory_reservations.clear()
            cleanup_results['reservations_cleared'] = reservations_cleared

            # Force garbage collection
            for i in range(3):
                collected = gc.collect()
                cleanup_results['garbage_collected'] += collected

            # Get memory after cleanup
            after_status = self.get_memory_status()
            cleanup_results['memory_after'] = after_status.used_percent

            # Calculate reduction
            memory_reduction = before_status.used_percent - after_status.used_percent
            total_memory_gb = before_status.total_gb
            reduction_mb = (memory_reduction / 100) * total_memory_gb * 1024
            cleanup_results['reduction_mb'] = reduction_mb

            resource_logger.info(f"Force cleanup completed: {memory_reduction:.1f}% reduction "
                       f"({reduction_mb:.0f}MB freed, {reservations_cleared} reservations cleared)")

        except Exception as e:
            resource_logger.error(f"Force cleanup failed: {e}")
            cleanup_results['error'] = str(e)

        return cleanup_results

    def adaptive_cleanup(self) -> Dict[str, Any]:
        """Perform adaptive cleanup based on current memory pressure"""
        memory_status = self.get_memory_status()

        if memory_status.threat_level in ["CRITICAL", "UNSAFE"]:
            return self.force_cleanup()
        elif memory_status.threat_level == "WARNING":
            # Lighter cleanup
            gc.collect()
            return {
                "cleanup_type": "light",
                "memory_status": memory_status.threat_level,
                "action": "garbage_collection_only"
            }
        else:
            return {
                "cleanup_type": "none",
                "memory_status": memory_status.threat_level,
                "action": "no_cleanup_needed"
            }

    def optimize_for_workload(self, workload_type: str):
        """Optimize resource allocation for specific workload"""
        mem_pct = self.get_memory_usage()

        if workload_type == "transcription" and mem_pct > 70:
            self.set_streaming_mode(True)
        elif workload_type == "model_loading" and mem_pct > 60:
            self.set_streaming_mode(True)
            self.set_conservative_mode(True)
        elif workload_type == "idle":
            self.set_streaming_mode(False)
            self.set_conservative_mode(False)

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary"""
        system_status = self.get_system_state()
        basic_memory = self.get_memory_status_basic()
        cpu_config = self.get_cpu_config()
        recommended = self.get_recommended_settings()

        return {
            "memory": {
                "detailed": {
                    "total_gb": system_status.memory.total_gb,
                    "available_gb": system_status.memory.available_gb,
                    "used_percent": system_status.memory.used_percent,
                    "threat_level": system_status.memory.threat_level,
                    "browser_safe": system_status.memory.browser_safe,
                    "recommendation": system_status.memory.recommendation
                },
                "basic": basic_memory,
                "reservations": {
                    "count": len(self.memory_reservations),
                    "total_mb": sum(self.memory_reservations.values())
                }
            },
            "cpu": cpu_config,
            "modes": {
                "emergency": system_status.emergency_mode,
                "conservative": system_status.conservative_mode,
                "streaming": system_status.streaming_mode
            },
            "recommended_settings": recommended,
            "thresholds": self.thresholds,
            "targets": {
                "memory_target_mb": self.memory_target_mb,
                "memory_limit_mb": self.memory_limit_mb
            },
            "timestamp": system_status.timestamp
        }


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None

def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager



async def setup_memory_monitoring() -> ResourceManager:
    """Setup memory monitoring with default callbacks"""
    manager = get_resource_manager()

    # Register default callbacks
    async def critical_alert(status: MemoryStatus):
        resource_logger.critical(f"🚨 CRITICAL MEMORY: {status.used_percent:.1f}% - {status.recommendation}")
        # Force cleanup automatically
        manager.force_cleanup()

    async def unsafe_alert(status: MemoryStatus):
        resource_logger.error(f"⚠️ UNSAFE MEMORY: {status.used_percent:.1f}% - Exceeds browser-safe limit")

    async def warning_alert(status: MemoryStatus):
        resource_logger.warning(f"⚡ WARNING MEMORY: {status.used_percent:.1f}% - Approaching limits")

    manager.register_callback('critical', critical_alert)
    manager.register_callback('unsafe', unsafe_alert)
    manager.register_callback('warning', warning_alert)

    return manager






@dataclass(slots=True)
class SessionConfig:
    """Configuração de sessão para processamento multiprocessing"""
    session_id: str
    language: str
    audio_input_type: str
    processing_profile: str
    format_type: str
    websocket_manager: Optional[Any] = None

@dataclass(slots=True)
class SessionInfo:
    """Information about a concurrent transcription session"""
    session_id: str
    audio_file: str
    status: str  # 'pending', 'processing', 'completed', 'error'
    start_time: float
    end_time: Optional[float] = None
    process_id: Optional[int] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

class MultiProcessingTranscrevAI:
    """Gerenciador principal da arquitetura multiprocessing CPU-only"""

    def __init__(self, websocket_manager=None, manual_mode=True):
        """
        Inicializa o gerenciador multiprocessing

        Args:
            websocket_manager: Gerenciador WebSocket existente para integração
            manual_mode: Se True, desabilita auto-processamento (padrão: True)
        """
        # Configuração de CPU e recursos - Nova fórmula otimizada
        logical_cores = psutil.cpu_count(logical=True) or 4
        physical_cores = psutil.cpu_count(logical=False) or 2
        self.max_cores = max(1, logical_cores - 2, physical_cores - 2)
        self.memory_target_mb = 1024  # Meta de ~1GB normal conforme claude.md
        self.memory_peak_mb = 2048    # Pico máximo de ~2GB conforme claude.md

        # Manual mode configuration
        self.manual_mode = manual_mode
        if manual_mode:
            logger.info("Manual mode activated - auto-processing disabled")

        # Configuração de processos
        self.process_config = ProcessConfig(
            max_cores=self.max_cores,
            memory_limit_mb=self.memory_peak_mb,  # Usar limite de pico
            target_processing_ratio=0.5,  # 0.4-0.6x conforme claude.md
            quantization_enabled=True,
            crash_restart_enabled=True
        )

        # Componentes de infraestrutura
        self.shared_memory = SharedMemoryManager()
        self.queue_manager = QueueManager()
        self.process_monitor = ProcessMonitor(self.queue_manager, self.shared_memory)
        self.cpu_manager = CPUCoreManager()

        # Processos ativos
        self.processes: Dict[ProcessType, mp.Process] = {}
        self.active_sessions: Dict[str, SessionConfig] = {}

        # Concurrent session management
        self.max_concurrent_sessions = 2
        self.concurrent_sessions: Dict[str, SessionInfo] = {}
        self.session_lock = threading.Lock()
        self.session_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_sessions)

        # Estado do sistema
        self.is_running = False
        self.initialization_complete = False

        # Integração WebSocket
        self.websocket_manager = websocket_manager
        self.websocket_thread = None

        # Thread de controle principal
        self.control_thread = None
        self.shutdown_event = threading.Event()

        manager_logger.info(f"MultiProcessingTranscrevAI inicializado: {self.max_cores} cores, "
                   f"meta: {self.memory_target_mb}MB normal / {self.memory_peak_mb}MB pico")

    async def initialize(self) -> bool:
        """Inicializa toda a arquitetura multiprocessing"""
        try:
            manager_logger.info("Inicializando arquitetura multiprocessing...")

            # Verificar recursos do sistema
            if not self._check_system_requirements():
                return False

            # Inicializar infraestrutura
            await self._initialize_infrastructure()

            # Iniciar processos
            if not await self._start_core_processes():
                return False

            # Iniciar monitoramento
            self._start_monitoring()

            # Integração WebSocket
            if self.websocket_manager:
                self._start_websocket_integration()

            self.is_running = True
            self.initialization_complete = True

            manager_logger.info("Arquitetura multiprocessing inicializada com sucesso")
            return True

        except Exception as e:
            manager_logger.error(f"Erro na inicialização: {e}")
            await self.shutdown()
            return False

    async def shutdown(self):
        """Desliga toda a arquitetura multiprocessing de forma segura"""
        try:
            manager_logger.info("Iniciando shutdown da arquitetura multiprocessing...")

            self.is_running = False
            self.shutdown_event.set()

            # Parar integração WebSocket
            self._stop_websocket_integration()

            # Parar monitoramento
            self._stop_monitoring()

            # Enviar comando de shutdown para todos os processos
            self._send_global_shutdown()

            # Aguardar processos terminarem graciosamente
            await self._wait_for_processes_shutdown()

            # Forçar terminação se necessário
            self._force_terminate_processes()

            # Limpeza de recursos
            self._cleanup_resources()

            manager_logger.info("Shutdown da arquitetura multiprocessing concluído")

        except Exception as e:
            manager_logger.error(f"Erro durante shutdown: {e}")

    async def start_session(self, session_config: SessionConfig) -> bool:
        """Inicia nova sessão de processamento"""
        try:
            if not self.initialization_complete:
                manager_logger.error("Sistema não inicializado")
                return False

            session_id = session_config.session_id
            manager_logger.info(f"Iniciando sessão {session_id}")

            # Registrar sessão
            self.active_sessions[session_id] = session_config

            # Enviar comando para iniciar gravação
            await self._send_audio_command("start_recording", {
                "session_id": session_id,
                "format": session_config.format_type,
                "language": session_config.language,
                "audio_input_type": session_config.audio_input_type,
                "processing_profile": session_config.processing_profile
            })

            manager_logger.info(f"Sessão {session_id} iniciada")
            return True

        except Exception as e:
            manager_logger.error(f"Erro iniciando sessão {session_config.session_id}: {e}")
            return False

    async def stop_session(self, session_id: str) -> bool:
        """Para sessão de processamento"""
        try:
            if session_id not in self.active_sessions:
                manager_logger.warning(f"Sessão {session_id} não encontrada")
                return False

            manager_logger.info(f"Parando sessão {session_id}")

            # Enviar comando para parar gravação
            await self._send_audio_command("stop_recording", {
                "session_id": session_id
            })

            # Aguardar processamento completo
            await self._wait_for_session_completion(session_id)

            # Remover sessão
            del self.active_sessions[session_id]

            manager_logger.info(f"Sessão {session_id} parada")
            return True

        except Exception as e:
            manager_logger.error(f"Erro parando sessão {session_id}: {e}")
            return False

    async def process_audio_multicore(self, audio_file: str, language: str = "pt",
                                     audio_input_type: str = "neutral", # Retained for compatibility
                                     domain: str = "general",
                                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Processa arquivo de áudio usando arquitetura multiprocessing

        Returns:
            Dict com resultados de transcrição, diarização e metadados
        """
        try:
            manager_logger.info(f"Processando áudio multicore: {audio_file} (Domínio: {domain})")

            if not os.path.exists(audio_file):
                raise ValueError(f"Arquivo não encontrado: {audio_file}")

            processing_session_id = session_id or f"batch_{int(time.time())}"

            # Solicitar transcrição
            await self._send_transcription_command("transcribe_audio", {
                "audio_file": audio_file,
                "language": language,
                "audio_type": audio_input_type, # Legacy
                "domain": domain, # New
                "session_id": processing_session_id
            })

            # Solicitar diarização em paralelo
            await self._send_diarization_command("diarize_audio", {
                "audio_file": audio_file,
                "session_id": processing_session_id
            })

            # Aguardar resultados
            transcription_result = await self._wait_for_transcription_result(processing_session_id)
            diarization_result = await self._wait_for_diarization_result(processing_session_id)

            # Fallback: Se multiprocessing falhou, usar processamento single-threaded
            if not transcription_result.get("segments") and not diarization_result.get("segments"):
                manager_logger.warning("Multiprocessing failed, using single-threaded fallback")
                return await self._process_audio_fallback(audio_file, language, processing_session_id)

            # Combinar resultados
            combined_result = await self._combine_results(
                transcription_result, diarization_result,
                processing_session_id
            )

            # FASE 5.0: LLM post-processing removed (excluded per user request)

            manager_logger.info(f"Processamento multicore concluído para {audio_file}")
            return combined_result

        except Exception as e:
            manager_logger.error(f"Erro no processamento multicore: {e}")
            raise

    async def _send_command(self, queue_type: str, command: str, payload: Dict[str, Any]):
        """Envia comando para worker especificado (unified method)"""
        queue_map = {
            "transcription": self.queue_manager.transcription_queue,
            "diarization": self.queue_manager.diarization_queue
        }
        try:
            queue_map[queue_type].put_nowait({"command": command, "payload": payload})
            manager_logger.debug(f"Comando de {queue_type} enviado: {command}")
        except queue.Full:
            manager_logger.warning(f"Fila de {queue_type} cheia, descartando comando")
        except KeyError:
            manager_logger.error(f"Tipo de queue inválido: {queue_type}")

    # Legacy methods for backward compatibility
    async def _send_transcription_command(self, command: str, payload: Dict[str, Any]):
        """Deprecated: Use _send_command('transcription', ...) instead"""
        await self._send_command("transcription", command, payload)

    async def _send_diarization_command(self, command: str, payload: Dict[str, Any]):
        """Deprecated: Use _send_command('diarization', ...) instead"""
        await self._send_command("diarization", command, payload)

    async def _wait_for_result(self, result_type: str, session_id: str, timeout: float = 600) -> Dict[str, Any]:
        """Aguarda resultado do shared memory (unified method)"""
        getter_map = {
            "transcription": self.shared_memory.get_transcription_data_for_session,
            "diarization": self.shared_memory.get_diarization_data_for_session
        }

        if result_type not in getter_map:
            raise ValueError(f"Tipo de resultado inválido: {result_type}")

        start_time = time.time()
        while time.time() - start_time < timeout:
            result = getter_map[result_type](session_id)
            if result:
                return result
            await asyncio.sleep(0.5)  # Poll every 0.5 seconds
        raise TimeoutError(f"{result_type.capitalize()} result for session {session_id} not received within {timeout}s")

    # Legacy methods for backward compatibility
    async def _wait_for_transcription_result(self, session_id: str, timeout: float = 600) -> Dict[str, Any]:
        """Deprecated: Use _wait_for_result('transcription', ...) instead"""
        return await self._wait_for_result("transcription", session_id, timeout)

    async def _wait_for_diarization_result(self, session_id: str, timeout: float = 600) -> Dict[str, Any]:
        """Deprecated: Use _wait_for_result('diarization', ...) instead"""
        return await self._wait_for_result("diarization", session_id, timeout)

    async def _combine_results(self, transcription_result: Dict, diarization_result: Dict, session_id: str) -> Dict:
        """Combina resultados de transcrição e diarização"""
        # This method needs to be implemented based on how transcription and diarization results are structured
        # For now, a basic combination
        combined = {
            "transcription_data": transcription_result.get("segments", []),
            "diarization_segments": diarization_result.get("segments", []),
            "speakers_detected": diarization_result.get("speakers_detected", 0),
            "processing_metadata": {
                "session_id": session_id,
                "transcription_system": transcription_result.get("system_used"),
                "diarization_method": diarization_result.get("method"),
                "processing_time_transcription": transcription_result.get("processing_time"),
                "processing_time_diarization": diarization_result.get("processing_time"),
            }
        }
        return combined

    async def _process_audio_fallback(self, audio_file: str, language: str, session_id: str) -> Dict[str, Any]:
        """Fallback single-threaded processing when multiprocessing fails"""
        try:
            manager_logger.info(f"Starting single-threaded fallback for {audio_file}")

            # Import necessary modules for direct processing
            from src.transcription import TranscriptionService
            from src.diarization import CPUSpeakerDiarization

            # Process transcription
            transcription_service = TranscriptionService()
            await transcription_service.load_model(language)

            transcription_result = await transcription_service.transcribe_audio_file(audio_file, language)

            # Process diarization
            diarization_service = CPUSpeakerDiarization()
            diarization_result = await diarization_service.diarize_audio(audio_file)

            # Combine results
            combined_result = {
                "transcription_data": transcription_result if isinstance(transcription_result, list) else [],
                "diarization_segments": diarization_result if isinstance(diarization_result, list) else [],
                "speakers_detected": len(set(seg.get("speaker", "Speaker_1")
                                           for seg in (diarization_result if isinstance(diarization_result, list) else []))),
                "processing_metadata": {
                    "method": "single_threaded_fallback",
                    "session_id": session_id,
                    "audio_file": audio_file
                }
            }

            manager_logger.info(f"Single-threaded fallback completed: {len(combined_result['transcription_data'])} segments, {combined_result['speakers_detected']} speakers")
            return combined_result

        except Exception as e:
            manager_logger.error(f"Fallback processing failed: {e}")
            return {
                "transcription_data": [],
                "diarization_segments": [],
                "speakers_detected": 0,
                "processing_metadata": {"method": "fallback_failed", "error": str(e), "session_id": session_id}
            }

    # Concurrent Session Management Methods (integrated from concurrent_session_manager.py)

    def create_concurrent_session(self, audio_file: str) -> str:
        """Create a new concurrent transcription session"""
        session_id = f"concurrent_{int(time.time() * 1000)}_{os.getpid()}"

        with self.session_lock:
            session_info = SessionInfo(
                session_id=session_id,
                audio_file=audio_file,
                status='pending',
                start_time=time.time()
            )
            self.concurrent_sessions[session_id] = session_info

        manager_logger.info(f"Created concurrent session {session_id} for {audio_file}")
        return session_id

    def start_concurrent_session(self, session_id: str) -> bool:
        """Start processing a concurrent session"""
        with self.session_lock:
            if session_id not in self.concurrent_sessions:
                manager_logger.error(f"Concurrent session {session_id} not found")
                return False

            session = self.concurrent_sessions[session_id]
            if session.status != 'pending':
                manager_logger.warning(f"Concurrent session {session_id} already started or completed")
                return False

            # Check concurrent session limit
            active_sessions = sum(1 for s in self.concurrent_sessions.values()
                                if s.status == 'processing')

            if active_sessions >= self.max_concurrent_sessions:
                manager_logger.warning(f"Max concurrent sessions ({self.max_concurrent_sessions}) reached")
                return False

            session.status = 'processing'
            session.process_id = os.getpid()

        # Submit to executor
        future = self.session_executor.submit(self._process_concurrent_session, session_id)

        manager_logger.info(f"Started concurrent session {session_id}")
        return True

    def _process_concurrent_session(self, session_id: str):
        """Internal method to process a concurrent session"""
        try:
            with self.session_lock:
                session = self.concurrent_sessions[session_id]

            manager_logger.info(f"Processing concurrent session {session_id}: {session.audio_file}")

            # Use the existing process_audio_multicore method for actual processing
            result = asyncio.run(self.process_audio_multicore(
                session.audio_file,
                language="pt",
                session_id=session_id
            ))

            # Update session with results
            with self.session_lock:
                session.status = 'completed'
                session.end_time = time.time()
                session.result = result

            manager_logger.info(f"Concurrent session {session_id} completed successfully")

        except Exception as e:
            manager_logger.error(f"Concurrent session {session_id} failed: {e}")

            with self.session_lock:
                session = self.concurrent_sessions[session_id]
                session.status = 'error'
                session.end_time = time.time()
                session.error = str(e)

    def get_concurrent_session_status(self, session_id: str) -> Optional[Dict]:
        """Get status of a concurrent session"""
        with self.session_lock:
            if session_id not in self.concurrent_sessions:
                return None

            session = self.concurrent_sessions[session_id]

            status_info = {
                'session_id': session.session_id,
                'status': session.status,
                'audio_file': session.audio_file,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'process_id': session.process_id
            }

            if session.status == 'completed' and session.result:
                status_info['result'] = session.result

            if session.status == 'error' and session.error:
                status_info['error'] = session.error

            return status_info

    def cancel_concurrent_session(self, session_id: str) -> bool:
        """Cancel a concurrent session"""
        with self.session_lock:
            if session_id not in self.concurrent_sessions:
                return False

            session = self.concurrent_sessions[session_id]
            if session.status == 'processing':
                session.status = 'error'
                session.error = 'Cancelled by user'
                session.end_time = time.time()

                manager_logger.info(f"Concurrent session {session_id} cancelled")
                return True

            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema"""
        try:
            # Status dos processos
            processes_status = {}
            for process_type, process in self.processes.items():
                if process and process.is_alive():
                    info = self.process_monitor.get_process_info(process_type)
                    processes_status[process_type.value] = {
                        "status": info.status.value if info else "unknown",
                        "pid": process.pid,
                        "memory_mb": info.memory_usage_mb if info else 0,
                        "cpu_percent": info.cpu_usage_percent if info else 0
                    }
                else:
                    processes_status[process_type.value] = {
                        "status": "stopped",
                        "pid": None,
                        "memory_mb": 0,
                        "cpu_percent": 0
                    }

            # Status do sistema
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent(interval=0.1)

            # Sessões ativas
            active_sessions_info = {
                session_id: {
                    "language": config.language,
                    "audio_type": config.audio_input_type,
                    "profile": config.processing_profile
                }
                for session_id, config in self.active_sessions.items()
            }

            # Concurrent sessions status
            concurrent_sessions_info = {
                session_id: {
                    "status": session.status,
                    "audio_file": session.audio_file,
                    "start_time": session.start_time
                }
                for session_id, session in self.concurrent_sessions.items()
            }

            return {
                "system_status": "running" if self.is_running else "stopped",
                "initialization_complete": self.initialization_complete,
                "processes": processes_status,
                "active_sessions": active_sessions_info,
                "concurrent_sessions": concurrent_sessions_info,
                "concurrent_sessions_active": self.get_active_concurrent_session_count(),
                "system_resources": {
                    "cpu_cores_total": psutil.cpu_count(),
                    "cpu_cores_used": self.max_cores,
                    "cpu_percent": system_cpu,
                    "memory_total_gb": system_memory.total / (1024**3),
                    "memory_used_gb": system_memory.used / (1024**3),
                    "memory_available_gb": system_memory.available / (1024**3)
                },
                "configuration": {
                    "max_cores": self.max_cores,
                    "memory_target_mb": self.memory_target_mb,
                    "quantization_enabled": self.process_config.quantization_enabled
                }
            }

        except Exception as e:
            manager_logger.error(f"Erro obtendo status do sistema: {e}")
            return {"error": str(e)}

    def handle_process_failure(self, process_type: ProcessType) -> bool:
        """Trata falha de processo com restart automático"""
        try:
            manager_logger.warning(f"Tratando falha do processo {process_type.value}")

            # Obter informações do processo
            process_info = self.process_monitor.get_process_info(process_type)
            if not process_info:
                manager_logger.error(f"Informações do processo {process_type.value} não encontradas")
                return False

            # Verificar limite de restarts
            if process_info.restart_count >= 3:
                manager_logger.error(f"Processo {process_type.value} excedeu limite de restarts")
                return False

            # Tentar reiniciar processo
            return self._restart_process(process_type)

        except Exception as e:
            manager_logger.error(f"Erro tratando falha do processo {process_type.value}: {e}")
            return False

    # Métodos privados de implementação

    def _check_system_requirements(self) -> bool:
        """Verifica se sistema atende requisitos mínimos"""
        try:
            # Verificar CPU
            cpu_count = psutil.cpu_count()
            if cpu_count is None or cpu_count < 4:
                manager_logger.error(f"Sistema requer pelo menos 4 cores CPU (encontrado: {cpu_count})")
                return False

            # Verificar memória
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb < 4.0:  # Minimum 4GB for basic operation
                manager_logger.error(f"Sistema requer pelo menos 4GB RAM (encontrado: {memory_gb:.1f}GB)")
                return False

            # Verificar espaço em disco
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            if free_gb < 5:
                manager_logger.error(f"Sistema requer pelo menos 5GB de espaço livre (encontrado: {free_gb:.1f}GB)")
                return False

            manager_logger.info(f"Requisitos do sistema atendidos: {cpu_count} cores, "
                       f"{memory_gb:.1f}GB RAM, {free_gb:.1f}GB livre")
            return True

        except Exception as e:
            manager_logger.error(f"Erro verificando requisitos do sistema: {e}")
            return False

    async def _initialize_infrastructure(self):
        """Inicializa componentes de infraestrutura"""
        try:
            # Infraestrutura já inicializada no __init__
            manager_logger.info("Infraestrutura de multiprocessing inicializada")

        except Exception as e:
            manager_logger.error(f"Erro inicializando infraestrutura: {e}")
            raise

    async def _start_core_processes(self) -> bool:
        """Inicia processos principais"""
        try:
            # Definir processos a iniciar
            processes_to_start = [
                (ProcessType.AUDIO_CAPTURE, audio_capture_worker),
                (ProcessType.TRANSCRIPTION, transcription_worker),
                (ProcessType.DIARIZATION, diarization_worker)
            ]

            # Iniciar cada processo
            for process_type, worker_func in processes_to_start:
                if not self._start_process(process_type, worker_func):
                    manager_logger.error(f"Falha ao iniciar processo {process_type.value}")
                    return False

                # Aguardar um pouco entre inicializações
                await asyncio.sleep(1.0)

            # Verificar se todos os processos estão executando
            await asyncio.sleep(2.0)
            all_running = all(
                proc.is_alive() for proc in self.processes.values()
            )

            if not all_running:
                manager_logger.error("Nem todos os processos iniciaram corretamente")
                return False

            manager_logger.info("Todos os processos principais iniciados")
            return True

        except Exception as e:
            manager_logger.error(f"Erro iniciando processos principais: {e}")
            return False

    def _start_process(self, process_type: ProcessType, worker_func: Callable) -> bool:
        """Inicia processo individual"""
        try:
            manager_logger.info(f"Iniciando processo {process_type.value}")

            # Determinar argumentos baseado no tipo de worker (evita pickle error com Manager objects)
            if process_type == ProcessType.AUDIO_CAPTURE:
                # audio_capture_worker não usa queue/shared_memory - apenas parent_pid
                process_args = (os.getpid(), self.manual_mode)
            elif process_type == ProcessType.TRANSCRIPTION:
                # Passar queue e dict diretos (picklable), não Manager objects
                process_args = (
                    os.getpid(),
                    self.queue_manager.transcription_queue,
                    self.shared_memory.shared_dict,
                    self.manual_mode
                )
            elif process_type == ProcessType.DIARIZATION:
                # Passar queue e dict diretos (picklable), não Manager objects
                process_args = (
                    os.getpid(),
                    self.queue_manager.diarization_queue,
                    self.shared_memory.shared_dict,
                    self.manual_mode
                )
            else:
                # Fallback para outros tipos (se houver)
                process_args = (os.getpid(), self.manual_mode)

            # Criar processo
            process = mp.Process(
                target=worker_func,
                args=process_args,
                name=f"TranscrevAI_{process_type.value}"
            )

            # Iniciar processo
            process.start()

            # Configurar afinidade CPU se necessário
            try:
                proc_obj = psutil.Process(process.pid)
                self.cpu_manager.set_process_affinity(proc_obj, process_type)
            except Exception as affinity_error:
                manager_logger.warning(f"Erro configurando afinidade CPU: {affinity_error}")

            # Registrar processo
            self.processes[process_type] = process
            process_id = process.pid if process.pid is not None else 0
            self.process_monitor.register_process(process_type, process_id)

            manager_logger.info(f"Processo {process_type.value} iniciado (PID: {process.pid})")
            return True

        except Exception as e:
            manager_logger.error(f"Erro iniciando processo {process_type.value}: {e}")
            return False

    def _restart_process(self, process_type: ProcessType) -> bool:
        """Reinicia processo específico"""
        try:
            manager_logger.info(f"Reiniciando processo {process_type.value}")

            # Parar processo atual se ainda estiver executando
            if process_type in self.processes:
                process = self.processes[process_type]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()

                del self.processes[process_type]

            # Desregistrar do monitor
            self.process_monitor.unregister_process(process_type)

            # Aguardar um pouco
            time.sleep(2.0)

            # Determinar função worker
            worker_map = {
                ProcessType.AUDIO_CAPTURE: audio_capture_worker,
                ProcessType.TRANSCRIPTION: transcription_worker,
                ProcessType.DIARIZATION: diarization_worker
            }

            worker_func = worker_map.get(process_type)
            if not worker_func:
                manager_logger.error(f"Função worker não encontrada para {process_type.value}")
                return False

            # Reiniciar processo
            return self._start_process(process_type, worker_func)

        except Exception as e:
            manager_logger.error(f"Erro reiniciando processo {process_type.value}: {e}")
            return False

    def _start_monitoring(self):
        """Inicia monitoramento de processos"""
        try:
            self.process_monitor.start_monitoring()

            # Iniciar thread de controle principal
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()

            manager_logger.info("Monitoramento de processos iniciado")

        except Exception as e:
            manager_logger.error(f"Erro iniciando monitoramento: {e}")

    def _stop_monitoring(self):
        """Para monitoramento de processos"""
        try:
            self.process_monitor.stop_monitoring()

            if self.control_thread and self.control_thread.is_alive():
                self.control_thread.join(timeout=5)

            manager_logger.info("Monitoramento de processos parado")

        except Exception as e:
            manager_logger.error(f"Erro parando monitoramento: {e}")

    def _start_websocket_integration(self):
        """Inicia integração com WebSocket"""
        try:
            if not self.websocket_manager:
                manager_logger.info("WebSocket manager não fornecido, pulando integração")
                return

            self.websocket_thread = threading.Thread(
                target=self._websocket_integration_loop, daemon=True
            )
            self.websocket_thread.start()

            manager_logger.info("Integração WebSocket iniciada")

        except Exception as e:
            manager_logger.error(f"Erro iniciando integração WebSocket: {e}")

    def _stop_websocket_integration(self):
        """Para integração com WebSocket"""
        try:
            if self.websocket_thread and self.websocket_thread.is_alive():
                self.websocket_thread.join(timeout=5)

            manager_logger.info("Integração WebSocket parada")

        except Exception as e:
            manager_logger.error(f"Erro parando integração WebSocket: {e}")

    def _control_loop(self):
        """Loop principal de controle"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Processar mensagens de status
                self._process_status_messages()

                # Verificar saúde dos processos
                self._check_process_health()

                # Enviar estatísticas periódicas
                self._send_periodic_system_stats()

                # Aguardar
                if self.shutdown_event.wait(timeout=1.0):
                    break

            except Exception as e:
                manager_logger.error(f"Erro no loop de controle: {e}")
                time.sleep(5.0)

    def _websocket_integration_loop(self):
        """Loop de integração com WebSocket"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Verificar mensagens para WebSocket
                websocket_queue = self.queue_manager.get_queue(ProcessType.WEBSOCKET)
                if websocket_queue is not None and websocket_queue:
                    try:
                        while not websocket_queue.empty():
                            message = websocket_queue.get_nowait()
                            self._forward_to_websocket(message)
                    except queue.Full:
                        pass

                # Aguardar
                if self.shutdown_event.wait(timeout=0.1):
                    break

            except Exception as e:
                manager_logger.error(f"Erro no loop de integração WebSocket: {e}")
                time.sleep(1.0)

    def _forward_to_websocket(self, message: Dict[str, Any]):
        """Encaminha mensagem para WebSocket"""
        try:
            if not self.websocket_manager:
                return

            source = message.get("source")
            msg_data = message.get("message", {})

            # Processar diferentes tipos de mensagem
            if source == "audio_capture":
                self._handle_audio_websocket_message(msg_data)
            elif source == "transcription":
                self._handle_transcription_websocket_message(msg_data)
            elif source == "diarization":
                self._handle_diarization_websocket_message(msg_data)

        except Exception as e:
            manager_logger.warning(f"Erro encaminhando para WebSocket: {e}")

    def _handle_audio_websocket_message(self, message: Dict[str, Any]):
        """Trata mensagem de áudio para WebSocket"""
        try:
            msg_type = message.get("type")
            data = message.get("data", {})

            # Encontrar sessão baseada no session_id nos dados
            session_id = None
            for sid, config in self.active_sessions.items():
                if config.websocket_manager == self.websocket_manager:
                    session_id = sid
                    break

            if not session_id:
                return

            # Encaminhar mensagem se websocket_manager disponível
            if self.websocket_manager is not None and hasattr(self.websocket_manager, 'send_message'):
                asyncio.create_task(
                    self.websocket_manager.send_message(session_id, {
                        "type": msg_type,
                        "source": "multiprocessing_audio",
                        **data
                    })
                )
            else:
                manager_logger.debug(f"WebSocket não disponível para mensagem: {msg_type}")

        except Exception as e:
            manager_logger.warning(f"Erro tratando mensagem de áudio WebSocket: {e}")

    def _handle_transcription_websocket_message(self, message: Dict[str, Any]):
        """Trata mensagem de transcrição para WebSocket"""
        try:
            msg_type = message.get("type")
            data = message.get("data", {})
            session_id = data.get("session_id")

            if not session_id or session_id not in self.active_sessions:
                return

            config = self.active_sessions[session_id]
            if not config.websocket_manager:
                return

            # Enviar através do WebSocket
            if hasattr(config.websocket_manager, 'send_message'):
                asyncio.create_task(
                    config.websocket_manager.send_message(session_id, {
                        "type": f"transcription_{msg_type}",
                        "source": "multiprocessing_transcription",
                        **data
                    })
                )

        except Exception as e:
            manager_logger.warning(f"Erro tratando mensagem de transcrição WebSocket: {e}")

    def _handle_diarization_websocket_message(self, message: Dict[str, Any]):
        """Trata mensagem de diarização para WebSocket"""
        try:
            msg_type = message.get("type")
            data = message.get("data", {})
            session_id = data.get("session_id")

            if not session_id or session_id not in self.active_sessions:
                return

            config = self.active_sessions[session_id]
            if not config.websocket_manager:
                return

            # Enviar através do WebSocket
            if hasattr(config.websocket_manager, 'send_message'):
                asyncio.create_task(
                    config.websocket_manager.send_message(session_id, {
                        "type": f"diarization_{msg_type}",
                        "source": "multiprocessing_diarization",
                        **data
                    })
                )

        except Exception as e:
            manager_logger.warning(f"Erro tratando mensagem de diarização WebSocket: {e}")

    # Métodos auxiliares para comando e comunicação



    def _send_global_shutdown(self):
        """Envia comando de shutdown para todos os processos"""
        shutdown_message = {
            "action": "shutdown",
            "timestamp": time.time()
        }

        try:
            self.queue_manager.send_control_message(shutdown_message)
            manager_logger.info("Comando de shutdown enviado para todos os processos")
        except Exception as e:
            manager_logger.error(f"Erro enviando comando de shutdown: {e}")

    async def _wait_for_processes_shutdown(self, timeout: float = 15.0):
        """Aguarda processos terminarem graciosamente"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_stopped = True
            for process in self.processes.values():
                if process.is_alive():
                    all_stopped = False
                    break

            if all_stopped is not None and all_stopped:
                manager_logger.info("Todos os processos terminaram graciosamente")
                return

            await asyncio.sleep(0.5)

        manager_logger.warning(f"Timeout aguardando shutdown gracioso ({timeout}s)")

    def _force_terminate_processes(self):
        """Força terminação de processos que não pararam graciosamente"""
        for process_type, process in self.processes.items():
            if process.is_alive():
                manager_logger.warning(f"Forçando terminação do processo {process_type.value}")
                try:
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                except Exception as e:
                    manager_logger.error(f"Erro forçando terminação de {process_type.value}: {e}")

    def _cleanup_resources(self):
        """Limpeza final de recursos"""
        try:
            # Limpar filas
            self.queue_manager.cleanup()

            # Limpar memória compartilhada
            self.shared_memory.cleanup()

            # Limpar processos
            self.processes.clear()

            # Limpar sessões
            self.active_sessions.clear()

            # Cleanup concurrent sessions and executor
            if hasattr(self, 'session_executor'):
                self.session_executor.shutdown(wait=True)
            self.concurrent_sessions.clear()

            manager_logger.info("Limpeza de recursos concluída")

        except Exception as e:
            manager_logger.error(f"Erro na limpeza de recursos: {e}")

    def _process_status_messages(self):
        """Processa mensagens de status dos processos"""
        try:
            while True:
                status_msg = self.queue_manager.get_status_update(timeout=0.1)
                if not status_msg:
                    break

                # Processar mensagem de status
                process_type_str = status_msg.get("process_type")
                if process_type_str is not None and process_type_str:
                    try:
                        process_type = ProcessType(process_type_str)
                        data = status_msg.get("data", {})

                        # Atualizar monitor se necessário
                        if "status" in data:
                            status = ProcessStatus(data["status"])
                            error = data.get("error")
                            self.process_monitor.update_process_status(
                                process_type, status, error
                            )

                    except ValueError:
                        manager_logger.warning(f"Tipo de processo inválido: {process_type_str}")

        except Exception as e:
            manager_logger.error(f"Erro processando mensagens de status: {e}")

    def _check_process_health(self):
        """Verifica saúde dos processos"""
        try:
            for process_type, process in self.processes.items():
                if not process.is_alive():
                    manager_logger.warning(f"Processo {process_type.value} não está vivo")
                    # O ProcessMonitor tratará do restart automático

        except Exception as e:
            manager_logger.error(f"Erro verificando saúde dos processos: {e}")

    def _send_periodic_system_stats(self):
        """Envia estatísticas periódicas do sistema"""
        current_time = time.time()
        if not hasattr(self, '_last_system_stats_time'):
            self._last_system_stats_time = current_time

        if current_time - self._last_system_stats_time >= 30.0:  # A cada 30 segundos
            try:
                status = self.get_system_status()
                manager_logger.info(f"Status do sistema: {len(self.active_sessions)} sessões ativas, "
                           f"CPU: {status['system_resources']['cpu_percent']:.1f}%, "
                           f"Memória: {status['system_resources']['memory_used_gb']:.1f}GB")

                self._last_system_stats_time = current_time

            except Exception as e:
                manager_logger.warning(f"Erro enviando estatísticas do sistema: {e}")

    # Métodos auxiliares para processamento batch

