"""
WebSocket Browser Safety Enhancements
Implementa throttling, debouncing e connection recovery
"""

import asyncio
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class WebSocketSafetyManager:
    """
    Advanced WebSocket safety manager para prevenir browser stuttering
    """

    def __init__(self):
        # Throttling: max 2 messages por segundo por sessão
        self.message_timestamps = {}  # session_id -> [timestamps]
        self.max_messages_per_second = 2

        # Debouncing: agrupar progress updates
        self.pending_progress = {}  # session_id -> latest_progress
        self.debounce_delay = 0.5  # 500ms
        self.debounce_tasks = {}   # session_id -> asyncio.Task

        # Connection recovery
        self.failed_connections = set()
        self.reconnect_attempts = {}  # session_id -> attempt_count
        self.max_reconnect_attempts = 3

    async def safe_send_message(self, websocket_manager, session_id: str, message: Dict) -> bool:
        """
        Envio seguro de mensagem com throttling e safety checks
        """
        try:
            # 1. Throttling check
            if not self._should_send_message(session_id):
                logger.debug(f"Message throttled for {session_id}")
                return False

            # 2. Memory spike prevention
            if self._is_memory_critical():
                logger.warning("Memory critical - throttling non-essential messages")
                if message.get('type') not in ['error', 'complete', 'critical']:
                    return False

            # 3. Browser safety - progress debouncing
            if message.get('type') == 'progress':
                return await self._handle_progress_debouncing(websocket_manager, session_id, message)

            # 4. Send with timeout and retry
            success = await self._send_with_retry(websocket_manager, session_id, message)

            if success:
                self._record_message_sent(session_id)

            return success

        except Exception as e:
            logger.error(f"Safe send failed for {session_id}: {e}")
            return False

    def _should_send_message(self, session_id: str) -> bool:
        """
        Verifica se mensagem pode ser enviada (throttling)
        """
        now = time.time()

        # Inicializar se necessário
        if session_id not in self.message_timestamps:
            self.message_timestamps[session_id] = []

        timestamps = self.message_timestamps[session_id]

        # Remover timestamps antigos (> 1 segundo)
        timestamps = [ts for ts in timestamps if now - ts < 1.0]
        self.message_timestamps[session_id] = timestamps

        # Verificar se pode enviar
        return len(timestamps) < self.max_messages_per_second

    def _record_message_sent(self, session_id: str):
        """Registrar que mensagem foi enviada"""
        now = time.time()
        if session_id not in self.message_timestamps:
            self.message_timestamps[session_id] = []
        self.message_timestamps[session_id].append(now)

    async def _handle_progress_debouncing(self, websocket_manager, session_id: str, message: Dict) -> bool:
        """
        Handle progress messages with debouncing para evitar flooding
        """
        # Cancelar task anterior se existir
        if session_id in self.debounce_tasks:
            self.debounce_tasks[session_id].cancel()

        # Atualizar progress pendente
        self.pending_progress[session_id] = message

        # Criar nova task de debounce
        self.debounce_tasks[session_id] = asyncio.create_task(
            self._send_debounced_progress(websocket_manager, session_id)
        )

        return True  # Message will be sent after debounce

    async def _send_debounced_progress(self, websocket_manager, session_id: str):
        """
        Enviar progress após debounce delay
        """
        try:
            await asyncio.sleep(self.debounce_delay)

            if session_id in self.pending_progress:
                message = self.pending_progress[session_id]
                await self._send_with_retry(websocket_manager, session_id, message)
                del self.pending_progress[session_id]
                self._record_message_sent(session_id)

        except asyncio.CancelledError:
            # Task foi cancelada, progress mais recente será enviado
            pass
        except Exception as e:
            logger.error(f"Debounced progress send failed: {e}")

    async def _send_with_retry(self, websocket_manager, session_id: str, message: Dict) -> bool:
        """
        Enviar mensagem com retry logic
        """
        websocket = websocket_manager.connections.get(session_id)
        if not websocket:
            return False

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                await asyncio.wait_for(websocket.send_json(message), timeout=3.0)

                # Reset failure counter on success
                if session_id in self.failed_connections:
                    self.failed_connections.remove(session_id)
                if session_id in self.reconnect_attempts:
                    del self.reconnect_attempts[session_id]

                return True

            except asyncio.TimeoutError:
                logger.warning(f"WebSocket timeout for {session_id}, attempt {attempt + 1}")
                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                continue

            except Exception as e:
                logger.error(f"WebSocket send error for {session_id}: {e}")
                self.failed_connections.add(session_id)
                break

        # Connection failed
        await self._handle_connection_failure(websocket_manager, session_id)
        return False

    async def _handle_connection_failure(self, websocket_manager, session_id: str):
        """
        Handle connection failure com recovery attempt
        """
        attempt_count = self.reconnect_attempts.get(session_id, 0)

        if attempt_count < self.max_reconnect_attempts:
            self.reconnect_attempts[session_id] = attempt_count + 1
            logger.info(f"Scheduling reconnect attempt {attempt_count + 1} for {session_id}")

            # Schedule reconnect attempt
            asyncio.create_task(self._attempt_reconnect(websocket_manager, session_id))
        else:
            logger.error(f"Max reconnect attempts reached for {session_id}")
            await websocket_manager.disconnect_websocket(session_id)

    async def _attempt_reconnect(self, websocket_manager, session_id: str):
        """
        Attempt to recover connection
        """
        try:
            await asyncio.sleep(1.0)  # Wait before reconnect

            # Send reconnection notification
            reconnect_message = {
                'type': 'connection_recovery',
                'message': 'Attempting to reconnect...',
                'session_id': session_id,
                'timestamp': time.time()
            }

            # Try to send reconnection message
            websocket = websocket_manager.connections.get(session_id)
            if websocket:
                try:
                    await websocket.send_json(reconnect_message)
                    logger.info(f"Connection recovery successful for {session_id}")
                except:
                    logger.warning(f"Connection recovery failed for {session_id}")

        except Exception as e:
            logger.error(f"Reconnect attempt failed for {session_id}: {e}")

    def _is_memory_critical(self) -> bool:
        """
        Check if system memory is critical
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent > 85.0  # Critical if >85% memory usage
        except:
            return False

    def cleanup_session(self, session_id: str):
        """
        Cleanup resources for session
        """
        # Remove throttling data
        if session_id in self.message_timestamps:
            del self.message_timestamps[session_id]

        # Cancel debounce tasks
        if session_id in self.debounce_tasks:
            self.debounce_tasks[session_id].cancel()
            del self.debounce_tasks[session_id]

        # Remove pending progress
        if session_id in self.pending_progress:
            del self.pending_progress[session_id]

        # Remove failure tracking
        self.failed_connections.discard(session_id)
        if session_id in self.reconnect_attempts:
            del self.reconnect_attempts[session_id]

# Função helper para integrar com main.py
def create_websocket_safety_manager():
    """Create WebSocket safety manager instance"""
    return WebSocketSafetyManager()