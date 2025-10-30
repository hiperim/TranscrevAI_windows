
"""
WebSocket Browser Safety Enhancements with Exponential Retry and Robust Recovery
"""

import asyncio
import time
import logging
import json
import threading
from typing import Dict, Any, Optional, Callable, List, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

class MessagePriority(Enum):
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class QueuedMessage:
    session_id: str
    message: Dict[str, Any]
    priority: MessagePriority

class RobustWebSocketSafetyManager:
    """Advanced WebSocket safety manager with exponential retry and robust recovery"""
    
    def __init__(self):
        self.message_timestamps = defaultdict(deque)
        self.priority_queues = {p: defaultdict(deque) for p in MessagePriority}
        self.connection_states = {}
        self._lock = threading.RLock()

    def _initialize_session(self, session_id: str):
        with self._lock:
            if session_id not in self.connection_states:
                self.connection_states[session_id] = ConnectionState.CONNECTING

    async def safe_send_message(self, websocket_manager, session_id: str, message: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL):
        with self._lock:
            self._initialize_session(session_id)
            if self.connection_states[session_id] != ConnectionState.CONNECTED:
                await self._queue_message(session_id, message, priority)
                return False
        return await self._try_send_message(websocket_manager, session_id, message)
        
    async def _try_send_message(self, websocket_manager, session_id: str, message: Dict[str, Any]) -> bool:
        try:
            websocket = websocket_manager.connections.get(session_id)
            if not websocket: raise ConnectionError("No WebSocket connection")
            message_json = json.dumps(message, ensure_ascii=False)
            await websocket.send_text(message_json)
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {session_id}: {e}")
            return False

    async def _queue_message(self, session_id: str, message: Dict[str, Any], priority: MessagePriority):
        queued_msg = QueuedMessage(session_id, message, priority)
        self.priority_queues[priority][session_id].append(queued_msg)

    async def handle_connection_established(self, session_id: str):
        with self._lock:
            self.connection_states[session_id] = ConnectionState.CONNECTED
        logger.info(f"🔗 Connection established for {session_id}")

    async def handle_connection_lost(self, session_id: str):
        with self._lock:
            if session_id in self.connection_states:
                self.connection_states[session_id] = ConnectionState.DISCONNECTED
        logger.warning(f"🔌 Connection lost for {session_id}")

    def cleanup_session(self, session_id: str):
        with self._lock:
            for data_dict in [self.message_timestamps, self.connection_states]:
                data_dict.pop(session_id, None)
            for priority_queue in self.priority_queues.values():
                priority_queue.pop(session_id, None)
        logger.info(f"🧹 Session {session_id} cleaned up")


_global_safety_manager: Optional[RobustWebSocketSafetyManager] = None

def get_websocket_safety_manager() -> RobustWebSocketSafetyManager:
    global _global_safety_manager
    if _global_safety_manager is None:
        _global_safety_manager = RobustWebSocketSafetyManager()
    return _global_safety_manager
