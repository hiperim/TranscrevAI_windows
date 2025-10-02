# Fix: Implement Missing IPC Methods in MultiProcessingTranscrevAI

**File:** `src/performance_optimizer.py`

**Change:** Implemented the following missing methods within the `MultiProcessingTranscrevAI` class:
- `_send_transcription_command(command: str, payload: Dict[str, Any])`
- `_send_diarization_command(command: str, payload: Dict[str, Any])`
- `_wait_for_transcription_result(session_id: str, timeout: float = 600) -> Dict[str, Any]`
- `_wait_for_diarization_result(session_id: str, timeout: float = 600) -> Dict[str, Any]`

**Justification:**
These methods were called within `MultiProcessingTranscrevAI.process_audio_multicore` but were not defined, leading to `AttributeError` and preventing the multiprocessing pipeline from functioning. Their implementation is critical for establishing proper inter-process communication (IPC).

- `_send_transcription_command` and `_send_diarization_command` are responsible for placing tasks (commands and their payloads) into the respective worker queues (`queue_manager.transcription_queue` and `queue_manager.diarization_queue`).
- `_wait_for_transcription_result` and `_wait_for_diarization_result` are responsible for actively polling the `SharedMemoryManager` (using the newly added session-specific methods) until the results for a given `session_id` are available or a timeout occurs.

This set of implementations completes the communication backbone for the multiprocessing architecture, allowing tasks to be dispatched to workers and their results to be retrieved reliably.
