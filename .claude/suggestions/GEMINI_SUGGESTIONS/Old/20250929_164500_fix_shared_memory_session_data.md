# Fix: Add Session-Specific Data Handling to SharedMemoryManager

**File:** `src/performance_optimizer.py`

**Change:** Added two new methods to the `SharedMemoryManager` class:
- `add_transcription_data_for_session(session_id: str, data: Dict[str, Any])`
- `get_transcription_data_for_session(session_id: str) -> Optional[Dict[str, Any]]`
- `add_diarization_data_for_session(session_id: str, data: Dict[str, Any])`
- `get_diarization_data_for_session(session_id: str) -> Optional[Dict[str, Any]]`

**Justification:**
This modification is a crucial part of resolving the `AttributeError: 'MultiProcessingTranscrevAI' object has no attribute 'session_results'` that occurred during profiling. The original design lacked a proper mechanism for worker processes to store and retrieve results in a session-specific manner via shared memory.

These new methods provide a robust way for transcription and diarization workers to:
1.  Store their processed results in a shared dictionary (`self.shared_dict`) using a unique key that includes the `session_id`.
2.  Retrieve these results from the shared dictionary, clearing them after retrieval to prevent stale data and manage memory.

This enables the `MultiProcessingTranscrevAI` class to correctly poll for results from individual sessions, facilitating proper inter-process communication and resolving a major blocking bug.
