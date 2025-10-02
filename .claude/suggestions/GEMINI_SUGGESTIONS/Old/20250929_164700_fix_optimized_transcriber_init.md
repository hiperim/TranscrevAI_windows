# Fix: OptimizedTranscriber Initialization Error

**File:** `src/performance_optimizer.py`

**Change:** In the `transcription_worker` function, the initialization of `OptimizedTranscriber` was modified from `OptimizedTranscriber(model_name="medium", cpu_manager=cpu_manager)` to `OptimizedTranscriber(cpu_manager=cpu_manager)`.

**Justification:**
The `OptimizedTranscriber` class (located in `src/transcription.py`) does not accept a `model_name` argument in its `__init__` method. This was causing a `TypeError` (`OptimizedTranscriber.__init__() got an unexpected keyword argument 'model_name'`) when the `transcription_worker` attempted to initialize it.

The `OptimizedTranscriber` internally creates a `TranscriptionService`, which is responsible for loading the Whisper model. Therefore, passing the `model_name` at this level is unnecessary and incorrect. Removing the `model_name` argument resolves this initialization error, allowing the `transcription_worker` to start correctly.
