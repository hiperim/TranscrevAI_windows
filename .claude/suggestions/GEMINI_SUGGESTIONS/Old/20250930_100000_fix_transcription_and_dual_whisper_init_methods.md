# Gemini Suggestions: Fixing `cpu_manager` Argument Error

This document outlines the changes made to fix the `cpu_manager` argument error in the transcription worker. The error was caused by the `OptimizedTranscriber` class not accepting a `cpu_manager` argument, which was being passed to it in `src/performance_optimizer.py`.

The fix involved passing the `cpu_manager` argument down through the layers of the transcription service, from `OptimizedTranscriber` to `FasterWhisperEngine`, and using it to dynamically set the number of CPU threads for the transcription model.

## 1. `src/transcription.py`

### `OptimizedTranscriber.__init__`

The constructor was updated to accept an optional `cpu_manager` argument and pass it to the `TranscriptionService`.

**Before:**
```python
    def __init__(self):
        self.service = TranscriptionService()
        logger.info("OptimizedTranscriber initialized as wrapper")
```

**After:**
```python
    def __init__(self, cpu_manager=None):
        self.service = TranscriptionService(cpu_manager=cpu_manager)
        logger.info("OptimizedTranscriber initialized as wrapper")
```

### `TranscriptionService.__init__`

The constructor was updated to accept an optional `cpu_manager` argument and pass it to the `DualWhisperSystem`.

**Before:**
```python
    def __init__(self):
        self.dual_system = DualWhisperSystem(prefer_faster_whisper=True)
        logger.info("TranscriptionService initialized with dual whisper system")
```

**After:**
```python
    def __init__(self, cpu_manager=None):
        self.dual_system = DualWhisperSystem(prefer_faster_whisper=True, cpu_manager=cpu_manager)
        logger.info("TranscriptionService initialized with dual whisper system")
```

## 2. `dual_whisper_system.py`

### `DualWhisperSystem.__init__`

The constructor was updated to accept an optional `cpu_manager` argument and pass it to the `FasterWhisperEngine`.

**Before:**
```python
    def __init__(self, prefer_faster_whisper: bool = True):
        self.prefer_faster_whisper = prefer_faster_whisper
        self.faster_whisper_engine = FasterWhisperEngine()
        self.openai_int8_engine = OpenAIWhisperINT8Engine()
```

**After:**
```python
    def __init__(self, prefer_faster_whisper: bool = True, cpu_manager=None):
        self.prefer_faster_whisper = prefer_faster_whisper
        self.faster_whisper_engine = FasterWhisperEngine(cpu_manager=cpu_manager)
        self.openai_int8_engine = OpenAIWhisperINT8Engine()
```

### `FasterWhisperEngine.__init__`

The constructor was updated to accept an optional `cpu_manager` argument and store it.

**Before:**
```python
    def __init__(self):
        self.model = None
        self.model_loaded = False
```

**After:**
```python
    def __init__(self, cpu_manager=None):
        self.model = None
        self.model_loaded = False
        self.cpu_manager = cpu_manager
```

### `FasterWhisperEngine.load_model`

The `load_model` method was updated to use the `cpu_manager` to dynamically set the `cpu_threads` parameter when loading the Whisper model.

**Before:**
```python
            # Load with CPU optimizations for PT-BR
            self.model = WhisperModel(
                WHISPER_MODEL_PATH,  # Load the fine-tuned model from the specified path
                device="cpu",
                compute_type="int8",  # INT8 quantization for speed
                cpu_threads=4,        # OPTIMIZED: Increased to 4 for better parallelism
                download_root=None,
                local_files_only=False
            )
```

**After:**
```python
            cpu_threads = 4
            if self.cpu_manager:
                from src.performance_optimizer import ProcessType
                cpu_threads = self.cpu_manager.get_cores_for_process(ProcessType.TRANSCRIPTION)

            # Load with CPU optimizations for PT-BR
            self.model = WhisperModel(
                WHISPER_MODEL_PATH,  # Load the fine-tuned model from the specified path
                device="cpu",
                compute_type="int8",  # INT8 quantization for speed
                cpu_threads=cpu_threads,
                download_root=None,
                local_files_only=False
            )
```
