# Suggestion: Make `condition_on_previous_text` Adaptive

**File:** `dual_whisper_system.py`

**Change:** In the `FasterWhisperEngine.transcribe` method, the `condition_on_previous_text` parameter will be changed from being hardcoded to `False` to being adaptive, set to `not use_vad`.

**Justification:**
Hardcoding `condition_on_previous_text` to `False` is a safe way to prevent errors in long audio with silences, but it sacrifices accuracy in shorter, continuous audio segments by discarding valuable context.

This change makes the parameter's behavior dependent on the existing Voice Activity Detection (VAD) strategy:

- **For long audio (`use_vad=True`):** `condition_on_previous_text` will be `False`. This maintains the current behavior, preventing the model from getting stuck in repetitive loops after silent periods detected by VAD.
- **For short/medium audio (`use_vad=False`):** `condition_on_previous_text` will be `True`. This allows the model to use the previous transcription segment as context, significantly improving coherence and accuracy for continuous speech.

This adaptive approach provides the best of both worlds: robustness for long files and higher accuracy for shorter ones.
