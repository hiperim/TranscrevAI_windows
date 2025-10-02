# Suggestion: Integrate Fine-Tuned Portuguese Model

**Files:**
- `dev_tools/convert_model.py` (New file)
- `config/app_config.py`
- `dual_whisper_system.py`

**Change:**
1. A new script, `dev_tools/convert_model.py`, will be created. This script will download a specific Portuguese fine-tuned Whisper model from Hugging Face and convert it to the CTranslate2 format required by `faster-whisper`.
2. `config/app_config.py` will be updated with a `FINE_TUNED_MODEL_PATH` to specify the location of the converted model.
3. `dual_whisper_system.py` will be modified to load the model from this local path instead of loading the generic `"medium"` model by name.

**Justification:**
While tuning parameters helps, the most significant accuracy gains come from using a model that has been specifically fine-tuned on a relevant dataset. Standard Whisper models are trained on a wide variety of languages, but a model fine-tuned exclusively on Portuguese will have a much better understanding of the language's specific phonetics, vocabulary, and grammar.

- **Accuracy:** This is the most direct way to achieve a state-of-the-art Word Error Rate (WER) for Portuguese transcription.
- **Performance:** By converting a fine-tuned `medium` model, we retain the same performance characteristics and memory footprint as the generic `medium` model. This change provides a major accuracy boost with a negligible impact on processing speed, directly supporting the goal of achieving a 1x1 processing ratio.

This change moves the application from using a general-purpose model to a specialized, high-performance one.
