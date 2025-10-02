# Suggestion: Implement Dynamic Initial Prompts with UI Selection

**Files:**
- `config/app_config.py`
- `templates/index.html`
- `main.py`
- `src/performance_optimizer.py`
- `src/transcription.py`
- `dual_whisper_system.py`

**Change:**
This change introduces the ability to dynamically select a Whisper `initial_prompt` based on a user-provided `domain` chosen from a new dropdown menu in the UI.

1.  A dropdown for domain selection will be added to `index.html` with the label "Favor selecionar assunto geral do áudio a ser processado:".
2.  The `ADAPTIVE_PROMPTS` dictionary in the config will be expanded with more domains to populate this dropdown.
3.  The API in `main.py` will be updated to accept the `domain` parameter from the UI.
4.  The `domain` parameter will be passed through the entire processing pipeline, from the API layer to the transcription engine.
5.  The transcription engine will use the `domain` to select the appropriate prompt, improving contextual accuracy.

**Justification:**
A static `initial_prompt` is not optimal. Allowing the user to specify the audio's domain (e.g., `medical`, `legal`, `finance`) provides Whisper with powerful contextual hints.

- **Accuracy:** This significantly improves accuracy for audio with specialized terminology by "priming" the model with the correct vocabulary.
- **User Experience:** Adding a simple dropdown in the UI makes this powerful feature easily accessible.
- **Performance:** This is a logic-based change that occurs before transcription. It has **no negative impact on processing speed**, making it a pure win for accuracy and usability.
