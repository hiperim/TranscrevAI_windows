# Suggestion: Add Optional LLM-Based Post-Processing

**Files:**
- `config/app_config.py`
- `src/post_processing.py` (New file)
- `src/performance_optimizer.py`

**Change:**
1. A new file, `src/post_processing.py`, will be created to contain the logic for calling an external LLM (e.g., OpenAI's GPT) to correct and refine a transcription.
2. A `POST_PROCESSING_CONFIG` section will be added to `config/app_config.py` to enable/disable this feature and to hold the necessary API key.
3. The main processing pipeline in `src/performance_optimizer.py` will be updated to optionally call this new LLM correction function *after* the transcription is complete.

**Justification:**
Even highly accurate ASR models like Whisper can produce transcriptions with minor grammatical errors, awkward phrasing, or incorrect punctuation, especially in complex sentences.

- **Ultimate Accuracy:** Using a powerful LLM as a final editing pass can correct these subtle errors, resulting in a near-perfect, human-readable transcript. It can remove filler words, fix sentence structure, and ensure correct punctuation.
- **Performance-Aware Implementation:** This is a computationally expensive step involving a network request to an external service. To avoid impacting the 1x1 processing ratio goal, it will be implemented as a **strictly optional feature** that runs **after** the core transcription pipeline is finished. The real-time goal is met first, and this provides an additional layer of quality for those who need it and can accommodate the extra processing time.
- **Security:** The API key will be handled securely by loading it from environment variables, not hardcoding it in the source.
