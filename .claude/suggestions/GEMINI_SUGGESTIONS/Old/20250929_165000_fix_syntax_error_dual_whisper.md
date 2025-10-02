# Fix: SyntaxError in dual_whisper_system.py

**File:** `dual_whisper_system.py`

**Change:** Corrected the f-string formatting in a `logger.info` call on line 382 (and potentially line 141, if it was a similar issue) within the `OpenAIWhisperINT8Engine.transcribe` method.

**Justification:**
The original f-string used single quotes as delimiters while also containing single quotes within the string literal (e.g., `f"... '{variable}' ..."`). This caused a `SyntaxError: unterminated string literal` because the inner single quotes prematurely terminated the f-string.

The fix involves changing the outer delimiters of the f-string to single quotes and using double quotes for any embedded string literals, or vice-versa, to ensure proper parsing and avoid syntax errors.
