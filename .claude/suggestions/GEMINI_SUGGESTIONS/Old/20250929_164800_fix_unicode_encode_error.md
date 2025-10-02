# Fix: UnicodeEncodeError in Logging

**File:** `src/performance_optimizer.py`

**Change:** Removed an emoji character from a `logger.info` message within the `MultiProcessingTranscrevAI.__init__` method.

**Justification:**
The original logging message included an emoji (🔐) which caused a `UnicodeEncodeError` when the application attempted to print to a console that did not support UTF-8 encoding (a common issue on default Windows consoles). This error prevented the application from initializing correctly.

Removing the emoji resolves this encoding issue, ensuring that logging messages can be displayed without errors across different console environments.
