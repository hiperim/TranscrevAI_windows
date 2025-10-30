"""Custom exception classes for TranscrevAI.

Provides granular error types for better error handling and debugging.
All custom exceptions inherit from TranscrevAIError base class.
"""

from typing import Optional

class TranscrevAIError(Exception):
    """Base exception for all TranscrevAI errors.

    Attributes:
        message: Human-readable error description
        context: Additional error context (session_id, file_path, etc.)
    """
    def __init__(self, message: str, context: Optional[dict] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class TranscriptionError(TranscrevAIError):
    """Raised when audio transcription fails.

    Common causes:
    - Model loading failure
    - Invalid audio format
    - Corrupted audio file
    - Insufficient memory
    """
    pass


class DiarizationError(TranscrevAIError):
    """Raised when speaker diarization fails.

    Common causes:
    - Pyannote pipeline failure
    - Speaker alignment errors
    - Audio too short for diarization
    """
    pass


class AudioProcessingError(TranscrevAIError):
    """Raised when audio processing operations fail.

    Common causes:
    - Invalid audio format
    - Audio quality issues (too quiet, clipping)
    - File read/write errors
    """
    pass


class SessionError(TranscrevAIError):
    """Raised for session management errors.

    Common causes:
    - Session not found
    - Session already exists
    - Session timeout
    """
    pass


class ValidationError(TranscrevAIError):
    """Raised when input validation fails.

    Common causes:
    - File too large
    - Invalid format
    - Missing required fields
    """
    pass
