from typing import Optional

class TranscrevAIError(Exception):
    """Base exception for all TranscrevAI errors"""
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
    """Raised when audio transcription fails"""
    pass


class DiarizationError(TranscrevAIError):
    """Raised when speaker diarization fails"""
    pass


class AudioProcessingError(TranscrevAIError):
    """Raised when audio processing operations fail"""
    pass


class SessionError(TranscrevAIError):
    """Raised for session management errors"""
    pass


class ValidationError(TranscrevAIError):
    """Raised when input validation fails"""
    pass
