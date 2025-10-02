"""
Modern Transcription Service for TranscrevAI
Integrates Dual Whisper System (faster-whisper + openai-whisper INT8)
CPU-only architecture with PT-BR optimization
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional

from src.logging_setup import setup_app_logging
logger = setup_app_logging(logger_name="transcrevai.transcription")

# Import dual whisper system
from dual_whisper_system import DualWhisperSystem, TranscriptionResult


class TranscriptionService:
    """
    Modern transcription service using dual whisper engines
    """

    def __init__(self, cpu_manager=None):
        self.dual_system = DualWhisperSystem(prefer_faster_whisper=True, cpu_manager=cpu_manager)
        logger.info("TranscriptionService initialized with dual whisper system")

    async def transcribe_audio_file(self, audio_path: str, language: str = "pt", domain: str = "general") -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file using dual whisper system
        """
        try:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return {"error": "Audio file not found"}

            logger.info(f"Starting transcription: {audio_path} (Domain: {domain})")
            start_time = time.time()

            # Use dual whisper system, passing the domain
            result: TranscriptionResult = self.dual_system.transcribe(audio_path, domain=domain)

            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f}s using {result.system_used}")

            # Convert to expected format
            return {
                "text": result.text,
                "language": result.language,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "memory_used_mb": result.memory_used_mb,
                "segments": result.segments,
                "system_used": result.system_used,
                "model_name": result.model_name,
                "success": True
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"error": f"Transcription failed: {str(e)}", "success": False}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from dual system"""
        return self.dual_system.get_performance_summary()

    def reload_models(self):
        """Force reload models with updated configurations (CORREÇÃO 2.4)"""
        logger.info("Forcing model reload in TranscriptionService...")
        self.dual_system.reload_models()
        logger.info("Model reload complete")

    def cleanup(self):
        """Clean up resources"""
        try:
            # Cleanup is handled automatically by dual system
            logger.info("TranscriptionService cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


class OptimizedTranscriber:
    """
    Legacy compatibility wrapper for existing code
    """

    def __init__(self, cpu_manager=None):
        self.service = TranscriptionService(cpu_manager=cpu_manager)
        logger.info("OptimizedTranscriber initialized as wrapper")

    def transcribe_parallel(self, audio_path: str, domain: str = "general") -> Dict:
        """Legacy interface for parallel transcription"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.service.transcribe_audio_file(audio_path, domain=domain)
            )
            return result or {}
        finally:
            loop.close()


class TranscriptionProcess:
    """
    Legacy compatibility wrapper for process-based transcription
    """

    def __init__(self):
        self.service = TranscriptionService()
        logger.info("TranscriptionProcess initialized as wrapper")

    def transcribe_with_process_isolation(self, audio_path: str) -> Dict:
        """Legacy interface with process isolation"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.service.transcribe_audio_file(audio_path)
            )
            return result or {}
        finally:
            loop.close()


# Factory functions for backward compatibility
def create_transcription_service() -> TranscriptionService:
    """Create modern transcription service"""
    return TranscriptionService()

def create_optimized_transcriber() -> OptimizedTranscriber:
    """Create legacy wrapper for existing code"""
    return OptimizedTranscriber()

def create_transcription_process() -> TranscriptionProcess:
    """Create legacy process wrapper"""
    return TranscriptionProcess()