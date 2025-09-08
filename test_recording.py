#!/usr/bin/env python3
"""Test script to verify transcription and diarization accuracy"""

import asyncio
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_latest_recording():
    """Test the latest recording with enhanced TranscrevAI"""
    
    # Find the latest recording
    recordings_dir = Path("C:/TranscrevAI_windows/data/recordings")
    if not recordings_dir.exists():
        logger.error("Recordings directory not found")
        return
    
    # Get WAV files only (MP4 files have format issues with soundfile)
    recording_files = list(recordings_dir.glob("recording_*.wav"))
    if not recording_files:
        logger.error("No WAV recording files found")
        return
    
    # Get the most recent WAV recording
    latest_recording = max(recording_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Testing with latest recording: {latest_recording}")
    
    try:
        # Import the enhanced concurrent processor
        from src.concurrent_engine import concurrent_processor
        
        # Mock websocket manager for testing
        class MockWebSocketManager:
            async def send_message(self, session_id, message):
                logger.info(f"Progress: {message.get('type', 'unknown')} - {message.get('message', '')}")
        
        websocket_manager = MockWebSocketManager()
        
        # Test with Portuguese, conversation type, balanced profile
        logger.info("Starting enhanced concurrent processing...")
        result = await concurrent_processor.process_audio_concurrent(
            session_id="test_session",
            audio_file=str(latest_recording),
            language="pt", 
            websocket_manager=websocket_manager,
            audio_input_type="conversation",
            processing_profile="balanced"
        )
        
        # Extract results
        transcription_data = result.get("transcription_data", [])
        diarization_segments = result.get("diarization_segments", [])
        speakers_detected = result.get("speakers_detected", 0)
        complexity = result.get("complexity", "unknown")
        quality_metrics = result.get("quality_metrics", {})
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRANSCRIPTION AND DIARIZATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Audio complexity: {complexity}")
        logger.info(f"Speakers detected: {speakers_detected}")
        logger.info(f"Quality score: {quality_metrics.get('overall_quality', 0):.2f}")
        logger.info(f"Transcription segments: {len(transcription_data)}")
        logger.info(f"Diarization segments: {len(diarization_segments)}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FULL TRANSCRIPTION WITH SPEAKERS")
        logger.info(f"{'='*60}")
        
        if transcription_data:
            for segment in transcription_data:
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '').strip()
                confidence = segment.get('confidence', 0)
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                
                if text:
                    logger.info(f"{speaker}: {text} (confidence: {confidence:.2f}, time: {start:.1f}s-{end:.1f}s)")
        else:
            logger.warning("No transcription data available")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DIARIZATION DETAILS")
        logger.info(f"{'='*60}")
        
        if diarization_segments:
            for segment in diarization_segments:
                speaker = segment.get('speaker', 'Unknown')
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                confidence = segment.get('confidence', 0)
                
                logger.info(f"{speaker}: {start:.1f}s-{end:.1f}s (confidence: {confidence:.2f})")
        else:
            logger.warning("No diarization data available")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"QUALITY METRICS DETAILS") 
        logger.info(f"{'='*60}")
        
        for metric, value in quality_metrics.items():
            if metric == "detailed_metrics":
                logger.info("Detailed metrics:")
                for key, val in value.items():
                    logger.info(f"  {key}: {val}")
            else:
                logger.info(f"{metric}: {value}")
        
        # Expected vs Actual comparison
        logger.info(f"\n{'='*60}")
        logger.info(f"ACCURACY ASSESSMENT")
        logger.info(f"{'='*60}")
        
        expected_speakers = 2
        expected_content_keywords = ["Rogério", "primeiro dia", "empresa", "hierarquia", "claro", "desculpa"]
        
        if speakers_detected == expected_speakers:
            logger.info("✅ Speaker detection: CORRECT (detected 2 speakers as expected)")
        else:
            logger.warning(f"❌ Speaker detection: Expected {expected_speakers}, got {speakers_detected}")
        
        # Check for expected content
        full_text = " ".join(seg.get('text', '') for seg in transcription_data).lower()
        found_keywords = [kw for kw in expected_content_keywords if kw.lower() in full_text]
        
        logger.info(f"Content keywords found: {len(found_keywords)}/{len(expected_content_keywords)}")
        logger.info(f"Found keywords: {found_keywords}")
        
        if len(found_keywords) >= len(expected_content_keywords) // 2:
            logger.info("✅ Content accuracy: GOOD (found expected keywords)")
        else:
            logger.warning("❌ Content accuracy: POOR (missing expected keywords)")
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_latest_recording())