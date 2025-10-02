# Standalone function for audio transcription in workers
def process_audio_file_standalone(audio_file_path: str, language: str = "pt"):
    """
    Standalone function to process audio file within worker process.
    Avoids multiprocessing serialization issues by creating OptimizedTranscriber locally.
    """
    import asyncio
    import json
    from pathlib import Path
    from src.logging_setup import setup_app_logging

    worker_logger = setup_app_logging(logger_name="transcrevai.transcription_standalone")

    try:
        # Import and instantiate OptimizedTranscriber within the worker process
        from src.transcription import OptimizedTranscriber
        transcriber = OptimizedTranscriber(model_name="medium")

        worker_logger.info(f"Processing audio file: {audio_file_path}")

        # Create event loop for async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Execute transcription
            result = loop.run_until_complete(
                transcriber.transcribe_parallel(
                    audio_file_path
                    
                )
            )

            if result:
                # Save result to JSON file
                output_file = str(audio_file_path).replace(".wav", "_transcription_result.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                worker_logger.info(f"Transcription completed: {output_file}")
                return True
            else:
                worker_logger.error(f"Transcription failed for: {audio_file_path}")
                return False

        finally:
            loop.close()

    except Exception as e:
        worker_logger.error(f"Error processing {audio_file_path}: {e}")
        return False