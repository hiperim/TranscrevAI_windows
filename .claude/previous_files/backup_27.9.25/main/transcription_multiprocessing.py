# Specialized transcription function for multiprocessing architecture
def process_audio_file_multiprocessing(audio_file_path: str, language: str = "pt"):
    """
    Multiprocessing-specific transcription function.
    Optimized for single-process execution within multiprocessing worker.
    """
    import json
    import numpy as np
    from pathlib import Path
    from src.logging_setup import setup_app_logging

    worker_logger = setup_app_logging(logger_name="transcrevai.multiprocessing_transcription")

    try:
        # Import and instantiate OptimizedTranscriber within the worker process
        from src.transcription import OptimizedTranscriber
        from src.audio_processing import RobustAudioLoader

        transcriber = OptimizedTranscriber(model_name="medium")
        audio_loader = RobustAudioLoader()

        worker_logger.info(f"Loading audio file: {audio_file_path}")

        # Load audio data directly
        audio_data = audio_loader.load_audio(audio_file_path)
        if audio_data is None:
            worker_logger.error(f"Failed to load audio: {audio_file_path}")
            return False

        worker_logger.info(f"Processing audio file: {audio_file_path} (duration: {len(audio_data)/16000:.2f}s)")

        # Use single-process transcription optimized for multiprocessing worker
        # Split into reasonable chunks for this specific worker
        chunk_duration = 30  # seconds - optimal for single worker
        sample_rate = 16000
        chunk_size = chunk_duration * sample_rate

        results = []
        total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)

        for i in range(0, len(audio_data), chunk_size):
            chunk_data = audio_data[i:i+chunk_size]
            chunk_start_time = i / sample_rate

            worker_logger.info(f"Processing chunk {i//chunk_size + 1}/{total_chunks}")

            # Process single chunk in this worker
            chunk_result = transcriber.transcribe_chunk((
                audio_file_path,
                chunk_start_time,
                chunk_start_time + len(chunk_data)/sample_rate
            ))

            if chunk_result and 'text' in chunk_result:
                results.append({
                    'start': chunk_start_time,
                    'end': chunk_start_time + len(chunk_data)/sample_rate,
                    'text': chunk_result['text'],
                    'confidence': chunk_result.get('confidence', 0.0)
                })

        # Combine results
        if results:
            combined_result = {
                'transcription': results,
                'full_text': ' '.join([r['text'] for r in results if r['text'].strip()]),
                'duration': len(audio_data) / sample_rate,
                'chunks_processed': len(results),
                'processing_method': 'multiprocessing_single_worker'
            }

            # Save result to JSON file
            output_file = str(audio_file_path).replace(".wav", "_transcription_result.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(combined_result, f, ensure_ascii=False, indent=2)

            worker_logger.info(f"Transcription completed: {len(results)} chunks, output: {output_file}")
            return True
        else:
            worker_logger.error(f"No transcription results for: {audio_file_path}")
            return False

    except Exception as e:
        worker_logger.error(f"Error in multiprocessing transcription {audio_file_path}: {e}")
        import traceback
        worker_logger.debug(f"Traceback: {traceback.format_exc()}")
        return False