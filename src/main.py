import toga
import logging
import asyncio
import os
from threading import RLock
import time
from async_timeout import timeout as async_timeout
from src.audio_processing import AudioRecorder
from src.transcription import transcribe_audio_with_progress
from src.speaker_diarization import SpeakerDiarization
from src.subtitle_generator import generate_srt
from src.file_manager import FileManager
from toga.style import Pack
from toga.style.pack import COLUMN
from config.app_config import MODEL_DIR 

logger = logging.getLogger(__name__)

class TranscrevAI(toga.App):

# Still needs windows/desktop implementation for UI, layout and start/resume/stop processes

    async def startup(self):
        self.processing_pipeline = asyncio.Queue()
        self.worker_task = asyncio.create_task(self.process_tasks())
        FileManager.ensure_directory_exists(FileManager.get_data_path("inputs"))
        FileManager.ensure_directory_exists(FileManager.get_data_path("processed"))
        FileManager.ensure_directory_exists(FileManager.get_data_path("transcripts"))
        self.recorder = AudioRecorder()
        self.processing_lock = RLock()
        # UI 
        self.language_selector = toga.Selection(items=["English", "Portuguese", "Spanish"], style=Pack(padding=5))
        self.start_button = toga.Button("Start Listening", on_press=self.start_recording, style=Pack(padding=5))
        self.pause_button = toga.Button("Pause", on_press=self.pause_recording, style=Pack(padding=5), enabled=False)
        self.stop_button = toga.Button("Stop Listening", on_press=self.stop_recording, style=Pack(padding=5), enabled=False)
        self.transcription_progress = toga.ProgressBar(style=Pack(padding=5))
        self.diarization_progress = toga.ProgressBar(style=Pack(padding=5))
        # Layout UI
        main_box = toga.Box(children=
                                     [self.language_selector,
                                     self.start_button,
                                     self.pause_button,
                                     self.stop_button,
                                     toga.Label("Transcription Progress:", style=Pack(padding=(10, 5))),
                                     self.transcription_progress,
                                     toga.Label("Diarization Progress:", style=Pack(padding=(10, 5))),
                                     self.diarization_progress,],
                            style=Pack(direction=COLUMN, padding=10))
        self.main_window = toga.MainWindow(title="TranscrevAI")
        self.main_window.content = main_box
        self.main_window.show()

    async def _update_button_states(self, is_error=False):
        self.start_button.enabled = is_error
        self.pause_button.enabled = not is_error
        self.stop_button.enabled = not is_error
        await self.main_window.app._impl.loop.run_in_executor(None, lambda: None)

    async def process_tasks(self):
        # Process all pipelined tasks
        while True:
            task = await self.processing_pipeline.get()
            try:
                async with self.processing_lock:
                    if task['type'] == 'recording':
                        logger.info("Starting transcription task")
                        await self._perform_transcription(task)
                        logger.info("Transcription task completed")
                    elif task['type'] == 'diarization':
                        logger.info("Starting diarization task")
                        await self._perform_diarization(task)
                        logger.info("Diarization task completed")
                    else:
                        logger.warning(f"Unknown task type encountered: {task['type']}")
            except Exception as e:
                logger.error(f"Task processing error: {e}")
            finally:
                self.processing_pipeline.task_done()

    async def start_recording(self, widget):
        try:
            await self._update_button_states(False)
            language_map = {"English": ("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip", "en"),
                            "Portuguese": ("https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip", "pt"),
                            "Spanish": ("https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip", "es")}
            selected_language = self.language_selector.value or "English"
            if selected_language not in language_map:
                raise ValueError(f"Unsupported language: {selected_language}")
            
            async with async_timeout(300):
                model_url, lang_code = language_map[selected_language]
                model_path = await FileManager.download_and_extract_model(model_url, lang_code, MODEL_DIR)

            await self.recorder.start_recording()
            # Add tasks to processing pipeline
            await self.processing_pipeline.put({'type': 'recording', 'model': model_path, 'language': lang_code, 'start_time': time.time()})
            await self.processing_pipeline.put({'type': 'diarization', 'audio_file': self.recorder.wav_file, 'language': lang_code})
        except asyncio.TimeoutError:
            logger.error("Model download timeout")
            await self._update_button_states(True)
        except Exception as e:
            logger.error(f"Recording start error: {e}")
            await self._update_button_states(True)
            raise
    
    async def _perform_transcription(self, task):
        # Handle transcription pipeline tasks
        try:
            logger.info(f"Starting transcription for {task['language']}")
            async for progress, transcription_data in transcribe_audio_with_progress(self.recorder.wav_file, task['model'], task['language']):
                self.transcription_progress.value = progress
                # Save incremental transcripts
                if progress % 10 == 0:
                    output_dir = FileManager.get_data_path("transcripts")
                    filename = f"transcript_{int(time.time())}.txt"
                    FileManager.save_transcript(transcription_data, filename=os.path.join(output_dir, filename))
            logger.info("Transcription completed successfully")
            return transcription_data
        except asyncio.CancelledError:
            logger.info("Transcription task cancelled")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise

    async def _perform_diarization(self, task):
        # Handle diarization pipeline tasks
        try:
            logger.info("Starting diarization processing")
            diarizer = SpeakerDiarization()
            segments = await diarizer.diarize_audio(task['audio_file'])
            speaker_ids = {s["speaker"] for s in segments}
            speaker_count = len(speaker_ids)
            logger.info(f"Detected {speaker_count} speakers")
            self.diarization_progress.value = 100
            return segments
        except Exception as e:
            logger.error(f"Diarization processing error: {e}")
            raise

    def pause_recording(self, widget):
        if not self.recorder.is_recording:
            logger.warning("Cannot pause inactive recording")
            return
        self.recorder.pause_recording()
        self.transcription_progress.value = 0
        self.diarization_progress.value = 0

    async def stop_recording(self, widget):
        if not self.recorder.is_recording:
            logger.warning("Cannot stop inactive recording")
            return
        await self.recorder.stop_recording()
        await self._update_button_states(True)
        # Clear pipeline queue
        while not self.processing_pipeline.empty():
            self.processing_pipeline.get_nowait()
            self.processing_pipeline.task_done()

    def validate_model_path(self, language):
        language_map = {"English": ("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"),
                        "Portuguese": ("https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip"),
                        "Spanish": ("https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip")}
        if language not in language_map:
            raise ValueError(f"Unsuported language: {language}")
        try:
            model_path = os.path.join(MODEL_DIR, language)
            required_files = ["am/final.mdl", "conf/model.conf", "graph/phones/word_boundary.int", "graph/Gr.fst", "graph/HCLr.fst", "ivector/final.ie"]
            for file in required_files:
                full_path = os.path.join(model_path, file)
                if not os.path.exists(full_path):
                    raise Exception(f"Missing model component: {file}")
            logger.info(f"Model path validated: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error validating or downloading model: {e}")
            return False

    def main(self):
        # Entry point with async context validation
        if not asyncio.get_event_loop().is_running():
            asyncio.run(self.main_async())
        return TranscrevAI("TranscrevAI", "org.transcrevai")

    async def main_async(self):
        # Async entry point
        await self.startup()

if __name__ == "__main__":
    TranscrevAI().main_loop()