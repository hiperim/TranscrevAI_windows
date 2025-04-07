import logging
import asyncio
import os
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
from src.audio_processing import AudioRecorder
from src.transcription import transcribe_audio_with_progress
from src.speaker_diarization import SpeakerDiarization
from src.file_manager import FileManager
from src.subtitle_generator import generate_srt
from config.app_config import MODEL_DIR

logger = logging.getLogger(__name__)
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state (simple implementation for single user)
app_state = {"recording": False, "paused": False, "progress": {"transcription": 0, "diarization": 0}, "current_task": None}

@app.route("/")
def index():
    return render_template("gui.html")

@socketio.on("start")
def handle_start(language):
    if app_state["recording"]:
        return
    app_state["recording"] = True
    app_state["paused"] = False
    socketio.start_background_task(processing_pipeline_wrapper, language)

async def processing_pipeline(language):
    try:
        recorder = AudioRecorder()
        language_map = {"en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
                        "pt": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
                        "es": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"}
        model_url = language_map.get(language)
        if not model_url:
            raise ValueError(f"Unsupported language: {language}")

        model_path = await FileManager.download_and_extract_model(model_url, language, MODEL_DIR)
        await recorder.start_recording()

        # Transcription w/ async
        transcription = []
        async for progress, data in transcribe_audio_with_progress(recorder.wav_file, model_path, language):
            transcription.extend(data)
            socketio.emit("progress", {"transcription": progress, "diarization": app_state["progress"]["diarization"]})

        # Diarization
        diarizer = SpeakerDiarization()
        diarization_segments = await diarizer.diarize_audio(recorder.wav_file)
        socketio.emit("progress", {"transcription": 100, "diarization": 100})
        segments = []
        total_segments = len(diarization_segments)  # Get from your diarization logic

        for idx, segment in enumerate(diarization_segments):
            segments.append(segment)
            progress = int((idx + 1) / total_segments * 100)
            socketio.emit('progress', {"transcription": 100, "diarization": progress})

        # Generate subtitles
        output_dir = FileManager.get_data_path("transcripts")
        srt_path = os.path.join(output_dir, f"transcript_{int(time.time())}.srt")
        await generate_srt(transcription, diarization_segments, srt_path)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        socketio.emit("error", str(e))
    finally:
        app_state["recording"] = False
        await recorder.stop_recording()

def processing_pipeline_wrapper(language):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(processing_pipeline(language))
    except Exception as e:
        logger.error(f"Pipeline wrapper error: {str(e)}")
        socketio.emit("error", str(e))
    finally:
        loop.close()

@socketio.on("pause")
def handle_pause():
    app_state["paused"] = not app_state["paused"]

@socketio.on("stop")
def handle_stop():
    app_state["recording"] = False
    app_state["paused"] = False

if __name__ == "__main__":
    socketio.run(app, debug=False, port=5000)