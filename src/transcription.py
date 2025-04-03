import wave
import json
import logging
import os
import requests
import zipfile
from src.file_manager import FileManager
from vosk import Model, KaldiRecognizer
from config.app_config import MODEL_DIR 

logger = logging.getLogger(__name__)

def load_language_model(language_code):
    model_path = f"models/{language_code}"
    if not is_model_available(model_path):
        raise FileNotFoundError(f"Model for {language_code} not found.")
    return Model(model_path)

def download_extract_model(url, output_dir):
    response = requests.get(url, stream=True)
    zip_path = os.path.join(output_dir, "model.zip")
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            member.filename = os.path.basename(member.filename)  # no 'zip slip' vulnerability
            zip_ref.extract(member, output_dir)
    os.remove(zip_path)

def is_model_available(model_path):
    required_files = ["am/final.mdl", "conf/model.conf", "graph/phones/word_boundary.int", "graph/Gr.fst", "graph/HCLr.fst", "ivector/final.ie"]
    return all(os.path.exists(os.path.join(model_path, file)) for file in required_files)

def _parse_result(result):
    return {"start": result.get("result", [{}])[0].get("start", 0),
            "end": result.get("result", [{}])[-1].get("end", 0),
            "text": result.get("text", "")}

async def transcribe_audio_with_progress(wav_file, model_path, language_code, sample_rate=16000):
    try:
        logger.info(f"Loading model from {model_path}")
        model_dir = os.path.join(FileManager.get_base_directory(), "models")
        model_path = os.path.join(model_dir, language_code)
        model = Model(os.path.join(MODEL_DIR, language_code))
        with wave.open(wav_file, "rb") as wf:
            recognizer = KaldiRecognizer(model, wf.getframerate())
            total_frames = wf.getnframes()
            if total_frames == 0:
                raise ValueError("Audio file is empty.")
            processed_frames = 0
            transcription_data = []
            while True:
                data = wf.readframes(4096)
                if len(data) == 0:
                    break
                processed_frames += len(data)
                progress_percentage = min(100, int((processed_frames / total_frames) * 100))
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    transcription_data.append(_parse_result(result))
                yield progress_percentage, transcription_data
            final_result = json.loads(recognizer.FinalResult())
            transcription_data.append(_parse_result(final_result))
            yield 100, transcription_data
            logger.info(f"Transcription completed for {wav_file}")
    except FileNotFoundError as e:
        logger.error(f"Audio file not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid audio file: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Error loading Vosk model: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise