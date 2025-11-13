"""
Downloads all required models from Hugging Face Hub into a project-local cache directory (`models/.cache`). Ensures the application can run 100% offline in any environment (local, Docker).
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables (for HF_TOKEN)
load_dotenv()

# --- Configuration
LOCAL_CACHE_DIR = Path(__file__).parent.parent / "models" / ".cache"
os.environ['HF_HOME'] = str(LOCAL_CACHE_DIR)

print(f"Target local cache directory set via HF_HOME: {LOCAL_CACHE_DIR}")

# List of all models required
MODELS_TO_DOWNLOAD = [
    "Systran/faster-whisper-medium",
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
]

def download_models():
    """Downloads and saves all specified models to the local cache directory."""
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("WARNING: HUGGING_FACE_HUB_TOKEN not found in .env file. Downloads may fail for gated models.")

    for repo_id in MODELS_TO_DOWNLOAD:
        print("-" * 80)
        print(f"Downloading model: {repo_id}")
        
        try:
            # cache_dir forces downloads to project-local cache
            snapshot_download(
                repo_id=repo_id,
                token=hf_token,
                resume_download=True
            )
            print(f"Successfully downloaded {repo_id} to local cache.")
        except Exception as e:
            print(f"FAILED to download {repo_id}. Error: {e}")
            print("Please check your internet connection and Hugging Face token.")

if __name__ == "__main__":
    print("Starting download of all required models into the project-local cache...")
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    download_models()
    print("\nAll model downloads attempted.")