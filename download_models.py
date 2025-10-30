#!/usr/bin/env python3
"""Download Pyannote models using the environment's default cache path."""
import os
from huggingface_hub import snapshot_download

token = os.environ.get('HUGGING_FACE_HUB_TOKEN')

# By REMOVING the `cache_dir` parameter, `snapshot_download` will respect the
# HF_HOME environment variable set in the Dockerfile. This ensures it downloads
# to the modern `/hub` subdirectory, matching where `hf_hub_download` looks at runtime.

print("Downloading pyannote/speaker-diarization-3.1 (all files)...")
snapshot_download(
    repo_id='pyannote/speaker-diarization-3.1',
    token=token,
    allow_patterns=["*"],
    local_dir_use_symlinks=False
)

print("Downloading pyannote/segmentation-3.0 (all files)...")
snapshot_download(
    repo_id='pyannote/segmentation-3.0',
    token=token,
    allow_patterns=["*"],
    local_dir_use_symlinks=False
)

print("Downloading pyannote/wespeaker-voxceleb-resnet34-LM (embedding model)...")
snapshot_download(
    repo_id='pyannote/wespeaker-voxceleb-resnet34-LM',
    token=token,
    allow_patterns=["*"],
    local_dir_use_symlinks=False
)

print('\n✓ Pyannote models pre-downloaded to the correct cache directory.')

# Optional: You can still print the cache structure to verify
hf_home = os.getenv("HF_HOME", "/root/.cache/huggingface")
print(f'\nVerifying cache structure in: {hf_home}\n')
for root, dirs, files in os.walk(hf_home):
    level = root.replace(hf_home, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    # Show first 3 files in each directory
    for file in sorted(files)[:3]:
        print(f'{subindent}{file}')
    if len(files) > 3:
        print(f'{subindent}... and {len(files)-3} more files')
