"""
Downloads all required models from Hugging Face Hub into a project-local cache directory (`models/.cache`). Ensures the application can run 100% offline in any environment (local, Docker).

IMPORTANT: Converts all symlinks to real file copies to ensure Docker compatibility.
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables (for HF_TOKEN)
load_dotenv()

# --- Configuration
# Use HF_HOME if already set (e.g., in Docker), otherwise use project-local cache
if 'HF_HOME' in os.environ:
    LOCAL_CACHE_DIR = Path(os.environ['HF_HOME'])
    print(f"Using existing HF_HOME: {LOCAL_CACHE_DIR}")
else:
    LOCAL_CACHE_DIR = Path(__file__).parent.parent / "models" / ".cache"
    os.environ['HF_HOME'] = str(LOCAL_CACHE_DIR)
    print(f"Setting HF_HOME to: {LOCAL_CACHE_DIR}")

print(f"Target local cache directory: {LOCAL_CACHE_DIR}")

# List of all models required
MODELS_TO_DOWNLOAD = [
    "Systran/faster-whisper-medium",
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
]

def convert_symlinks_to_files(cache_dir: Path):
    """
    Converts all symlinks in the cache to real file copies.
    This is crucial for Docker compatibility, as symlinks can break during COPY operations.

    Also removes orphaned blobs directory after conversion to save disk space.
    """
    print("\n" + "="*80)
    print("Converting symlinks to real files for Docker compatibility...")
    print("="*80)

    symlinks_converted = 0
    errors = []
    total_size_copied = 0

    for item in cache_dir.rglob("*"):
        if item.is_symlink():
            try:
                # Get the actual file the symlink points to
                target = item.resolve()

                # Verify target exists
                if not target.exists():
                    errors.append(f"Symlink target missing: {item} -> {target}")
                    continue

                # Get original size for verification
                original_size = target.stat().st_size

                # Remove the symlink
                item.unlink()

                # Copy the real file to replace the symlink
                try:
                    shutil.copy2(target, item)
                except PermissionError:
                    # Fallback for Windows privilege issues
                    print(f"  WARNING: Permission denied on copy2, using copyfile fallback")
                    shutil.copyfile(target, item)

                # Verify the copy was successful
                if not item.exists():
                    errors.append(f"Copy failed - file doesn't exist: {item}")
                    continue

                copied_size = item.stat().st_size
                if original_size != copied_size:
                    errors.append(f"Size mismatch: {item} ({copied_size} bytes vs {original_size} bytes)")
                    continue

                symlinks_converted += 1
                total_size_copied += copied_size

                if symlinks_converted % 10 == 0:
                    print(f"  Converted {symlinks_converted} symlinks ({total_size_copied / 1024**2:.1f} MB)...")

            except Exception as e:
                errors.append(f"Failed to convert {item}: {e}")

    print(f"\n✓ Converted {symlinks_converted} symlinks to real files")
    print(f"  Total data copied: {total_size_copied / 1024**3:.2f} GB")

    # Report errors if any
    if errors:
        print(f"\n⚠️  {len(errors)} errors occurred during conversion:")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

        raise RuntimeError(f"Symlink conversion had {len(errors)} errors. Build cannot continue.")

    # Clean up orphaned blobs directory to save space
    cleanup_orphaned_blobs(cache_dir)

def cleanup_orphaned_blobs(cache_dir: Path):
    """
    Removes the blobs/ directory after symlink conversion.
    After conversion, all files are in snapshots/ and blobs/ is redundant.
    This saves approximately 50% of the cache size.
    """
    print("\n" + "="*80)
    print("Cleaning up orphaned blobs directory...")
    print("="*80)

    blobs_dir = cache_dir / "hub"

    if not blobs_dir.exists():
        print("  No hub directory found, skipping cleanup")
        return

    # Calculate size before cleanup
    total_freed = 0
    blobs_found = []

    for model_dir in blobs_dir.iterdir():
        if model_dir.is_dir():
            blobs_subdir = model_dir / "blobs"
            if blobs_subdir.exists():
                blobs_found.append(blobs_subdir)
                # Calculate size
                for blob_file in blobs_subdir.rglob('*'):
                    if blob_file.is_file():
                        total_freed += blob_file.stat().st_size

    if not blobs_found:
        print("  No blobs directories found, cache already optimized")
        return

    print(f"  Found {len(blobs_found)} blobs directories")
    print(f"  Space to be freed: {total_freed / 1024**3:.2f} GB")

    # Remove all blobs directories
    for blobs_subdir in blobs_found:
        try:
            shutil.rmtree(blobs_subdir)
            print(f"  ✓ Removed: {blobs_subdir.relative_to(cache_dir)}")
        except Exception as e:
            print(f"  ⚠️  Failed to remove {blobs_subdir}: {e}")

    print(f"\n✓ Cleanup complete - freed {total_freed / 1024**3:.2f} GB")
    print("="*80)

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

    # Convert symlinks to real files for Docker compatibility
    convert_symlinks_to_files(LOCAL_CACHE_DIR)

    # No need to patch config - will use local_files_only=True at runtime

def patch_pipeline_config(cache_dir: Path):
    """
    Patches the speaker-diarization pipeline config.yaml to use absolute local paths
    instead of HuggingFace repo IDs. This allows offline loading without tokens.
    """
    print("\n" + "="*80)
    print("Patching pipeline config for offline usage...")
    print("="*80)

    # Locate the pipeline config
    pipeline_dir = cache_dir / "hub" / "models--pyannote--speaker-diarization-3.1"
    snapshots_dir = pipeline_dir / "snapshots"

    if not snapshots_dir.exists():
        print("⚠️  Pipeline not found, skipping config patch")
        return

    # Find pipeline snapshot hash
    pipeline_snapshot = list(snapshots_dir.iterdir())[0]
    config_path = pipeline_snapshot / "config.yaml"

    if not config_path.exists():
        print(f"⚠️  Config not found at {config_path}")
        return

    # Find sub-model snapshot hashes
    segmentation_dir = cache_dir / "hub" / "models--pyannote--segmentation-3.0" / "snapshots"
    embedding_dir = cache_dir / "hub" / "models--pyannote--wespeaker-voxceleb-resnet34-LM" / "snapshots"

    segmentation_snapshot = list(segmentation_dir.iterdir())[0] if segmentation_dir.exists() else None
    embedding_snapshot = list(embedding_dir.iterdir())[0] if embedding_dir.exists() else None

    if not segmentation_snapshot or not embedding_snapshot:
        print("⚠️  Sub-models not found, skipping config patch")
        return

    # Read current config
    with open(config_path, 'r') as f:
        config_content = f.read()

    print(f"  Original config:\n{config_content[:200]}...")

    # Replace repo IDs with absolute paths
    config_content = config_content.replace(
        "segmentation: pyannote/segmentation-3.0",
        f"segmentation: {segmentation_snapshot}"
    )
    config_content = config_content.replace(
        "embedding: pyannote/wespeaker-voxceleb-resnet34-LM",
        f"embedding: {embedding_snapshot}"
    )

    # Write patched config
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✓ Patched config.yaml with local paths:")
    print(f"  segmentation: {segmentation_snapshot}")
    print(f"  embedding: {embedding_snapshot}")
    print("="*80)

if __name__ == "__main__":
    print("Starting download of all required models into the project-local cache...")
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    download_models()
    print("\nAll model downloads attempted.")