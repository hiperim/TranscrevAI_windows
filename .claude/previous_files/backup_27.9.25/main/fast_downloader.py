# Fast Multi-threaded Downloader for Whisper Models
import os
import requests
import threading
import hashlib
import time
from pathlib import Path
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.logging_setup import setup_app_logging

logger = setup_app_logging(logger_name="transcrevai.fast_downloader")

class FastDownloader:
    """Multi-threaded downloader with resume support"""

    def __init__(self, num_threads: int = 8):
        self.num_threads = num_threads
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TranscrevAI/1.0 (Fast Downloader)'
        })

    def supports_range_requests(self, url: str) -> bool:
        """Check if server supports range requests"""
        try:
            response = self.session.head(url, timeout=10)
            accept_ranges = response.headers.get('Accept-Ranges', '').lower()
            content_length = response.headers.get('Content-Length')

            supports_ranges = 'bytes' in accept_ranges
            has_length = content_length is not None

            logger.info(f"Range support: {supports_ranges}, Content-Length: {content_length}")
            return supports_ranges and has_length

        except Exception as e:
            logger.warning(f"Error checking range support: {e}")
            return False

    def get_file_size(self, url: str) -> Optional[int]:
        """Get file size from server"""
        try:
            response = self.session.head(url, timeout=10)
            content_length = response.headers.get('Content-Length')
            return int(content_length) if content_length else None
        except Exception as e:
            logger.error(f"Error getting file size: {e}")
            return None

    def download_chunk(self, url: str, start: int, end: int, chunk_file: str) -> bool:
        """Download a specific chunk of the file"""
        try:
            headers = {'Range': f'bytes={start}-{end}'}
            response = self.session.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            with open(chunk_file, 'wb') as f:
                for data in response.iter_content(chunk_size=8192):
                    if data:
                        f.write(data)

            actual_size = os.path.getsize(chunk_file)
            expected_size = end - start + 1

            if actual_size != expected_size:
                logger.warning(f"Chunk size mismatch: {actual_size} != {expected_size}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error downloading chunk {start}-{end}: {e}")
            return False

    def merge_chunks(self, chunk_files: list, output_file: str) -> bool:
        """Merge downloaded chunks into final file"""
        try:
            with open(output_file, 'wb') as outfile:
                for chunk_file in chunk_files:
                    if not os.path.exists(chunk_file):
                        logger.error(f"Missing chunk file: {chunk_file}")
                        return False

                    with open(chunk_file, 'rb') as infile:
                        outfile.write(infile.read())

                    # Clean up chunk file
                    os.remove(chunk_file)

            logger.info(f"Successfully merged {len(chunk_files)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error merging chunks: {e}")
            return False

    def download_single_threaded(self, url: str, output_file: str,
                                progress_callback: Optional[Callable] = None) -> bool:
        """Fallback single-threaded download"""
        try:
            logger.info(f"Single-threaded download: {url}")
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0

            with open(output_file, 'wb') as f:
                for data in response.iter_content(chunk_size=8192):
                    if data:
                        f.write(data)
                        downloaded += len(data)

                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(progress, downloaded, total_size)

            logger.info(f"Single-threaded download completed: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Single-threaded download failed: {e}")
            return False

    def download_multi_threaded(self, url: str, output_file: str,
                              progress_callback: Optional[Callable] = None) -> bool:
        """Multi-threaded download with range requests"""

        # Get file size
        file_size = self.get_file_size(url)
        if not file_size:
            logger.warning("Cannot determine file size, falling back to single-threaded")
            return self.download_single_threaded(url, output_file, progress_callback)

        # Check range support
        if not self.supports_range_requests(url):
            logger.warning("Server doesn't support range requests, falling back to single-threaded")
            return self.download_single_threaded(url, output_file, progress_callback)

        logger.info(f"Multi-threaded download: {file_size} bytes with {self.num_threads} threads")

        # Calculate chunk sizes
        chunk_size = file_size // self.num_threads
        chunks = []
        chunk_files = []

        for i in range(self.num_threads):
            start = i * chunk_size
            end = start + chunk_size - 1

            # Last chunk gets remainder
            if i == self.num_threads - 1:
                end = file_size - 1

            chunk_file = f"{output_file}.part{i}"
            chunks.append((start, end, chunk_file))
            chunk_files.append(chunk_file)

        # Download chunks in parallel
        start_time = time.time()
        success_count = 0

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_chunk = {
                executor.submit(self.download_chunk, url, start, end, chunk_file): (start, end, chunk_file)
                for start, end, chunk_file in chunks
            }

            for future in as_completed(future_to_chunk):
                start, end, chunk_file = future_to_chunk[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        logger.info(f"Chunk {start}-{end} completed ({success_count}/{self.num_threads})")
                    else:
                        logger.error(f"Chunk {start}-{end} failed")

                except Exception as e:
                    logger.error(f"Chunk {start}-{end} exception: {e}")

        download_time = time.time() - start_time

        # Check if all chunks downloaded successfully
        if success_count != self.num_threads:
            logger.error(f"Only {success_count}/{self.num_threads} chunks succeeded")
            # Clean up partial chunks
            for chunk_file in chunk_files:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            return False

        # Merge chunks
        if not self.merge_chunks(chunk_files, output_file):
            return False

        # Verify final file size
        actual_size = os.path.getsize(output_file)
        if actual_size != file_size:
            logger.error(f"Final file size mismatch: {actual_size} != {file_size}")
            return False

        speed_mbps = (file_size / 1024 / 1024) / download_time * 8
        logger.info(f"Multi-threaded download completed in {download_time:.1f}s ({speed_mbps:.1f} Mbps)")

        return True

    def download(self, url: str, output_file: str,
                progress_callback: Optional[Callable] = None) -> bool:
        """Main download method with automatic fallback"""

        # Create output directory if needed
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Try multi-threaded first
        logger.info(f"Starting download: {url}")
        start_time = time.time()

        success = self.download_multi_threaded(url, output_file, progress_callback)

        if not success:
            logger.warning("Multi-threaded download failed, trying single-threaded")
            success = self.download_single_threaded(url, output_file, progress_callback)

        if success:
            total_time = time.time() - start_time
            file_size = os.path.getsize(output_file)
            speed_mbps = (file_size / 1024 / 1024) / total_time * 8
            logger.info(f"Download successful: {output_file} ({speed_mbps:.1f} Mbps)")

        return success


def progress_callback(percent: float, downloaded: int, total: int):
    """Sample progress callback"""
    mb_downloaded = downloaded / 1024 / 1024
    mb_total = total / 1024 / 1024
    print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)


# Test function
def test_fast_downloader():
    """Test the fast downloader"""
    downloader = FastDownloader(num_threads=4)
    test_url = "https://github.com/openai/whisper/raw/main/README.md"
    output_file = "/tmp/claude/test_download.md"

    success = downloader.download(test_url, output_file, progress_callback)
    print(f"\nTest download {'succeeded' if success else 'failed'}")

    if success and os.path.exists(output_file):
        print(f"File size: {os.path.getsize(output_file)} bytes")


if __name__ == "__main__":
    test_fast_downloader()