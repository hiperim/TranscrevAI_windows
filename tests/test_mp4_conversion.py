
import unittest
from pathlib import Path
import os
import wave
import struct

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processing import convert_wav_to_mp4

class TestMP4Conversion(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path("tests") / "temp_test_files"
        self.temp_dir.mkdir(exist_ok=True)
        self.wav_path = self.temp_dir / "test.wav"
        self.mp4_path = self.temp_dir / "test.mp4"

        # Create a dummy WAV file
        with wave.open(str(self.wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            for _ in range(16000 * 2): # 2 seconds of silence
                wf.writeframes(struct.pack('<h', 0))

    def tearDown(self):
        if self.wav_path.exists():
            self.wav_path.unlink()
        if self.mp4_path.exists():
            self.mp4_path.unlink()
        if self.temp_dir.exists():
            self.temp_dir.rmdir()

    def test_wav_to_mp4_conversion(self):
        """Test that the WAV to MP4 conversion works correctly."""
        success = convert_wav_to_mp4(str(self.wav_path), str(self.mp4_path))
        
        self.assertTrue(success, "Conversion function should return True")
        self.assertTrue(self.mp4_path.exists(), "MP4 file should be created")
        
        # Verify that the created file is a valid MP4 file
        # We can do this by checking the file size and by running ffprobe
        self.assertGreater(self.mp4_path.stat().st_size, 0, "MP4 file should not be empty")
        
        # A more robust check would be to use ffprobe, but for now, we'll just check existence and size

if __name__ == '__main__':
    unittest.main()
