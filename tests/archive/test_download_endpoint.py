import unittest
from fastapi.testclient import TestClient
from pathlib import Path
import os
import wave
import struct

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app, app_state

class TestDownloadEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.temp_dir = Path("tests") / "temp_test_files"
        self.temp_dir.mkdir(exist_ok=True)
        self.wav_path = self.temp_dir / "test_download.wav"

        # Create a dummy WAV file
        with wave.open(str(self.wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            for _ in range(16000): # 1 second of silence
                wf.writeframes(struct.pack('<h', 0))

    def tearDown(self):
        if self.wav_path.exists():
            self.wav_path.unlink()
        mp4_path = self.temp_dir / "test_download.mp4"
        if mp4_path.exists():
            mp4_path.unlink()
        if self.temp_dir.exists():
            self.temp_dir.rmdir()

    def test_download_mp4_conversion(self):
        """Test that the on-demand MP4 conversion in the download endpoint works."""
        with TestClient(app) as client:
            session_id = "test_mp4_download_session"
            
            # Manually create a session to simulate a completed live recording
            session_data = {
                "id": session_id,
                "audio_format": "mp4",
                "files": {
                    "audio": str(self.wav_path)
                }
            }
            app_state.session_manager.sessions[session_id] = session_data

            # Make a request to the download endpoint
            response = client.get(f"/api/download/{session_id}/audio")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['content-type'], 'video/mp4')
            
            # Check that the MP4 file was created
            mp4_path = self.temp_dir / "test_download.mp4"
            self.assertTrue(mp4_path.exists())
            self.assertGreater(mp4_path.stat().st_size, 0)

if __name__ == '__main__':
    unittest.main()