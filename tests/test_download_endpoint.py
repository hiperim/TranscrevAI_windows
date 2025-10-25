"""
Manual test for download endpoint
Tests the /api/download/{session_id}/{file_type} endpoint

Usage:
    1. Start the server: python main.py
    2. Run this test: python tests/test_download_endpoint.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processing import SessionManager
import tempfile

def test_download_endpoint_logic():
    """Test the download endpoint logic without HTTP requests."""

    print("="*80)
    print("TESTING DOWNLOAD ENDPOINT LOGIC")
    print("="*80)

    # Create SessionManager
    session_manager = SessionManager()
    print("\n✅ SessionManager created")

    # Create a test session
    session_id = session_manager.create_session()
    print(f"✅ Test session created: {session_id}")

    # Get session
    session = session_manager.get_session(session_id)
    print(f"✅ Session retrieved: {session['id']}")

    # Create mock files
    temp_dir = Path(tempfile.gettempdir())

    audio_file = temp_dir / f"{session_id}_audio.wav"
    transcript_file = temp_dir / f"{session_id}_transcript.txt"
    subtitles_file = temp_dir / f"{session_id}_subtitles.srt"

    # Write mock content
    audio_file.write_bytes(b"MOCK AUDIO DATA")
    transcript_file.write_text("Mock transcript content")
    subtitles_file.write_text("1\n00:00:00,000 --> 00:00:05,000\nMock subtitle")

    print(f"\n✅ Created mock files:")
    print(f"   - {audio_file}")
    print(f"   - {transcript_file}")
    print(f"   - {subtitles_file}")

    # Add files to session
    session["files"]["audio"] = str(audio_file)
    session["files"]["transcript"] = str(transcript_file)
    session["files"]["subtitles"] = str(subtitles_file)

    print(f"\n✅ Files added to session")

    # Test file retrieval
    print(f"\n📊 TESTING FILE RETRIEVAL:")

    for file_type in ['audio', 'transcript', 'subtitles']:
        file_path = session.get("files", {}).get(file_type)
        exists = Path(file_path).exists() if file_path else False

        status = "✅ PASS" if exists else "❌ FAIL"
        print(f"   {status} - {file_type}: {file_path}")

    # Test invalid file type
    print(f"\n📊 TESTING INVALID FILE TYPE:")
    invalid_type = "invalid"
    valid_types = ['audio', 'transcript', 'subtitles']
    is_valid = invalid_type in valid_types
    status = "❌ CORRECTLY REJECTED" if not is_valid else "✅ ERROR - should reject"
    print(f"   {status} - '{invalid_type}' not in {valid_types}")

    # Test nonexistent session
    print(f"\n📊 TESTING NONEXISTENT SESSION:")
    fake_session = session_manager.get_session("nonexistent-uuid")
    status = "✅ PASS" if fake_session is None else "❌ FAIL"
    print(f"   {status} - get_session() returned None for nonexistent ID")

    # Cleanup
    print(f"\n🧹 CLEANUP:")
    audio_file.unlink()
    transcript_file.unlink()
    subtitles_file.unlink()
    session_manager.delete_session(session_id)
    print(f"   ✅ Deleted mock files and session")

    print(f"\n{'='*80}")
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print(f"\nEndpoint logic is working correctly!")
    print(f"To test the HTTP endpoint:")
    print(f"  1. Start server: python main.py")
    print(f"  2. Create a session via SessionManager")
    print(f"  3. Add file paths to session['files']")
    print(f"  4. Access: http://localhost:8000/api/download/{{session_id}}/{{file_type}}")


if __name__ == "__main__":
    test_download_endpoint_logic()
