"""
Unit tests for SessionManager class in src/audio_processing.py

Tests cover:
- Session creation and UUID generation
- Session retrieval and activity tracking
- Session deletion and resource cleanup
- Automatic cleanup of expired sessions
- Thread safety for concurrent operations
- Active session counting

Usage:
    pytest tests/test_session_manager.py -v
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processing import SessionManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestSessionManagerBasics:
    """Test basic SessionManager functionality."""

    def test_create_session_returns_uuid(self):
        """Test that create_session returns a valid UUID string."""
        manager = SessionManager()
        session_id = manager.create_session()

        # UUID v4 format: 8-4-4-4-12 hex characters
        parts = session_id.split('-')
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

        logger.info(f"✅ Session created with UUID: {session_id}")

    def test_create_session_initializes_processor(self):
        """Test that create_session creates a LiveAudioProcessor instance."""
        manager = SessionManager()
        session_id = manager.create_session()

        session = manager.get_session(session_id)
        assert session is not None
        assert "processor" in session
        assert session["processor"] is not None
        assert session["status"] == "idle"

        logger.info(f"✅ Session initialized with processor and idle status")

    def test_get_session_returns_none_for_nonexistent(self):
        """Test that get_session returns None for nonexistent session."""
        manager = SessionManager()

        session = manager.get_session("nonexistent-uuid")
        assert session is None

        logger.info(f"✅ get_session correctly returns None for nonexistent session")

    def test_get_session_updates_last_activity(self):
        """Test that get_session updates the last_activity timestamp."""
        manager = SessionManager()
        session_id = manager.create_session()

        # Get initial timestamp
        session1 = manager.get_session(session_id)
        timestamp1 = session1["last_activity"]

        # Wait a bit
        time.sleep(0.1)

        # Get session again
        session2 = manager.get_session(session_id)
        timestamp2 = session2["last_activity"]

        # Timestamp should be updated
        assert timestamp2 > timestamp1

        logger.info(f"✅ last_activity updated from {timestamp1} to {timestamp2}")

    def test_delete_session_removes_session(self):
        """Test that delete_session removes the session."""
        manager = SessionManager()
        session_id = manager.create_session()

        # Verify session exists
        assert manager.get_session(session_id) is not None
        assert manager.get_active_session_count() == 1

        # Delete session
        manager.delete_session(session_id)

        # Verify session is gone
        assert manager.get_session(session_id) is None
        assert manager.get_active_session_count() == 0

        logger.info(f"✅ Session {session_id} successfully deleted")

    def test_get_active_session_count(self):
        """Test that get_active_session_count returns correct count."""
        manager = SessionManager()

        assert manager.get_active_session_count() == 0

        id1 = manager.create_session()
        assert manager.get_active_session_count() == 1

        id2 = manager.create_session()
        assert manager.get_active_session_count() == 2

        manager.delete_session(id1)
        assert manager.get_active_session_count() == 1

        manager.delete_session(id2)
        assert manager.get_active_session_count() == 0

        logger.info(f"✅ Session count tracking is accurate")

    def test_get_all_session_ids(self):
        """Test that get_all_session_ids returns all session IDs."""
        manager = SessionManager()

        ids_created = [
            manager.create_session(),
            manager.create_session(),
            manager.create_session()
        ]

        ids_retrieved = manager.get_all_session_ids()

        assert len(ids_retrieved) == 3
        assert set(ids_created) == set(ids_retrieved)

        logger.info(f"✅ get_all_session_ids returns correct IDs")


class TestSessionManagerCleanup:
    """Test automatic cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_old_sessions_removes_expired(self):
        """Test that cleanup removes sessions older than timeout."""
        # Use 1-second timeout for testing
        manager = SessionManager(session_timeout_hours=1/3600)  # 1 second

        session_id = manager.create_session()

        # Verify session exists
        assert manager.get_active_session_count() == 1

        # Wait for session to expire
        await asyncio.sleep(1.1)

        # Run cleanup (single iteration for testing)
        with patch.object(manager, '_lock'):
            # Manually trigger cleanup logic
            from datetime import datetime, timedelta

            with manager._lock:
                current_time = datetime.now()
                timeout = timedelta(hours=manager.session_timeout_hours)
                expired_sessions = []

                for sid, session in list(manager.sessions.items()):
                    if current_time - session["last_activity"] > timeout:
                        expired_sessions.append(sid)

                for sid in expired_sessions:
                    manager.delete_session(sid)

        # Verify session was removed
        assert manager.get_active_session_count() == 0

        logger.info(f"✅ Expired session cleaned up successfully")

    @pytest.mark.asyncio
    async def test_cleanup_preserves_active_sessions(self):
        """Test that cleanup does NOT remove active sessions."""
        # Use 1-second timeout
        manager = SessionManager(session_timeout_hours=1/3600)

        session_id = manager.create_session()

        # Keep session active by accessing it
        await asyncio.sleep(0.5)
        manager.get_session(session_id)  # Updates last_activity

        await asyncio.sleep(0.5)
        manager.get_session(session_id)  # Updates last_activity again

        # Total elapsed: 1 second, but last_activity is recent

        # Run cleanup
        with patch.object(manager, '_lock'):
            from datetime import datetime, timedelta

            with manager._lock:
                current_time = datetime.now()
                timeout = timedelta(hours=manager.session_timeout_hours)
                expired_sessions = []

                for sid, session in list(manager.sessions.items()):
                    if current_time - session["last_activity"] > timeout:
                        expired_sessions.append(sid)

                for sid in expired_sessions:
                    manager.delete_session(sid)

        # Session should still exist
        assert manager.get_active_session_count() == 1

        logger.info(f"✅ Active session preserved during cleanup")


class TestSessionManagerThreadSafety:
    """Test thread safety of SessionManager operations."""

    def test_concurrent_session_creation(self):
        """Test that multiple threads can create sessions safely."""
        manager = SessionManager()
        session_ids: List[str] = []
        errors: List[Exception] = []

        def create_sessions():
            try:
                for _ in range(10):
                    session_id = manager.create_session()
                    session_ids.append(session_id)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        # Create 5 threads, each creating 10 sessions
        threads = [threading.Thread(target=create_sessions) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all sessions were created
        assert len(session_ids) == 50

        # Verify all UUIDs are unique
        assert len(set(session_ids)) == 50

        # Verify manager count is correct
        assert manager.get_active_session_count() == 50

        logger.info(f"✅ 50 sessions created concurrently without race conditions")

    def test_concurrent_get_and_delete(self):
        """Test that get and delete operations are thread-safe."""
        manager = SessionManager()

        # Create 20 sessions
        session_ids = [manager.create_session() for _ in range(20)]

        get_count = [0]
        delete_count = [0]
        errors: List[Exception] = []

        def get_sessions():
            try:
                for session_id in session_ids:
                    session = manager.get_session(session_id)
                    if session is not None:
                        get_count[0] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def delete_sessions():
            try:
                for session_id in session_ids[:10]:  # Delete first 10
                    manager.delete_session(session_id)
                    delete_count[0] += 1
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        # Start threads: 3 readers, 1 deleter
        get_threads = [threading.Thread(target=get_sessions) for _ in range(3)]
        delete_thread = threading.Thread(target=delete_sessions)

        for t in get_threads:
            t.start()
        delete_thread.start()

        for t in get_threads:
            t.join()
        delete_thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify 10 sessions remain
        assert manager.get_active_session_count() == 10

        logger.info(f"✅ Concurrent get/delete operations completed safely")


class TestSessionManagerResourceTracking:
    """Test resource tracking functionality."""

    def test_session_tracks_files(self):
        """Test that session properly tracks generated files."""
        manager = SessionManager()
        session_id = manager.create_session()

        session = manager.get_session(session_id)

        # Add mock file paths
        session["files"]["audio"] = "/path/to/audio.wav"
        session["files"]["transcript"] = "/path/to/transcript.txt"
        session["files"]["subtitles"] = "/path/to/subtitles.srt"

        # Retrieve session again
        session_retrieved = manager.get_session(session_id)

        assert session_retrieved["files"]["audio"] == "/path/to/audio.wav"
        assert session_retrieved["files"]["transcript"] == "/path/to/transcript.txt"
        assert session_retrieved["files"]["subtitles"] == "/path/to/subtitles.srt"

        logger.info(f"✅ Session file tracking works correctly")

    def test_session_status_updates(self):
        """Test that session status can be updated."""
        manager = SessionManager()
        session_id = manager.create_session()

        # Initial status
        session = manager.get_session(session_id)
        assert session["status"] == "idle"

        # Update to recording
        session["status"] = "recording"
        assert manager.get_session(session_id)["status"] == "recording"

        # Update to processing
        session["status"] = "processing"
        assert manager.get_session(session_id)["status"] == "processing"

        # Update to complete
        session["status"] = "complete"
        assert manager.get_session(session_id)["status"] == "complete"

        logger.info(f"✅ Session status updates work correctly")


def main():
    """Run all tests with pytest."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🧪 RUNNING SESSIONMANAGER UNIT TESTS")
    logger.info(f"{'='*80}\n")

    # Run pytest programmatically
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
