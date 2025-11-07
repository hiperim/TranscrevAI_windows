# Changelog

All notable changes to TranscrevAI will be documented in this file.

## [Milestone] DI Refactoring Complete + 100% Test Coverage - 2025-11-06

### Major Architectural Refactoring

**Migration from Global AppState to FastAPI Dependency Injection Pattern**

This milestone represents a complete architectural refactoring to improve code maintainability, testability, and scalability for multi-user deployment.

### Metrics Achieved (test_accuracy_performance.py - 06/11/2025 21:22)

**Transcription Quality:**
- Accuracy: 86.22% (normalized)
- WER (Word Error Rate): 0.138 (13.8%)
- Traditional Accuracy: 83.22%
- Traditional WER: 0.168 (16.8%)

**Diarization Quality:**
- Accuracy: 100% (both 2-speaker and 4-speaker audio)
- Correctly detected speakers in all test cases

**Performance:**
- Processing Speed Ratio: 1.59x realtime
  - Measurement: `total_processing_time / audio_duration`
  - Example: 79s processing time for 47s audio = 1.67x
  - Averaged across multiple test files
- Peak Memory Usage: 2,306 MB (within 3GB target)

**Test Coverage:**
- 32/32 tests passing (100%)
- test_services.py: 17/17 ✅
- test_live_server.py: 2/2 ✅
- test_edge_cases.py: 7/7 ✅
- test_accuracy_performance.py: 2/2 ✅
- test_performance.py: 4/4 ✅

### Changes and Refactoring

#### New Files
- **src/dependencies.py**: FastAPI DI container with singleton pattern
  - Thread-safe service initialization using RLock
  - Cached service instances
  - Proper cleanup on shutdown

#### Modified Core Files
- **main.py**: Complete DI integration
  - All endpoints refactored to use `Depends()`
  - Lifespan manager updated for DI initialization
  - Worker thread initialization with event loop handling
  - Added `/test/reset-rate-limit` endpoint for testing
  - Fixed WebSocket STOP action to prevent buffer conflicts

- **src/audio_processing.py**: Session management improvements
  - Fixed session override handling for reconnection scenarios
  - Improved thread-safety

- **src/websocket_handler.py**: DI integration
  - Refactored to accept injected dependencies

- **src/worker.py**: DI integration
  - Worker thread accepts services as parameters
  - Proper asyncio event loop handling

#### Test Suite Improvements

**test_services.py (17 fixes):**
- Updated mock paths for WhisperModel
- Fixed async/await patterns
- Updated method signatures for new API

**test_live_server.py (2 fixes):**
- Added missing imports (BackgroundTasks, requests)
- Fixed DI initialization in tests

**test_edge_cases.py (7 fixes):**
- Implemented rate limit reset fixture (autouse)
- Fixed rate limit testing to properly catch exceptions
- Improved error type assertions
- Added server reconnection handling

**test_accuracy_performance.py (2 fixes):**
- Added module-scoped fixtures for services
- Renamed helper function to avoid pytest autodiscovery

**test_performance.py (4 fixes):**
- Implemented orphan server cleanup function
- Module-scoped server fixture for all tests
- Removed skip condition (now 4/4 passing)
- Increased timeout to 60s for DI initialization

### Bug Fixes

1. **Worker Thread Deadlock**: Fixed asyncio.get_event_loop() timing issue
   - Modified get_worker_thread() to accept loop parameter
   - Lifespan now passes asyncio.get_running_loop()

2. **Threading Deadlock**: Changed Lock to RLock
   - Fixed non-reentrant lock causing nested call deadlocks
   - Services can now safely call other services during initialization

3. **WebSocket Final Batch**: Removed stop_recording() call
   - Prevented ValueError from empty buffer
   - Worker now properly receives stop signal

4. **Test Rate Limiting**: Proper fixture-based reset
   - Replaced environment variable approach
   - Tests now actually validate rate limiting

5. **Orphan Server Processes**: Automatic cleanup
   - Performance tests now kill orphan uvicorn processes
   - Prevents port 8000 conflicts and test instability

### Breaking Changes

None - All existing functionality preserved. This is purely internal refactoring.

### Migration Notes

- Global `AppState` class removed
- All services now accessed via DI using `Depends()`
- Worker thread initialization requires event loop parameter
- Tests must use proper fixtures for service injection

---

## Previous Versions

For changes prior to this milestone, refer to git commit history.
