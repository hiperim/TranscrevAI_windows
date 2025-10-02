# Gemini Suggestions: `tests/test_unit.py` Cleanup Plan

This document outlines the plan to remove unused and deprecated code from `tests/test_unit.py` based on the findings from the Vulture static analysis tool and manual review.

## Vulture Findings

```
tests\test_unit.py:13: unused import 'Optional' (90% confidence)
tests\test_unit.py:13: unused import 'Tuple' (90% confidence)
tests\test_unit.py:25: unused import 'patch' (90% confidence)
tests\test_unit.py:54: unused import 'SessionInfo' (90% confidence)
tests\test_unit.py:61: unused variable 'WebSocketMemoryManager' (60% confidence)
tests\test_unit.py:79: unused import 'align_transcription_with_diarization' (90% confidence)
tests\test_unit.py:79: unused import 'enhanced_diarization' (90% confidence)
tests\test_unit.py:478: unused variable 'attempt' (60% confidence)
tests\test_unit.py:1345: unused method 'extract_speakers_from_text' (60% confidence)
tests\test_unit.py:1647: unused variable 'performance_ok' (60% confidence)
tests\test_unit.py:1805: unused import 'align_transcription_with_diarization' (90% confidence)
tests\test_unit.py:1805: unused import 'enhanced_diarization' (90% confidence)
tests\test_unit.py:1806: unused import 'generate_srt' (90% confidence)
tests\test_unit.py:1808: unused import 'SessionConfig' (90% confidence)
tests\test_unit.py:1810: unused import 'OptimizedAudioProcessor' (90% confidence)
tests\test_unit.py:1810: unused import 'RobustAudioLoader' (90% confidence)
tests\test_unit.py:1811: unused import 'TranscriptionService' (90% confidence)
tests\test_unit.py:1969: unused import 'enhanced_diarization' (90% confidence)
tests\test_unit.py:2119: unused attribute 'test_timeout' (60% confidence)
```

## Analysis and Cleanup Plan

1.  **Remove Unused Imports:** The following imports will be removed:
    *   `Optional` and `Tuple` from `typing`
    *   `patch` from `unittest.mock`
    *   `SessionInfo` from `src.performance_optimizer`
    *   `align_transcription_with_diarization` and `enhanced_diarization` from `src.diarization` (multiple occurrences)
    *   `generate_srt` from `src.subtitle_generator`
    *   `SessionConfig` from `src.performance_optimizer`
    *   `OptimizedAudioProcessor` and `RobustAudioLoader` from `src.audio_processing`
    *   `TranscriptionService` from `src.transcription`

2.  **Remove Unused Variables:** The following unused variables will be removed:
    *   `WebSocketMemoryManager` (compatibility alias)
    *   `performance_ok` in `TestRealUserCompliance.test_rule_1_performance_standards`

3.  **Remove Unused Methods:** The following unused method will be removed:
    *   `extract_speakers_from_text` in `BenchmarkTextProcessor`

4.  **Ignored Vulture Findings:**
    *   The unused loop variable `attempt` will be ignored as it is a common and acceptable pattern.
    *   The unused attribute `test_timeout` will be ignored as it appears to be a Vulture error (the attribute is not defined).
