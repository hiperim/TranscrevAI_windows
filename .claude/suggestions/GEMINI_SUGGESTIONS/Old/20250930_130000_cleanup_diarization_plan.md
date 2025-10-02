# Gemini Suggestions: `diarization.py` Cleanup Plan

This document outlines the plan to remove unused code from `src/diarization.py` based on the findings from the Vulture static analysis tool.

## Vulture Findings

```
src\diarization.py:21: unused import 'mp' (90% confidence)
src\diarization.py:28: unused import 'json' (90% confidence)
src\diarization.py:29: unused import 'Path' (90% confidence)
src\diarization.py:89: unused attribute 'min_speakers' (60% confidence)
src\diarization.py:91: unused attribute 'confidence_threshold' (60% confidence)
src\diarization.py:92: unused attribute 'analysis_thresholds' (60% confidence)
src\diarization.py:95: unused attribute 'min_speakers' (60% confidence)
src\diarization.py:97: unused attribute 'confidence_threshold' (60% confidence)
src\diarization.py:98: unused attribute 'analysis_thresholds' (60% confidence)
src\diarization.py:106: unused attribute 'available_methods' (60% confidence)
src\diarization.py:110: unused attribute 'embedding_cache' (60% confidence)
src\diarization.py:333: unused variable 'audio_quality' (60% confidence)
src\diarization.py:672: unused method '_meets_performance_targets' (60% confidence)
src\diarization.py:711: unused class 'DiarizationProcess' (60% confidence)
src\diarization.py:1216: unused function 'align_transcription_with_diarization' (60% confidence)
src\diarization.py:1219: unused variable 'language' (100% confidence)
src\diarization.py:1270: unused function 'diarization_worker' (60% confidence)
src\diarization.py:1285: unused variable 'enhanced_diarization' (60% confidence)
src\diarization.py:1288: unused variable 'OptimizedSpeakerDiarization' (60% confidence)
```

## Cleanup Plan

1.  **Remove Unused Imports:** The following imports will be removed after verifying they are not used:
    *   `multiprocessing as mp`
    *   `json`
    *   `pathlib.Path`

2.  **Investigate and Remove Unused Attributes:** The following attributes of the `CPUSpeakerDiarization` class will be investigated. If they are confirmed to be unused, they will be removed:
    *   `min_speakers`
    *   `confidence_threshold`
    *   `analysis_thresholds`
    *   `available_methods`
    *   `embedding_cache`

3.  **Investigate and Remove Unused Variables:** The following variables will be investigated and removed if they are confirmed to be unused:
    *   `audio_quality` in `_select_optimal_method`
    *   `language` in `align_transcription_with_diarization`

4.  **Investigate and Remove Unused Methods and Functions:** The following methods and functions will be investigated and removed if they are confirmed to be unused:
    *   `_meets_performance_targets`
    *   `align_transcription_with_diarization`
    *   `diarization_worker`

5.  **Investigate and Remove Unused Classes:** The following class will be investigated and removed if it is confirmed to be unused:
    *   `DiarizationProcess`

6.  **Investigate and Remove Unused Global Variables:** The following global variables will be investigated and removed if they are confirmed to be unused:
    *   `enhanced_diarization`
    *   `OptimizedSpeakerDiarization`
