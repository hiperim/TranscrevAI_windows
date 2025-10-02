# Comprehensive Project Changes Log (2025-09-30) - Last 30-40 Minutes

This document details the modifications made to the TranscrevAI project in the last 30-40 minutes of the current session.

## 5. SPRINT 3 - Speaker Embedding Integration (Resemblyzer)

### 5.1 `requirements.txt`
- **Added:** `resemblyzer>=0.1.0,<1.0.0` as a new dependency.

### 5.2 `src/diarization.py`
- **New Class `LazyVoiceEncoder`:** Implemented for lazy loading of `resemblyzer.VoiceEncoder` using the `__getattr__` pattern.
- **`CPUSpeakerDiarization.__init__`:** Instantiated `self.speaker_encoder = LazyVoiceEncoder()`.
- **New Method `_extract_speaker_embedding`:** Extracts 256-dimensional speaker embeddings from audio data using `resemblyzer.preprocess_wav` and `self.speaker_encoder.embed_utterance`.
- **`_clustering_diarization` method:**
    - **Removed:** All MFCC extraction and scaling logic.
    - **Implemented:** Audio segmentation into fixed-size chunks (2s length, 1s overlap).
    - **Integrated:** Calls `self._extract_speaker_embedding` for each audio chunk to get embeddings.
    - **Modified:** Uses the extracted embeddings (`features_scaled = embeddings`) for clustering.
    - **Modified:** Changed `AgglomerativeClustering` metric to `cosine` and linkage to `average` (from `euclidean` and `ward`).
    - **Modified:** Adapted segment creation logic to map cluster labels back to `segments_for_embedding`.
    - **Cleaned up:** Removed unused variables (`features`, `mfccs`) from the cleanup section.
- **`_calculate_confidence_scores` method:**
    - **Modified:** Changed `euclidean_distances` to `cosine_distances` for distance calculation.

## Rationale for Chosen Modifications

The primary goal of these modifications is to significantly improve the accuracy of speaker diarization, which was a major bottleneck in the project. The previous MFCC-based approach, despite various optimizations, proved insufficient for robust speaker discrimination, especially with short audio segments.

1.  **Transition from MFCCs to Speaker Embeddings (Resemblyzer):**
    *   **Why:** MFCCs are primarily designed for *speech content* analysis, not *speaker identity*. Speaker embeddings, on the other hand, are specifically trained to capture unique voice characteristics, making them far more discriminative for identifying *who* spoke. This directly addresses the root cause of inaccurate diarization.
    *   **Choice of Resemblyzer:** Resemblyzer was selected because it is a lightweight, CPU-friendly, PyTorch-based solution that does *not* require a Hugging Face token, aligning with project constraints. Its `preprocess_wav` function simplifies audio preparation, and its `VoiceEncoder` provides a straightforward API for embedding extraction.
    *   **Benefits:** Expected to provide a substantial leap in speaker detection accuracy and robustness, particularly for distinguishing between similar-sounding speakers and in short audio clips.

2.  **Lazy Loading of `VoiceEncoder` (`LazyVoiceEncoder` class):**
    *   **Why:** Deep learning models, even lightweight ones, consume significant memory upon loading. Lazy loading ensures that the `VoiceEncoder` model is only loaded into memory when it's actually needed (i.e., the first time an embedding is requested).
    *   **Benefits:** Optimizes memory usage and reduces application startup time, especially in scenarios where diarization might not be required for every transcription.

3.  **Audio Segmentation for Embedding Extraction:**
    *   **Why:** Speaker embeddings are typically extracted from meaningful segments of speech rather than individual frames (like MFCCs). Segmenting the audio into fixed-size, overlapping chunks (e.g., 2 seconds with 1 second overlap) provides sufficient context for the embedding model to generate robust speaker representations.
    *   **Benefits:** Improves the quality of the extracted embeddings by providing adequate temporal context, leading to more accurate clustering.

4.  **Agglomerative Clustering with Cosine Metric and Average Linkage:**
    *   **Why:** With speaker embeddings, which are high-dimensional vectors representing speaker identity, cosine distance is a more appropriate metric for measuring similarity than Euclidean distance. Average linkage is robust for irregular clusters, which is common in speaker data.
    *   **Benefits:** Optimizes the clustering algorithm to work effectively with the new speaker embedding features, leading to better speaker separation.

5.  **Adapted Segment Creation Logic:**
    *   **Why:** The previous segment creation logic was tied to MFCC frame durations. With embeddings extracted from larger audio chunks, the segment creation needed to be adapted to map the cluster labels back to these embedding-derived segments.
    *   **Benefits:** Ensures that the diarization output accurately reflects the speaker labels for the identified audio segments.

6.  **`_calculate_confidence_scores` with Cosine Distances:**
    *   **Why:** The confidence scoring mechanism relies on distance to cluster centroids. Since we switched to cosine distance for clustering, the confidence calculation should also use cosine distances for consistency and accuracy.
    *   **Benefits:** Provides a more accurate and consistent confidence score for speaker assignments, which is crucial for the multi-stage adaptive clustering pipeline.

