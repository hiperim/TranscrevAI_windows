# tests/metrics.py
"""
Metrics for measuring transcription quality improvement.
Implements WER (Word Error Rate) and CER (Character Error Rate).
"""

from difflib import SequenceMatcher
from typing import Tuple


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (Substitutions + Deletions + Insertions) / Total words in reference

    Lower is better (0.0 = perfect match, 1.0 = completely different)

    Args:
        reference: Expected/correct text
        hypothesis: Predicted/transcribed text

    Returns:
        WER as float (0.0 to 1.0+)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Use Levenshtein distance (dynamic programming)
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    # Initialize first column and row
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    # Calculate edit distance
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # WER = edit distance / reference length
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).

    Similar to WER but at character level.
    More sensitive to accentuation and punctuation errors.

    Args:
        reference: Expected/correct text
        hypothesis: Predicted/transcribed text

    Returns:
        CER as float (0.0 to 1.0+)
    """
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    # Use Levenshtein distance
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    cer = d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)
    return cer


def calculate_similarity(reference: str, hypothesis: str) -> float:
    """
    Calculate similarity ratio using SequenceMatcher.

    Returns value between 0.0 and 1.0 (higher is better).
    Useful as a complementary metric to WER/CER.

    Args:
        reference: Expected/correct text
        hypothesis: Predicted/transcribed text

    Returns:
        Similarity ratio (0.0 to 1.0)
    """
    return SequenceMatcher(None, reference.lower(), hypothesis.lower()).ratio()
