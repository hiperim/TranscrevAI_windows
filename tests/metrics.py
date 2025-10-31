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


def normalize_text(text: str) -> str:
    """
    Normalize text for content-focused comparison by removing punctuation and formatting.

    This normalization removes elements that are considered features (punctuation, capitalization)
    rather than transcription errors, allowing WER to focus on actual word content.

    Transformations applied:
    - Convert to lowercase
    - Remove common punctuation marks (. , ! ? : ; " ' -)
    - Normalize whitespace (multiple spaces → single space, trim)

    Args:
        text: Text to normalize

    Returns:
        Normalized text with only lowercase words and single spaces

    Examples:
        >>> normalize_text("Eu acho que sim.")
        "eu acho que sim"
        >>> normalize_text("Olá,  mundo!")
        "olá mundo"
    """
    import re

    # Convert to lowercase
    text = text.lower()

    # Remove common punctuation
    text = re.sub(r'[.,!?:;"\'\\-]', '', text)

    # Normalize whitespace: replace multiple spaces with single space and trim
    text = ' '.join(text.split())

    return text


def calculate_wer_normalized(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) after normalizing both texts.

    This metric focuses on word content accuracy without penalizing punctuation or capitalization,
    which are considered enhancement features rather than transcription errors.

    The normalization process:
    1. Converts to lowercase
    2. Removes punctuation
    3. Normalizes whitespace

    Then calculates standard WER on the normalized texts.

    Args:
        reference: Expected/correct text
        hypothesis: Predicted/transcribed text

    Returns:
        WER as float (0.0 to 1.0+), where lower is better

    Examples:
        >>> calculate_wer_normalized("eu acho que sim", "Eu acho que sim.")
        0.0  # Perfect match when normalized
        >>> calculate_wer_normalized("olá mundo", "Olá, mundo!")
        0.0  # Punctuation doesn't count as error
    """
    normalized_ref = normalize_text(reference)
    normalized_hyp = normalize_text(hypothesis)

    return calculate_wer(normalized_ref, normalized_hyp)


def calculate_dual_wer(reference: str, hypothesis: str) -> dict:
    """
    Calculate both traditional and normalized WER simultaneously.

    Returns both metrics in a single call for convenience and consistency.
    This allows comparing traditional WER (penalizes punctuation) vs normalized WER
    (focuses on word content).

    Args:
        reference: Expected/correct text
        hypothesis: Predicted/transcribed text

    Returns:
        Dictionary with both WER metrics:
        {
            'wer_traditional': float,          # Traditional WER (with punctuation penalty)
            'wer_normalized': float,           # Normalized WER (content-focused)
            'accuracy_traditional': float,     # 1 - wer_traditional, as percentage
            'accuracy_normalized': float       # 1 - wer_normalized, as percentage
        }

    Example:
        >>> result = calculate_dual_wer("eu acho que sim", "Eu acho que sim.")
        >>> result
        {
            'wer_traditional': 0.5,           # Penalized for capitalization and punctuation
            'wer_normalized': 0.0,            # Perfect match on content
            'accuracy_traditional': 50.0,
            'accuracy_normalized': 100.0
        }
    """
    wer_trad = calculate_wer(reference, hypothesis)
    wer_norm = calculate_wer_normalized(reference, hypothesis)

    return {
        'wer_traditional': wer_trad,
        'wer_normalized': wer_norm,
        'accuracy_traditional_percent': (1 - wer_trad) * 100,
        'accuracy_normalized_percent': (1 - wer_norm) * 100
    }
