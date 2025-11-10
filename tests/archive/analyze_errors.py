# tests/analyze_errors.py
"""
Error Analysis Tool - Training Set Only

Analyzes transcription errors from TRAINING SET (d.speakers.wav, q.speakers.wav)
to identify patterns and suggest PT-BR correction rules.

IMPORTANT: This tool ONLY analyzes training set to prevent overfitting.
"""

import asyncio
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import difflib

# Paths
AUDIO_DIR = Path(__file__).parent.parent / "data" / "recordings"
TRUTH_DIR = Path(__file__).parent / "ground_truth"
REPORTS_DIR = Path(__file__).parent.parent / ".claude" / "test_reports"

# TRAINING SET ONLY
TRAIN_FILES = {
    "d.speakers.wav": {"text_file": "d_speakers.txt"},
    "q.speakers.wav": {"text_file": "q_speakers.txt"}
}

def normalize_for_comparison(text: str) -> str:
    """Normalize text for word-level comparison."""
    text = text.lower()
    # Keep words and PT-BR chars, but remove punctuation
    text = re.sub(r'[^\w\sàáâãèéêìíîòóôõùúûç]', '', text)
    return ' '.join(text.split())

def get_word_diff(expected: str, actual: str) -> List[Tuple[str, str, str]]:
    """
    Compare expected vs actual text word-by-word.

    Returns list of (operation, expected_word, actual_word):
    - ('match', word, word): Words match
    - ('replace', expected, actual): Word was replaced
    - ('delete', expected, ''): Word was deleted (missing in transcription)
    - ('insert', '', actual): Word was inserted (extra in transcription)
    """
    expected_words = expected.split()
    actual_words = actual.split()

    matcher = difflib.SequenceMatcher(None, expected_words, actual_words)
    differences = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                differences.append(('match', expected_words[i], actual_words[j]))
        elif tag == 'replace':
            # Align replacements
            for i, j in zip(range(i1, i2), range(j1, j2)):
                differences.append(('replace', expected_words[i], actual_words[j]))
            # Handle mismatched lengths
            if (i2 - i1) > (j2 - j1):
                for i in range(j2 - j1, i2 - i1):
                    differences.append(('delete', expected_words[i1 + i], ''))
            elif (j2 - j1) > (i2 - i1):
                for j in range(i2 - i1, j2 - j1):
                    differences.append(('insert', '', actual_words[j1 + j]))
        elif tag == 'delete':
            for i in range(i1, i2):
                differences.append(('delete', expected_words[i], ''))
        elif tag == 'insert':
            for j in range(j1, j2):
                differences.append(('insert', '', actual_words[j]))

    return differences

def analyze_errors():
    """Analyze errors from training set transcriptions."""

    print("\n" + "="*70)
    print("ERROR ANALYSIS - TRAINING SET ONLY")
    print("Files: d.speakers.wav, q.speakers.wav")
    print("="*70)

    # Collect errors
    all_errors = []  # List of (expected, actual) tuples
    replacements = defaultdict(Counter)  # replacements[actual][expected] = count
    deletions = Counter()  # deleted words
    insertions = Counter()  # inserted words

    total_words = 0
    total_errors = 0

    for audio_file, truth_data in TRAIN_FILES.items():
        print(f"\n📄 Analyzing {audio_file}...")

        # Get ground truth
        truth_path = TRUTH_DIR / truth_data["text_file"]
        expected_raw = truth_path.read_text(encoding="utf-8").strip()

        # Get transcription from latest baseline test report
        # Find most recent baseline_train_set report
        reports = list(REPORTS_DIR.glob("baseline_train_set_*.json"))
        if not reports:
            print("  ⚠️  No baseline_train_set report found. Run test_baseline_train_set.py first.")
            continue

        latest_report = max(reports, key=lambda p: p.stat().st_mtime)
        report_data = json.loads(latest_report.read_text(encoding='utf-8'))

        # Find this file's result
        file_result = None
        for result in report_data['results']:
            if result['file'] == audio_file:
                file_result = result
                break

        if not file_result:
            print(f"  ⚠️  No result found for {audio_file}")
            continue

        actual_raw = file_result['actual_raw']

        # Normalize
        expected_norm = normalize_for_comparison(expected_raw)
        actual_norm = normalize_for_comparison(actual_raw)

        # Get word-level diff
        diffs = get_word_diff(expected_norm, actual_norm)

        # Count errors
        file_words = 0
        file_errors = 0

        for op, exp_word, act_word in diffs:
            if op == 'match':
                file_words += 1
            elif op == 'replace':
                file_words += 1
                file_errors += 1
                all_errors.append((exp_word, act_word))
                replacements[act_word][exp_word] += 1
            elif op == 'delete':
                file_words += 1
                file_errors += 1
                deletions[exp_word] += 1
            elif op == 'insert':
                file_errors += 1
                insertions[act_word] += 1

        total_words += file_words
        total_errors += file_errors

        error_rate = (file_errors / file_words * 100) if file_words > 0 else 0
        print(f"  Words: {file_words}, Errors: {file_errors} ({error_rate:.1f}%)")

    if total_words == 0:
        print("\n❌ No data to analyze. Run test_baseline_train_set.py first.")
        return

    # Summary
    print("\n" + "="*70)
    print("ERROR SUMMARY")
    print("="*70)
    print(f"Total words:  {total_words}")
    print(f"Total errors: {total_errors}")
    print(f"Error rate:   {total_errors/total_words*100:.2f}%")

    # TOP REPLACEMENT ERRORS
    print("\n" + "="*70)
    print("TOP 20 REPLACEMENT ERRORS (actual → expected)")
    print("="*70)

    # Flatten replacements for sorting
    replacement_pairs = []
    for actual, expected_counts in replacements.items():
        for expected, count in expected_counts.items():
            replacement_pairs.append((count, actual, expected))

    replacement_pairs.sort(reverse=True)

    print(f"{'Count':<8} {'Transcribed':<20} {'Expected':<20} {'Type'}")
    print("-" * 70)

    for count, actual, expected in replacement_pairs[:20]:
        # Classify error type
        error_type = classify_error(actual, expected)
        print(f"{count:<8} {actual:<20} {expected:<20} {error_type}")

    # TOP DELETIONS
    print("\n" + "="*70)
    print("TOP 10 DELETIONS (missing words)")
    print("="*70)
    print(f"{'Count':<8} {'Missing Word':<30}")
    print("-" * 70)
    for word, count in deletions.most_common(10):
        print(f"{count:<8} {word:<30}")

    # TOP INSERTIONS
    print("\n" + "="*70)
    print("TOP 10 INSERTIONS (extra words)")
    print("="*70)
    print(f"{'Count':<8} {'Extra Word':<30}")
    print("-" * 70)
    for word, count in insertions.most_common(10):
        print(f"{count:<8} {word:<30}")

    # LINGUISTIC ANALYSIS
    print("\n" + "="*70)
    print("LINGUISTIC ANALYSIS - SAFE CORRECTION RULES")
    print("="*70)

    safe_rules = analyze_safe_rules(replacement_pairs)

    if safe_rules:
        print("\n✅ SAFE RULES (Level 1 - Zero Ambiguity):")
        for actual, expected, count, reason in safe_rules['level1']:
            print(f"  '{actual}' → '{expected}' (count: {count}) - {reason}")

        print("\n⚠️  POTENTIALLY SAFE RULES (Level 2 - Low Ambiguity):")
        for actual, expected, count, reason in safe_rules['level2']:
            print(f"  '{actual}' → '{expected}' (count: {count}) - {reason}")

        print("\n❌ UNSAFE RULES (Level 3 - High Ambiguity - DO NOT ADD):")
        for actual, expected, count, reason in safe_rules['level3']:
            print(f"  '{actual}' → '{expected}' (count: {count}) - {reason}")

    print("\n" + "="*70)
    print("NOTE: Only Level 1 rules should be added automatically.")
    print("Level 2 requires manual review. Level 3 should NEVER be added.")
    print("="*70)

    return {
        'replacement_pairs': replacement_pairs,
        'safe_rules': safe_rules,
        'total_words': total_words,
        'total_errors': total_errors
    }

def classify_error(actual: str, expected: str) -> str:
    """Classify the type of error."""
    # Missing accent
    if remove_accents(actual) == remove_accents(expected):
        return "Missing accent"

    # Phonetic similarity (simplified)
    if actual in ['pra', 'pro', 'ta', 'ce'] or expected in ['para', 'para o', 'esta', 'voce']:
        return "Phonetic/Elisão"

    # Similar words
    if difflib.SequenceMatcher(None, actual, expected).ratio() > 0.7:
        return "Similar word"

    return "Substitution"

def remove_accents(text: str) -> str:
    """Remove accents from text."""
    accents = {
        'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i',
        'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u',
        'ç': 'c'
    }
    result = text
    for accented, plain in accents.items():
        result = result.replace(accented, plain)
    return result

def analyze_safe_rules(replacement_pairs: List[Tuple[int, str, str]]) -> Dict:
    """
    Analyze replacement pairs and classify into safety levels.

    Level 1 (Safe): Zero ambiguity - always correct
    Level 2 (Caution): Low ambiguity - review required
    Level 3 (Unsafe): High ambiguity - never add
    """

    # Known safe patterns (Level 1)
    SAFE_ELISOES = {
        'pra': 'para',
        'pro': 'para o',
        'pruma': 'para uma',
        'ta': 'está',
        'tava': 'estava',
        'tao': 'tão',
        'ce': 'você',
        'ceis': 'vocês',
        'ne': 'né',  # Já é a forma coloquial correta
    }

    # Known unsafe patterns (Level 3)
    UNSAFE_WORDS = {
        'e',   # Conjunção vs verbo "é"
        'da',  # Preposição vs verbo "dá"
        'de',  # Preposição vs verbo "dê"
        'a',   # Artigo vs verbo "há"
        'o',   # Artigo (muito comum)
        'as',  # Artigo (muito comum)
        'os',  # Artigo (muito comum)
    }

    level1 = []  # Safe
    level2 = []  # Caution
    level3 = []  # Unsafe

    for count, actual, expected in replacement_pairs[:30]:  # Top 30 errors

        # Skip if same word (shouldn't happen but just in case)
        if actual == expected:
            continue

        # Level 1: Known safe elisões
        if actual in SAFE_ELISOES and SAFE_ELISOES[actual] == expected:
            level1.append((actual, expected, count, "Known safe elisão"))
            continue

        # Level 3: Known unsafe (high ambiguity)
        if actual in UNSAFE_WORDS or expected in UNSAFE_WORDS:
            level3.append((actual, expected, count, "High ambiguity word"))
            continue

        # Level 1: Missing accent only (safe if word is uncommon as unaccented)
        if remove_accents(actual) == remove_accents(expected) and len(actual) > 2:
            # Check if it's a common word that needs context
            if actual not in ['esta', 'esse', 'essa']:  # These need context
                level1.append((actual, expected, count, "Missing accent (safe)"))
            else:
                level2.append((actual, expected, count, "Missing accent (context-dependent)"))
            continue

        # Level 2: Phonetically similar (needs review)
        similarity = difflib.SequenceMatcher(None, actual, expected).ratio()
        if similarity > 0.7:
            level2.append((actual, expected, count, f"Similar ({similarity:.0%}) - review needed"))
        else:
            level3.append((actual, expected, count, "Different words - likely context-dependent"))

    return {
        'level1': level1,
        'level2': level2,
        'level3': level3
    }

if __name__ == "__main__":
    analyze_errors()
