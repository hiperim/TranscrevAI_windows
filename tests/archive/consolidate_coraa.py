"""
Consolidate CORAA analysis results from multiple runs.
Combines error patterns and generates comprehensive analysis.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List

# Load all three analysis files
files = [
    "tests/logs/coraa_analysis/coraa_analysis_20251024_023057.json",  # Files 1-200
    "tests/logs/coraa_analysis/coraa_analysis_20251024_085007.json",  # Files 201-600
    "tests/logs/coraa_analysis/coraa_analysis_20251024_113322.json",  # Files 601-800
]

# Consolidate data
total_files = 0
total_words = 0
total_errors = 0
error_patterns = Counter()

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        total_files += data['files_processed']
        total_words += data['total_words']
        total_errors += data['total_errors']

        # Consolidate error patterns
        for error in data['top_50_errors']:
            key = (error['ground_truth'], error['transcribed'])
            error_patterns[key] += error['count']

# Calculate consolidated statistics
average_wer = total_errors / total_words if total_words > 0 else 0

# Get top 100 errors (for comprehensive analysis)
top_100_errors = [
    {
        "ground_truth": gt,
        "transcribed": tr,
        "count": count
    }
    for (gt, tr), count in error_patterns.most_common(100)
]

# Create consolidated result
consolidated = {
    "analysis_date": "2025-10-24",
    "total_files_analyzed": total_files,
    "file_ranges": "1-800 (dev set)",
    "total_words": total_words,
    "total_errors": total_errors,
    "average_wer": round(average_wer, 4),
    "top_100_errors": top_100_errors
}

# Save consolidated results
output_path = "tests/logs/coraa_analysis/consolidated_analysis.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(consolidated, f, indent=2, ensure_ascii=False)

print(f"✅ Consolidated {total_files} files")
print(f"📊 Total words: {total_words}")
print(f"❌ Total errors: {total_errors}")
print(f"📈 Average WER: {average_wer:.2%}")
print(f"💾 Saved to: {output_path}")

# Identify patterns with count >= 5 (universal patterns)
universal_patterns = [e for e in top_100_errors if e['count'] >= 5]
print(f"\n🎯 Universal patterns (count >= 5): {len(universal_patterns)}")
for i, pattern in enumerate(universal_patterns[:20], 1):
    print(f"{i}. '{pattern['ground_truth']}' → '{pattern['transcribed']}' ({pattern['count']}x)")
