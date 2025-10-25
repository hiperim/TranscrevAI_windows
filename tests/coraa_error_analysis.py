"""
CORAA Error Analysis Script
Analyzes Whisper medium transcription errors on CORAA corpus.

Processes first 100 files from dev set to identify:
- Top 50 most common word-level errors
- Systematic patterns in PT-BR transcription
- Data-driven correction opportunities

Usage:
    python tests/coraa_error_analysis.py
"""

import asyncio
import csv
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from difflib import SequenceMatcher
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CORAAnalyzer:
    """Analyzes Whisper transcription errors on CORAA corpus."""

    def __init__(self, coraa_dir: Path):
        """
        Initialize CORAA analyzer.

        Args:
            coraa_dir: Path to CORAA corpus directory
        """
        self.coraa_dir = coraa_dir
        self.dev_dir = coraa_dir / "dev"
        self.metadata_path = self.dev_dir / "metadata_dev_final.csv"
        self.error_patterns = Counter()
        self.total_words = 0
        self.total_errors = 0
        self.transcription_service = None

    async def analyze_corpus(self, num_files: int = 100, offset: int = 0) -> Dict:
        """
        Analyze first N files from CORAA dev set.

        Args:
            num_files: Number of files to process (default: 100)
            offset: Starting position - skip first N files (default: 0)

        Returns:
            Dictionary with analysis results:
            {
                "files_processed": int,
                "total_words": int,
                "total_errors": int,
                "average_wer": float,
                "top_50_errors": [
                    {"ground_truth": str, "transcribed": str, "count": int},
                    ...
                ]
            }
        """
        logger.info("\n" + "="*80)
        logger.info(f"CORAA ERROR ANALYSIS - Processing {num_files} files")
        logger.info("="*80 + "\n")

        # Initialize transcription service
        logger.info("Initializing Whisper transcription service...")
        self.transcription_service = TranscriptionService(
            model_name="medium",
            device="cpu"
        )
        await self.transcription_service.initialize(compute_type="int8")
        logger.info("Transcription service ready!\n")

        # Load metadata with offset
        logger.info(f"Loading metadata from {self.metadata_path}...")
        logger.info(f"Skipping first {offset} files (already analyzed)")
        audio_files = self._load_metadata(num_files, offset=offset)
        logger.info(f"Loaded {len(audio_files)} file entries (files {offset}-{offset+num_files-1})\n")

        # Process each file
        files_processed = 0
        files_skipped = 0

        for i, file_entry in enumerate(audio_files, 1):
            audio_path = self.dev_dir / file_entry["file_path"]
            ground_truth = file_entry["text"]

            # Skip if audio file doesn't exist
            if not audio_path.exists():
                logger.warning(f"[{i}/{num_files}] Audio not found: {audio_path.name}")
                files_skipped += 1
                continue

            logger.info(f"\n[{i}/{num_files}] Processing: {audio_path.name}")
            logger.info(f"Ground truth: {ground_truth[:100]}...")

            try:
                # Transcribe with our pipeline
                transcription = await self._transcribe_audio(audio_path)
                logger.info(f"Transcribed:  {transcription[:100]}...")

                # Find errors
                errors = self._find_word_errors(ground_truth, transcription)

                # Update counters
                gt_words = len(ground_truth.split())
                self.total_words += gt_words
                self.total_errors += len(errors)

                for gt_word, tr_word in errors:
                    self.error_patterns[(gt_word, tr_word)] += 1

                logger.info(f"Words: {gt_words}, Errors: {len(errors)}")
                files_processed += 1

            except Exception as e:
                logger.error(f"Error processing {audio_path.name}: {e}")
                files_skipped += 1
                continue

        # Calculate statistics
        average_wer = (self.total_errors / self.total_words) if self.total_words > 0 else 0

        # Get top 50 errors
        top_50_errors = [
            {
                "ground_truth": gt,
                "transcribed": tr,
                "count": count
            }
            for (gt, tr), count in self.error_patterns.most_common(50)
        ]

        results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "total_words": self.total_words,
            "total_errors": self.total_errors,
            "average_wer": average_wer,
            "top_50_errors": top_50_errors
        }

        # Save results
        self._save_results(results)

        return results

    def _load_metadata(self, num_files: int, offset: int = 0) -> List[Dict]:
        """
        Load metadata from CSV file.

        Args:
            num_files: Number of files to load
            offset: Starting position (skip first N files)

        Returns:
            List of dictionaries with file_path and text
        """
        audio_files = []

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Skip offset files
                if i < offset:
                    continue
                # Stop after num_files
                if i >= offset + num_files:
                    break
                audio_files.append({
                    "file_path": row["file_path"],
                    "text": row["text"]
                })

        return audio_files

    async def _transcribe_audio(self, audio_path: Path) -> str:
        """
        Transcribe audio file using current pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        result = await self.transcription_service.transcribe_with_enhancements(
            str(audio_path),
            word_timestamps=False
        )
        return result.text

    def _find_word_errors(
        self,
        ground_truth: str,
        transcription: str
    ) -> List[Tuple[str, str]]:
        """
        Find word-level substitution errors.

        Args:
            ground_truth: Ground truth text
            transcription: Transcribed text

        Returns:
            List of (ground_truth_word, transcribed_word) tuples
        """
        # Normalize and split
        gt_words = ground_truth.lower().split()
        tr_words = transcription.lower().split()

        errors = []
        matcher = SequenceMatcher(None, gt_words, tr_words)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Word substitution - pair up words
                for gt_w, tr_w in zip(gt_words[i1:i2], tr_words[j1:j2]):
                    errors.append((gt_w, tr_w))

        return errors

    def _save_results(self, results: Dict) -> None:
        """
        Save analysis results to JSON file.

        Args:
            results: Analysis results dictionary
        """
        # Create logs directory if needed
        logs_dir = Path(__file__).parent / "logs" / "coraa_analysis"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Save to timestamped file
        output_path = logs_dir / f"coraa_analysis_{results['timestamp']}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n\nResults saved to: {output_path}")


async def main():
    """Run CORAA error analysis."""

    coraa_dir = Path(r"C:\TranscrevAI_windows\coraa")

    if not coraa_dir.exists():
        logger.error(f"CORAA directory not found: {coraa_dir}")
        return

    analyzer = CORAAnalyzer(coraa_dir)
    # Process files 601-800 (200 NEW files, skipping first 600 already analyzed)
    results = await analyzer.analyze_corpus(num_files=200, offset=600)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80 + "\n")

    logger.info(f"Files processed: {results['files_processed']}")
    logger.info(f"Files skipped:   {results['files_skipped']}")
    logger.info(f"Total words:     {results['total_words']}")
    logger.info(f"Total errors:    {results['total_errors']}")
    logger.info(f"Average WER:     {results['average_wer']:.4f}\n")

    logger.info("TOP 20 MOST COMMON ERRORS:")
    logger.info(f"{'Ground Truth':<20} {'Transcribed':<20} {'Count':<10}")
    logger.info("-" * 50)

    for error in results['top_50_errors'][:20]:
        logger.info(
            f"{error['ground_truth']:<20} "
            f"{error['transcribed']:<20} "
            f"{error['count']:<10}"
        )

    logger.info("\n" + "="*80)
    logger.info(f"Full results (top 50 errors) saved to logs/coraa_analysis/")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
