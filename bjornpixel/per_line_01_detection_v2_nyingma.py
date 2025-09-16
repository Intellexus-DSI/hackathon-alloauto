#!/usr/bin/env python3
"""
Tibetan Verse Detection Script
Analyzes Classical Tibetan texts to identify verse sections based on syllable patterns.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

ignore_chars: List[str] = ["/", "@", "#", ";", ":", "(", ")"]
meter_counts: List[int] = [7, 9, 11]


def clean_line(line: str) -> str:
    """
    Clean a line by removing ignored characters.
    """
    for char in ignore_chars:
        line = line.replace(char, "")
    # replace all sequences of more than one space with a space
    line = re.sub(r"\s+", " ", line).strip()
    return line


def analyze_file(
    filepath: str,
    window_size: int = 4,
    min_section_length: int = 4,
    min_probability: float = 0.7,
    min_gap: int = 0,
) -> List[Dict]:
    """Analyze a single file for verse sections."""

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    # Clean and filter lines
    cleaned_lines = []
    for line in lines:
        line = clean_line(line)
        if line:
            cleaned_lines.append(line)

    if len(cleaned_lines) < min_section_length:
        return []

    return lines


def main():
    """Main function to process all files in the corpus directory."""

    # corpus_dir = "/home/mike/data/zozerta/derge_tshad_split_corpus/train"
    # output_dir = "/home/mike/data/zozerta/verse_prose/scrolling_window"
    corpus_dir = "/home/guyb/hackathon-alloauto/bjornpixel/Sungbum_flat_shortened_test"
    output_dir = "/home/guyb/hackathon-alloauto/bjornpixel/per_line_output"

    if not os.path.exists(corpus_dir):
        print(f"Error: Directory {corpus_dir} not found!")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    processed_files = 0

    # Process all .txt files in the directory
    for filename in sorted(os.listdir(corpus_dir)):
        if filename.endswith(".txt"):
            print(filename)
            filepath = os.path.join(corpus_dir, filename)
            print(f"Processing {filename}...")

            file_segments = analyze_file(filepath)

            # Save individual file results
            output_filename = filename.replace(".txt", "_verse_analysis.json")
            output_filepath = os.path.join(output_dir, output_filename)

            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(file_segments, f, ensure_ascii=False, indent=2)

            if file_segments:
                print(f"  Found {len(file_segments)} segments -> {output_filename}")
            else:
                print(f"  No segments found -> {output_filename}")

            processed_files += 1

            # Progress update every 100 files
            if processed_files % 100 == 0:
                print(
                    f"Processed {processed_files} files")

    print("\nAnalysis complete!")
    print(f"Processed {processed_files} files")
    print(f"Individual results saved to {output_dir}/ directory")


if __name__ == "__main__":
    main()
