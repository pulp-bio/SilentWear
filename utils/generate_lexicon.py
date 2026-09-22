# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate a letter-level lexicon file from label dictionaries.

Usage:
    python generate_lexicon.py [--mode {word,sentence}] [--output-dir DIR] [--output-file FILE]
"""

import argparse
import sys
from pathlib import Path


def phrase_to_letters(phrase: str) -> str:
    """Space-separated letters of a phrase, skipping whitespace."""
    return " ".join(c for c in phrase if c != " ")


def generate_lexicon(labels: dict) -> list[str]:
    return [f"{phrase} {phrase_to_letters(phrase)} |" for phrase in labels.values()]


def load_labels(mode: str, module: str) -> dict:
    try:
        mod = __import__(module, fromlist=["ORIGINAL_LABELS_WORDS", "ORIGINAL_LABELS_SENTENCES"])
    except ImportError as e:
        sys.exit(f"Cannot import '{module}': {e}")

    if mode == "sentence":
        return mod.ORIGINAL_LABELS_SENTENCES
    return mod.ORIGINAL_LABELS_WORDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate letter-level lexicon from label dictionary.")
    parser.add_argument("--mode", choices=["word", "sentence"], default="sentence",
                        help="Label modality (default: sentence)")
    parser.add_argument("--output-dir", default="./lexicon",
                        help="Output directory (default: ./lexicon)")
    parser.add_argument("--output-file", default="lexicon.txt",
                        help="Output filename (default: lexicon.txt)")
    parser.add_argument("--module", default="I_data_preparation.experimental_config",
                        help="Python module containing label dicts (default: utils.I_data_preparation.experimental_config)")
    args = parser.parse_args()

    labels = load_labels(args.mode, args.module)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_file

    lines = generate_lexicon(labels)
    out_path.write_text("\n".join(lines) + "\n")

    print(f"[SAVED] {out_path} with {len(lines)} entries")


if __name__ == "__main__":
    main()