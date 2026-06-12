#!/usr/bin/env python3
"""
Count total tokens in pretraining JSONL shards with a Qwen2 tokenizer.

By default:
- Reads files in ./pretraining_data matching shard_*_processed.jsonl
- Excludes validation shards (val_shard_*)
- Counts tokens from each JSON line's "text" field using HF tokenizer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count total tokens in pretraining_data (excluding validation shard)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("pretraining_data"),
        help="Directory containing shard_*.jsonl files.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Hugging Face tokenizer name/path.",
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Include val_shard_* files (default excludes them).",
    )
    parser.add_argument(
        "--show-per-file",
        action="store_true",
        help="Print token totals per shard file.",
    )
    return parser.parse_args()


def should_skip_file(path: Path, include_validation: bool) -> bool:
    name = path.name
    if name.startswith("val_shard_") and not include_validation:
        return True
    return False


def count_tokens_in_file(path: Path, tokenizer) -> tuple[int, int]:
    token_total = 0
    rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {err}") from err
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Missing/invalid 'text' at {path}:{line_no}")
            token_total += len(tokenizer(text)["input_ids"])
            rows += 1
    return token_total, rows


def main() -> None:
    args = parse_args()

    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    candidates = sorted(args.data_dir.glob("*_processed.jsonl"))
    shard_files = [p for p in candidates if not should_skip_file(p, args.include_validation)]

    if not shard_files:
        raise FileNotFoundError(
            f"No shard files found in {args.data_dir}. "
            "Expected files like shard_00000000_processed.jsonl"
        )

    grand_total_tokens = 0
    grand_total_rows = 0

    for shard_path in shard_files:
        file_tokens, file_rows = count_tokens_in_file(shard_path, tokenizer)
        grand_total_tokens += file_tokens
        grand_total_rows += file_rows
        if args.show_per_file:
            print(f"{shard_path.name}: rows={file_rows}, tokens={file_tokens}")

    print("Token counting complete")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Data directory: {args.data_dir}")
    print(f"Shard files counted: {len(shard_files)}")
    print(f"Total rows counted: {grand_total_rows}")
    print(f"Total tokens: {grand_total_tokens}")


if __name__ == "__main__":
    main()
