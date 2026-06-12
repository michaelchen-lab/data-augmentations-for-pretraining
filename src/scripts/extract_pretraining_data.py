#!/usr/bin/env python3
"""
Extract pretraining shards from DCLM-RefinedWeb to a target token budget.

This script:
- Downloads DCLM-refinedweb.paths.gz from CommonCrawl over HTTPS
- Processes shards starting at global-shard_03_of_10/local-shard_1_of_10, then continues
  through the remainder of global 03 (local 1 first, then 0, then 9..2 descending),
  then other global shards if the token budget is not met.
- Writes sequential output files under ``pretraining_data/<N>M/`` by default
  (``N`` = ``--tokens`` as an integer), e.g. ``pretraining_data/75M/shard_00000000_processed.jsonl``.
- Truncates the final output shard line-by-line to stay within token budget
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests
from transformers import AutoTokenizer

BASE_URL = "https://data.commoncrawl.org"
DATASET_PREFIX = "contrib/datacomp/DCLM-refinedweb"
PATHS_GZ_REL = f"{DATASET_PREFIX}/DCLM-refinedweb.paths.gz"

PATH_RE = re.compile(
    r"^(?:contrib/datacomp/DCLM-refinedweb/)?"
    r"global-shard_(?P<global_idx>\d+)_of_\d+/"
    r"local-shard_(?P<local_idx>\d+)_of_\d+/"
    r"shard_(?P<shard_num>\d+)_processed\.jsonl\.zstd$"
)


@dataclass(frozen=True)
class ShardPath:
    rel_path: str
    global_idx: int
    local_idx: int
    shard_num: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and extract DCLM-RefinedWeb shards to a target token count "
            "(in millions)."
        )
    )
    parser.add_argument(
        "--tokens",
        type=float,
        required=True,
        help="Target token count in millions (e.g. 75 for 75M tokens).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write shard_XXXXXXXX_processed.jsonl files. "
            "Defaults to pretraining_data/<N>M where N matches --tokens."
        ),
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("/tmp/dclm_download"),
        help="Temporary directory for downloaded/compressed files.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Hugging Face tokenizer name/path.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary files instead of deleting them after completion.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout in seconds for each request.",
    )
    parser.add_argument(
        "--start-global",
        type=int,
        default=3,
        help=(
            "1-based global shard index to start from (matches manifest "
            "`global-shard_XX_of_10`, e.g. 3 for global-shard_03_of_10)."
        ),
    )
    parser.add_argument(
        "--start-local",
        type=int,
        default=1,
        help=(
            "Local shard index to exhaust first (`local-shard_X_of_10`), "
            "then continue with remaining locals on that global shard."
        ),
    )
    return parser.parse_args()


def download_to_path(url: str, dest_path: Path, timeout_seconds: int) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with dest_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def parse_manifest(paths_gz_path: Path) -> list[ShardPath]:
    shard_paths: list[ShardPath] = []
    with gzip.open(paths_gz_path, "rt", encoding="utf-8") as f:
        for raw_line in f:
            rel_path = raw_line.strip()
            if not rel_path:
                continue
            match = PATH_RE.match(rel_path)
            if not match:
                continue
            shard_paths.append(
                ShardPath(
                    rel_path=rel_path,
                    global_idx=int(match.group("global_idx")),
                    local_idx=int(match.group("local_idx")),
                    shard_num=int(match.group("shard_num")),
                )
            )

    return shard_paths


def build_extraction_order(
    shards: list[ShardPath],
    *,
    start_global: int,
    start_local: int,
) -> list[ShardPath]:
    """
    Order shards for download:

    1. On ``start_global``, exhaust ``start_local`` first (shard indices ascending).

    2. On the same global shard: continue through remaining locals using the same
       wrap pattern as ``local-shard_9`` through ``local-shard_0`` when starting
       midway: locals ``0``, then ``9``, ``8``, …, ``2`` (skipping the start
       local, which was already consumed).

    3. Then every other ``global_idx`` not equal to ``start_global``, ascending
       by ``global_idx``; within each global, locals ``9``, ``8``, …, ``0``;
       within each bucket, ascending ``shard_num``.
    """
    if not (0 <= start_local <= 9):
        raise ValueError("--start-local must be between 0 and 9.")

    matching_global = [s for s in shards if s.global_idx == start_global]
    others = [s for s in shards if s.global_idx != start_global]

    def sorted_by_shard_num(items: list[ShardPath]) -> list[ShardPath]:
        return sorted(items, key=lambda s: s.shard_num)

    ordered: list[ShardPath] = []
    ordered.extend(
        sorted_by_shard_num([s for s in matching_global if s.local_idx == start_local])
    )
    continuation_locals = [0] + list(range(9, 1, -1))
    for loc in continuation_locals:
        if loc == start_local:
            continue
        ordered.extend(
            sorted_by_shard_num([s for s in matching_global if s.local_idx == loc])
        )

    for g in sorted({s.global_idx for s in others}):
        for loc in range(9, -1, -1):
            ordered.extend(sorted_by_shard_num([s for s in others if s.global_idx == g and s.local_idx == loc]))

    return ordered


def decompress_zstd(input_path: Path) -> Path:
    cmd = ["zstd", "-d", "-f", str(input_path)]
    subprocess.run(cmd, check=True)

    if input_path.suffix != ".zstd":
        raise ValueError(f"Expected .zstd file, got: {input_path}")
    return input_path.with_suffix("")


def tokens_for_text(text: str, tokenizer) -> int:
    return len(tokenizer(text)["input_ids"])


def write_truncated_shard(
    source_jsonl_path: Path,
    dest_jsonl_path: Path,
    tokenizer,
    token_budget_remaining: int,
) -> tuple[int, int, bool]:
    """
    Returns (tokens_written, rows_written, hit_budget).
    """
    tokens_written = 0
    rows_written = 0
    hit_budget = False

    with source_jsonl_path.open("r", encoding="utf-8") as src, dest_jsonl_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON at {source_jsonl_path}:{line_no}: {err}"
                ) from err

            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"Missing/invalid 'text' at {source_jsonl_path}:{line_no}"
                )

            line_tokens = tokens_for_text(text, tokenizer)
            if line_tokens > token_budget_remaining - tokens_written:
                hit_budget = True
                break

            dst.write(line)
            tokens_written += line_tokens
            rows_written += 1

    return tokens_written, rows_written, hit_budget


def cleanup_paths(paths: list[Path]) -> None:
    for p in paths:
        if p.exists():
            p.unlink()


def main() -> None:
    args = parse_args()

    if args.tokens <= 0:
        raise ValueError("--tokens must be > 0")

    if args.output_dir is None:
        args.output_dir = Path(f"pretraining_data/{int(args.tokens)}M")

    target_tokens = int(args.tokens * 1_000_000)
    output_dir = args.output_dir
    tmp_dir = args.tmp_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    manifest_gz = tmp_dir / "DCLM-refinedweb.paths.gz"
    manifest_url = f"{BASE_URL}/{PATHS_GZ_REL}"
    print(f"Downloading manifest: {manifest_url}")
    download_to_path(manifest_url, manifest_gz, timeout_seconds=args.timeout_seconds)

    manifest_shards = parse_manifest(manifest_gz)
    if not manifest_shards:
        raise RuntimeError("No usable shard paths found in manifest.")

    shard_paths = build_extraction_order(
        manifest_shards,
        start_global=args.start_global,
        start_local=args.start_local,
    )
    if not shard_paths:
        raise RuntimeError(
            f"No shard paths ordered for start global={args.start_global}, "
            f"local={args.start_local}."
        )

    print(
        "Ordering: "
        f"global-shard_{args.start_global:02d}_of_10/local-shard_{args.start_local}_of_10 first, "
        "then remainder of that global shard, then other globals."
    )
    print(f"Candidate shards in target order: {len(shard_paths)}")

    total_tokens_written = 0
    total_rows_written = 0
    downloaded_shards = 0
    budget_hit = False

    for src_idx, shard in enumerate(shard_paths, start=1):
        if total_tokens_written >= target_tokens:
            budget_hit = True
            break

        shard_url = f"{BASE_URL}/{shard.rel_path}"
        zstd_name = Path(shard.rel_path).name
        zstd_path = tmp_dir / zstd_name

        print(f"[{src_idx}/{len(shard_paths)}] Downloading {shard.rel_path}")
        download_to_path(shard_url, zstd_path, timeout_seconds=args.timeout_seconds)
        downloaded_shards += 1

        jsonl_path = decompress_zstd(zstd_path)
        output_idx = src_idx - 1
        out_name = f"shard_{output_idx:08d}_processed.jsonl"
        out_path = output_dir / out_name

        remaining = target_tokens - total_tokens_written
        file_tokens, file_rows, hit_budget = write_truncated_shard(
            source_jsonl_path=jsonl_path,
            dest_jsonl_path=out_path,
            tokenizer=tokenizer,
            token_budget_remaining=remaining,
        )

        total_tokens_written += file_tokens
        total_rows_written += file_rows
        print(
            f"[{src_idx}/{len(shard_paths)}] {shard.rel_path} -> {out_name} | "
            f"+{file_tokens / 1_000_000:.3f}M tok | "
            f"total {total_tokens_written / 1_000_000:.3f}M / "
            f"{target_tokens / 1_000_000:.3f}M"
        )

        if not args.keep_tmp:
            cleanup_paths([zstd_path, jsonl_path])

        if hit_budget:
            budget_hit = True
            break

    if not args.keep_tmp:
        cleanup_paths([manifest_gz])
    else:
        print(f"Kept temporary files in: {tmp_dir}")

    print("\nExtraction complete")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Downloaded shards: {downloaded_shards}")
    print(f"Total rows written: {total_rows_written}")
    print(f"Total tokens written: {total_tokens_written}")
    print(f"Target tokens: {target_tokens}")

    if not budget_hit and total_tokens_written < target_tokens:
        print(
            "Warning: exhausted all ordered manifest shards before reaching "
            "target token count."
        )

    if not args.keep_tmp and tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()


if __name__ == "__main__":
    main()
