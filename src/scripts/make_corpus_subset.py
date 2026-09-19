"""Builds additional unique-token budgets by truncating the released corpora.

The published 37M / 75M / 300M folders are nested prefixes of one deterministic
shard ordering (verified by checksum: shard 0 is byte-identical across all
three, and the final shard of each is a line-truncation of the next size up).
New budgets can therefore be cut from the 300M set without re-downloading
CommonCrawl, and they stay on the same document ordering as the released sets.

Truncation is line-aligned and targeted by byte count; the exact token count is
measured afterwards when the corpus is tokenized and packed.

    python src/scripts/make_corpus_subset.py --tokens 19 --tokens 150
"""
import argparse
import json
import os
from pathlib import Path

SOURCE = Path("./pretraining_data/300M")
# Measured over the released 75M set: 442.1 MB of JSONL for 75M Qwen2 tokens.
BYTES_PER_TOKEN = 5.895


def shard_sizes():
    sizes = []
    i = 0
    while True:
        p = SOURCE / f"shard_{i:08}_processed.jsonl"
        if not p.exists():
            break
        sizes.append((p, p.stat().st_size))
        i += 1
    return sizes


def build(target_tokens_m: int):
    out_dir = Path(f"./pretraining_data/{target_tokens_m}M")
    if (out_dir / ".built").exists():
        print(f"[skip] {out_dir} already built")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    budget = int(target_tokens_m * 1e6 * BYTES_PER_TOKEN)
    used = 0
    written = 0

    for src, size in shard_sizes():
        if used >= budget:
            break
        dst = out_dir / src.name
        remaining = budget - used
        if size <= remaining:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)  # whole shard: hard-link, no copy
            used += size
            written += 1
            continue

        # Partial shard: copy whole JSON lines until the byte budget is hit.
        n = 0
        with open(src, "r") as fin, open(dst, "w") as fout:
            for line in fin:
                b = len(line.encode())
                if n + b > remaining:
                    break
                fout.write(line)
                n += b
        used += n
        written += 1
        break

    (out_dir / ".built").write_text(
        json.dumps({"target_tokens_m": target_tokens_m, "shards": written, "bytes": used})
    )
    print(f"[built] {out_dir}: {written} shards, {used / 1e6:.1f} MB "
          f"(~{used / BYTES_PER_TOKEN / 1e6:.1f}M tokens)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, action="append", required=True)
    a = ap.parse_args()
    for t in a.tokens:
        build(t)
