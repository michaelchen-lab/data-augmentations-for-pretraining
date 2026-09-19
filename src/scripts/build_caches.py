"""Pre-tokenizes and packs every corpus budget used in the paper."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from constants import CORPUS_SHARDS, TOKENIZER_NAME
from dataset import get_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, action="append", default=None)
    ap.add_argument("--max-length", type=int, default=2048)
    a = ap.parse_args()
    budgets = a.tokens or sorted(CORPUS_SHARDS)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("\nbudget  train_blocks  actual_tokens_M  val_blocks")
    for t in budgets:
        args = argparse.Namespace(
            pretraining_tokens=t,
            training_files_no=CORPUS_SHARDS[t],
            model_max_length=a.max_length,
            train_max_samples=None,
            eval_max_samples=None,
        )
        train, val = get_dataset(args, tokenizer)
        tok = len(train) * a.max_length / 1e6
        print(f"{t:>5}M  {len(train):>12,}  {tok:>15.2f}  {len(val):>10,}")


if __name__ == "__main__":
    main()
