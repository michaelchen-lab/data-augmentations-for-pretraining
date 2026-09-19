from huggingface_hub import snapshot_download
from datasets import Dataset
from pathlib import Path
import json, os, time, random
from itertools import chain

from constants import DATA_REPO_ID


def tokenize_raw(examples, tokenizer=None):
    return tokenizer(examples["text"])


def group_texts(examples, args=None):
    """Pack documents into contiguous ``model_max_length`` blocks; drop the remainder."""
    block_size = args.model_max_length
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def choose_prediction_mode(pred_modes: list, l2r_percent: float = 50.0, chosen_mode: str = None):
    if chosen_mode is not None:
        return chosen_mode
    l2r_percent = min(max(float(l2r_percent), 0.0), 100.0)
    if random.random() * 100 < l2r_percent:
        return "<|l2r_pred|>"
    return "<|r2l_pred|>"


def add_prediction_mode(input_ids, pred_modes: list = None, chosen_mode: str = None, l2r_percent: float = 50.0):
    """Prepend a direction token; reverse the sequence for R2L."""
    import torch

    mode = choose_prediction_mode(pred_modes, l2r_percent=l2r_percent, chosen_mode=chosen_mode)
    if mode == "<|l2r_pred|>":
        input_ids = torch.nn.functional.pad(input_ids, (1, 0), "constant", pred_modes[mode])[:, :-1]
    elif mode == "<|r2l_pred|>":
        input_ids = torch.nn.functional.pad(torch.flip(input_ids, dims=[1]), (1, 0), "constant", pred_modes[mode])[:, :-1]
    return input_ids


def _packed_cache_dir(args) -> Path:
    """Cache location for the tokenized+packed corpus.

    Only the corpus identity and the block size affect the packed token ids.
    Augmentation-specific special tokens are appended to the vocabulary but
    never appear in the raw text, so runs with different augmentation configs
    share a cache entry.
    """
    key = f"{args.pretraining_tokens}M_n{args.training_files_no}_L{args.model_max_length}"
    return Path("./pretraining_data/packed_cache") / key


def _build_and_cache(args, tokenizer, cache_dir: Path):
    from datasets import DatasetDict

    train_json, eval_json = _read_jsonl_shards(args)
    train_dataset = Dataset.from_dict({"text": [s["text"] for s in train_json]})
    train_dataset = train_dataset.map(
        tokenize_raw, batched=True, remove_columns=["text"],
        fn_kwargs={"tokenizer": tokenizer}, num_proc=16, desc="tokenize train",
    )
    train_dataset = train_dataset.map(
        group_texts, batched=True, batch_size=1000,
        fn_kwargs={"args": args}, num_proc=16, desc="pack train",
    )
    eval_dataset = Dataset.from_dict({"text": [s["text"] for s in eval_json]})
    eval_dataset = eval_dataset.map(
        tokenize_raw, batched=True, remove_columns=["text"],
        fn_kwargs={"tokenizer": tokenizer}, num_proc=8, desc="tokenize val",
    )
    eval_dataset = eval_dataset.map(
        group_texts, batched=True, batch_size=1000,
        fn_kwargs={"args": args}, num_proc=8, desc="pack val",
    )
    DatasetDict({"train": train_dataset, "eval": eval_dataset}).save_to_disk(str(cache_dir))
    (cache_dir / ".done").touch()
    return train_dataset, eval_dataset


def _read_jsonl_shards(args):
    token_folder = f"{args.pretraining_tokens}M"
    data_dir = Path(f"./pretraining_data/{token_folder}")
    val_path = Path("./pretraining_data/val_shard_00000000_processed.jsonl")
    train_json, eval_json = [], []
    for i in range(args.training_files_no):
        num_str = f"{i:08}"
        with open(data_dir / f"shard_{num_str}_processed.jsonl", "r") as file:
            for line in file:
                train_json.append(json.loads(line.strip()))
    with open(val_path, "r") as file:
        for line in file:
            eval_json.append(json.loads(line.strip()))
    return train_json[: args.train_max_samples], eval_json[: args.eval_max_samples]


def get_dataset(args, tokenizer=None):
    token_folder = f"{args.pretraining_tokens}M"
    data_dir = Path(f"./pretraining_data/{token_folder}")
    val_path = Path("./pretraining_data/val_shard_00000000_processed.jsonl")

    missing_train = not data_dir.is_dir()
    missing_val = not val_path.exists()
    if missing_train or missing_val:
        patterns = []
        if missing_train:
            patterns.append(f"{token_folder}/*")
        if missing_val:
            patterns.append("val_shard_*")
        snapshot_download(
            repo_id=DATA_REPO_ID,
            repo_type="dataset",
            local_dir="./pretraining_data",
            allow_patterns=patterns,
        )

    from datasets import load_from_disk

    cache_dir = _packed_cache_dir(args)
    rank = int(os.environ.get("RANK", "0"))
    if not (cache_dir / ".done").exists():
        if rank == 0:
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            print(f"[dataset] building packed cache at {cache_dir}", flush=True)
            _build_and_cache(args, tokenizer, cache_dir)
        else:
            waited = 0
            while not (cache_dir / ".done").exists():
                time.sleep(5)
                waited += 5
                if waited > 7200:
                    raise TimeoutError(f"Timed out waiting for packed cache {cache_dir}")
    dd = load_from_disk(str(cache_dir))
    train_dataset, eval_dataset = dd["train"], dd["eval"]
    cols = ["input_ids", "attention_mask", "labels"]
    train_dataset = train_dataset.with_format("torch", columns=cols)
    eval_dataset = eval_dataset.with_format("torch", columns=cols)
    print(
        f"[dataset] loaded packed cache: {len(train_dataset)} train / {len(eval_dataset)} eval blocks",
        flush=True,
    )
    return train_dataset, eval_dataset
