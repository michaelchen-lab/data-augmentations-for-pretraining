"""
Evaluate every HuggingFace Trainer checkpoint under a run directory using
self-concept SSL with embed_window_size=1 (n=1) and 100% L2R only.

Training parity: defaults mirror ``src/train.py`` eval for ``self-concept``
(same ``per_device_eval_batch_size`` rule as training, dataloader workers,
prefetch, pin_memory; no fp16/bf16 unless you match your training run).

Writes one JSON per checkpoint under:
  results/<run_name>/l2r_n1_eval/<checkpoint_name>.json

Also appends a line to:
  results/<run_name>/l2r_n1_eval/summary.jsonl

Usage (from repo root):
  python src/eval_checkpoints_l2r.py --run-dir runs/bi-ssl

Multi-GPU (same launcher as training); only rank 0 writes JSON / summary:
  torchrun --nproc_per_node=2 src/eval_checkpoints_l2r.py --run-dir runs/bi-ssl

Optional 512-token *global* eval microbatch (2 GPUs -> 256 per device):
  torchrun --nproc_per_node=2 src/eval_checkpoints_l2r.py --run-dir runs/bi-ssl \\
    --global-eval-batch-size 512

Single GPU (e.g. RTX 3090): keep the *logical* eval batch at 512 but split forwards
(Trainer still reports the correct dataset-level mean eval loss):
  python src/eval_checkpoints_l2r.py --run-dir runs/bi-ssl \\
    --global-eval-batch-size 512 -ga 8
  # -> per_device_eval_batch_size=64 for each forward (512/8); 8x less VRAM than 512.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any, List, Tuple

os.environ.setdefault("WANDB_DISABLED", "true")

import torch
import torch.distributed as dist
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from dataset import get_dataset
from model import to_directional_lm_model, to_self_concept_model

# Match train.py: limit BLAS threads when workers load tensors
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _world_size_from_env() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L2R-only n=1 eval loss for all checkpoints in a runs/ subfolder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        "-r",
        type=str,
        required=True,
        help="Path to a single run folder (e.g. runs/bi-ssl) containing checkpoint-* dirs.",
    )
    p.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory for metrics (mirrors run name under here).",
    )
    p.add_argument(
        "--model-type",
        "-m-type",
        type=str,
        default="default",
        choices=["default", "self-concept"],
        help="Model objective to evaluate: 'default' uses LM loss; "
        "'self-concept' uses embedding SSL loss.",
    )
    p.add_argument(
        "--add-l2r-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to prepend <|l2r_pred|> at the start of each sequence. "
             "Set to match training: True if the model was trained with --l2r-percent < 100 "
             "(direction token was part of the input format), False if trained with "
             "--l2r-percent 100 (no direction token). l2r_percent is always 100%% at eval.",
    )
    p.add_argument("--model-hidden-size", "-hidden-sz", type=int, default=512)
    p.add_argument("--model-intermediate-size", "-int-sz", type=int, default=1536)
    p.add_argument("--model-num-layers", "-layers", type=int, default=20)
    p.add_argument("--model-num-attention-heads", "-att-heads", type=int, default=4)
    p.add_argument("--model-max-length", "-maxlen", type=int, default=2048)
    p.add_argument("--embed-loss-func", "-emb-loss", type=str, default="cosine", choices=["mse", "cosine"])
    p.add_argument(
        "--batch-size-per-device",
        "-bs",
        type=int,
        default=64,
        help="Logical per-device eval batch (same meaning as train.py -bs for self-concept). "
        "Trainer actually uses (logical // -ga) per forward. Ignored if --global-eval-batch-size is set.",
    )
    p.add_argument(
        "--global-eval-batch-size",
        type=int,
        default=None,
        help="Optional total eval microbatch across all GPUs (must divide WORLD_SIZE). "
        "Example: 512 with torchrun --nproc_per_node=2 -> 256 per device.",
    )
    p.add_argument(
        "--eval-gradient-accumulation-steps",
        "-ga",
        type=int,
        default=1,
        help="Split the *logical* per-device eval batch into this many smaller forwards "
        "(per_device_eval_batch_size = logical / steps). Reduces VRAM; HF averages "
        "batch losses with sample weighting so eval_loss matches one large forward. "
        "Logical per-device batch must be divisible by this value.",
    )
    p.add_argument("--training-files-no", type=int, default=3)
    p.add_argument(
        "--pretraining-tokens",
        "-pt",
        type=int,
        default=75,
        help="Training token budget in millions (selects pretraining_data/<N>M/ subfolder).",
    )
    p.add_argument("--val-files-no", type=int, default=1)
    p.add_argument("--train-max-samples", type=int, default=None)
    p.add_argument("--eval-max-samples", type=int, default=None)
    p.add_argument(
        "--dataloader-num-workers",
        "-workers",
        type=int,
        default=4,
        help="Match train.py default (set 0 to mimic single-threaded eval).",
    )
    p.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional subset of checkpoint dir names (e.g. checkpoint-720 checkpoint-1440).",
    )
    return p.parse_args()


def discover_checkpoints(run_dir: str, only_names: List[str] | None) -> List[Tuple[int, str]]:
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run-dir is not a directory: {run_dir}")
    found: List[Tuple[int, str]] = []
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if not os.path.isdir(path) or not name.startswith("checkpoint-"):
            continue
        suffix = name[len("checkpoint-") :]
        try:
            step = int(suffix)
        except ValueError:
            step = -1
        if only_names is not None and name not in only_names:
            continue
        found.append((step, path))
    found.sort(key=lambda x: x[0])
    return [(s, p) for s, p in found]


def tokenizer_from_checkpoint(checkpoint_path: str) -> AutoTokenizer:
    """Load the tokenizer saved alongside a Trainer checkpoint.

    ``src/train.py`` passes ``processing_class=tokenizer`` to the HF ``Trainer``, which
    always writes the tokenizer (including any added ``<|*_pred|>`` special tokens) into
    each ``checkpoint-*`` directory, so reading it back locally is sufficient.
    """
    return AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)


def _detect_next_i_enabled(tokenizer) -> bool:
    """True iff the tokenizer contains <|next_1_pred|> mapped to a non-unk id.

    Used by eval to decide whether to prepend [<|direction|>, <|next_1_pred|>] (for
    checkpoints trained with max_next_i > 1) vs. just [<|direction|>] (legacy n=1).
    """
    tid = tokenizer.convert_tokens_to_ids("<|next_1_pred|>")
    if tid is None:
        return False
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and tid == unk_id:
        return False
    return True


def _detect_max_next_i_train(tokenizer) -> int:
    """Infer the training-time max_next_i by counting <|next_{i}_pred|> tokens in the tokenizer.

    Returns 1 if none are present (legacy baseline).
    """
    unk_id = getattr(tokenizer, "unk_token_id", None)
    n = 0
    while True:
        tid = tokenizer.convert_tokens_to_ids(f"<|next_{n + 1}_pred|>")
        if tid is None or (unk_id is not None and tid == unk_id):
            break
        n += 1
    return max(1, n)


def build_eval_args(base: argparse.Namespace) -> Any:
    """Namespace compatible with get_dataset."""
    ns = argparse.Namespace(**vars(base))
    # Eval-only script: avoid loading train shards in get_dataset().
    ns.training_files_no = 0
    ns.model_type = base.model_type
    ns.embed_window_size = 1
    ns.l2r_percent = 100.0
    ns.output_embedding_size = 1024
    ns.embedding_batch_size_multiplier = 16
    ns.no_embed_sliding_window_attn = False
    ns.embed_dir = None
    ns.save_embed_filename = "./data/sample_embed.pt"
    ns.batch_size_per_device = base.batch_size_per_device
    ns.gradient_accumulation = 1
    return ns


def load_self_concept_model(
    checkpoint_path: str,
    tokenizer: AutoTokenizer,
    loss_func: str,
    add_l2r_token: bool,
) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(checkpoint_path)
    to_self_concept_model(
        model,
        output_embedding_size=None,
        loss_func=loss_func,
        shift=1,
        tokenizer=tokenizer,
        l2r_percent=100.0,
        add_l2r_token=add_l2r_token,
    )
    return model


def load_eval_model(
    checkpoint_path: str,
    tokenizer: AutoTokenizer,
    model_type: str,
    loss_func: str,
    add_l2r_token: bool,
    next_i_enabled: bool = False,
) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(checkpoint_path)
    if model_type == "default":
        if next_i_enabled:
            # Checkpoint was trained with max_next_i > 1: use fixed i=1 eval layout.
            # include_direction mirrors whether <|l2r_pred|> was part of the training format.
            to_directional_lm_model(
                model,
                tokenizer,
                l2r_percent=100.0,
                max_next_i=1,
                fixed_i=1,
                include_direction=add_l2r_token,
            )
        elif add_l2r_token:
            to_directional_lm_model(model, tokenizer, l2r_percent=100.0)
        return model
    if model_type == "self-concept":
        to_self_concept_model(
            model,
            output_embedding_size=None,
            loss_func=loss_func,
            shift=1,
            tokenizer=tokenizer,
            l2r_percent=100.0,
            add_l2r_token=add_l2r_token,
        )
        return model
    raise ValueError(f"Unsupported --model-type: {model_type}")


def resolve_logical_per_device_eval_batch_size(args: argparse.Namespace) -> int:
    """Logical per-device eval batch (before eval gradient accumulation splitting)."""
    ws = _world_size_from_env()
    if args.global_eval_batch_size is not None:
        g = args.global_eval_batch_size
        if g % ws != 0:
            raise ValueError(
                f"--global-eval-batch-size ({g}) must be divisible by WORLD_SIZE ({ws})"
            )
        return g // ws
    # train.py: self-concept uses full per_device_eval_batch_size (not halved like 'concept')
    return args.batch_size_per_device


def resolve_eval_micro_batch_size(
    logical_per_device: int, eval_gradient_accumulation_steps: int
) -> int:
    """Physical per_device_eval_batch_size passed to the Trainer."""
    ga = int(eval_gradient_accumulation_steps)
    if ga < 1:
        raise ValueError("--eval-gradient-accumulation-steps must be >= 1")
    if logical_per_device % ga != 0:
        raise ValueError(
            f"Logical per-device eval batch ({logical_per_device}) must be divisible by "
            f"--eval-gradient-accumulation-steps ({ga}). "
            f"Example: --global-eval-batch-size 512 -ga 8 -> logical/device=512, microbatch=64."
        )
    return logical_per_device // ga


def build_training_arguments(
    *,
    output_dir: str,
    per_device_eval_batch_size: int,
    dataloader_num_workers: int,
    local_rank: int,
) -> TrainingArguments:
    """TrainingArguments fields aligned with ``src/train.py`` for eval-only."""
    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=True,
        report_to="none",
        max_grad_norm=1.0,
        local_rank=local_rank,
    )
    if dataloader_num_workers > 0:
        kwargs["dataloader_prefetch_factor"] = 4
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    logical_per_device = resolve_logical_per_device_eval_batch_size(args)
    micro_per_device = resolve_eval_micro_batch_size(
        logical_per_device, args.eval_gradient_accumulation_steps
    )

    run_dir = os.path.abspath(args.run_dir)
    run_name = os.path.basename(run_dir.rstrip(os.sep))
    out_dir = os.path.join(os.path.abspath(args.results_root), run_name, "l2r_n1_eval")
    if get_rank() == 0:
        os.makedirs(out_dir, exist_ok=True)
    maybe_barrier()

    only = set(args.checkpoints) if args.checkpoints else None
    checkpoints = discover_checkpoints(run_dir, list(only) if only else None)
    if not checkpoints:
        if get_rank() == 0:
            print(f"No checkpoint-* directories found under {run_dir}", file=sys.stderr)
        sys.exit(1)

    eval_ns = build_eval_args(args)
    summary_path = os.path.join(out_dir, "summary.jsonl")

    first_ckpt = checkpoints[0][1]
    tokenizer = tokenizer_from_checkpoint(first_ckpt)
    next_i_enabled = _detect_next_i_enabled(tokenizer)
    max_next_i_train = _detect_max_next_i_train(tokenizer)
    _, eval_dataset = get_dataset(eval_ns, tokenizer)
    maybe_barrier()
    objective_label = "LM loss" if args.model_type == "default" else "self-concept SSL loss (n=1)"
    token_label = "<|l2r_pred|> prepended" if args.add_l2r_token else "<|l2r_pred|> omitted"
    next_i_label = (
        f"<|next_1_pred|> prepended (max_next_i_train={max_next_i_train})"
        if next_i_enabled
        else "no <|next_i_pred|> token"
    )
    if get_rank() == 0:
        print(
            f"Built eval dataset ({len(eval_dataset)} examples) using tokenizer from {first_ckpt}; "
            f"objective={objective_label}; {token_label}; {next_i_label}; "
            f"logical per-device eval batch={logical_per_device}, "
            f"eval_gradient_accumulation_steps={args.eval_gradient_accumulation_steps}, "
            f"Trainer per_device_eval_batch_size={micro_per_device} "
            f"(WORLD_SIZE={_world_size_from_env()}, "
            f"global logical batch={logical_per_device * _world_size_from_env()})"
        )

    for step, ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        out_json = os.path.join(out_dir, f"{ckpt_name}.json")
        if os.path.isfile(out_json):
            if get_rank() == 0:
                print(f"Skip {ckpt_name} (exists): {out_json}")
            maybe_barrier()
            continue

        if get_rank() == 0:
            print(f"Evaluating {ckpt_path} ({objective_label}) -> {out_json}")

        model = load_eval_model(
            ckpt_path,
            tokenizer,
            args.model_type,
            args.embed_loss_func,
            args.add_l2r_token,
            next_i_enabled=next_i_enabled,
        )

        tmp_train_dir = tempfile.mkdtemp(prefix=f"eval_l2r_{ckpt_name}_")
        try:
            training_args = build_training_arguments(
                output_dir=tmp_train_dir,
                per_device_eval_batch_size=micro_per_device,
                dataloader_num_workers=args.dataloader_num_workers,
                local_rank=local_rank,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )
            metrics = trainer.evaluate()
        finally:
            shutil.rmtree(tmp_train_dir, ignore_errors=True)

        maybe_barrier()

        eval_loss = metrics.get("eval_loss")
        record = {
            "run_dir": run_dir,
            "checkpoint_path": ckpt_path,
            "checkpoint": ckpt_name,
            "global_step": step,
            "eval_loss": float(eval_loss) if eval_loss is not None else None,
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (float, int))},
            "model_type": args.model_type,
            "objective": objective_label,
            "add_l2r_token": bool(args.add_l2r_token),
            "l2r_percent": 100.0,
            "embed_window_size": 1,
            "embed_loss_func": args.embed_loss_func,
            "logical_per_device_eval_batch_size": logical_per_device,
            "per_device_eval_micro_batch_size": micro_per_device,
            "eval_gradient_accumulation_steps": args.eval_gradient_accumulation_steps,
            "world_size_env": _world_size_from_env(),
            "world_size_runtime": get_world_size(),
            "global_logical_eval_batch": logical_per_device * get_world_size(),
            "global_eval_batch_size_arg": args.global_eval_batch_size,
            "batch_size_per_device_arg": args.batch_size_per_device,
            "dataloader_num_workers": args.dataloader_num_workers,
            "dataloader_prefetch_factor": (4 if args.dataloader_num_workers > 0 else None),
            "dataloader_pin_memory": torch.cuda.is_available(),
            "remove_unused_columns": True,
            "next_i_enabled": bool(next_i_enabled),
            "max_next_i_train": int(max_next_i_train),
        }
        if get_rank() == 0:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
                f.write("\n")
            with open(summary_path, "a", encoding="utf-8") as sf:
                sf.write(json.dumps(record) + "\n")

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        maybe_barrier()

    if get_rank() == 0:
        print(f"Done. Wrote per-checkpoint JSON and summary under {out_dir}")


if __name__ == "__main__":
    main()
