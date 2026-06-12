"""
Evaluate every checkpoint under a run directory using EleutherAI lm-evaluation-harness.

Writes one JSON per checkpoint under:
  results/<run_name>/zero_shot_eval/<checkpoint_name>.json

Also appends summary rows to:
  results/<run_name>/zero_shot_eval/summary.jsonl

Default task suite is tuned for small, data-constrained LMs and uses zero-shot eval.

Directional / next-token eval layout (``to_directional_lm_model``) is **off** unless you
pass ``--add-l2r-token`` and/or ``--add-next-1-pred-token`` (same knobs as
``eval_checkpoints_l2r.load_eval_model``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from model import to_directional_lm_model

try:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
except Exception as exc:  # pragma: no cover - import guard for optional dependency
    simple_evaluate = None
    HFLM = None
    _LM_EVAL_IMPORT_ERROR = exc
else:
    _LM_EVAL_IMPORT_ERROR = None


DEFAULT_TASKS = [
    "hellaswag",
    "piqa",
    # "arc_easy",
    "arc_challenge",
    # "openbookqa",
    "winogrande",
    # "boolq",
    "copa",
    # "lambada_openai",
    # "sciq",
    # babi removed: generate_until task with exact_match metric; unreliable at 150M scale.
    # "rte",             # GLUE RTE (binary entailment)
    # "commonsense_qa",
    # "blimp",
    # storycloze_2016 removed: relies on a legacy dataset script no longer
    # supported by current versions of the datasets library.
]

# Chance baseline by number of options (for centered-accuracy aggregate).
TASK_RANDOM_BASELINE = {
    "hellaswag": 0.25,
    "piqa": 0.50,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "openbookqa": 0.25,
    "winogrande": 0.50,
    "boolq": 0.50,
    "copa": 0.50,
    "sciq": 0.25,
    "rte": 0.50,
}


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _tokenizer_defines_token(tokenizer, text: str) -> bool:
    tid = tokenizer.convert_tokens_to_ids(text)
    if tid is None:
        return False
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and tid == unk_id:
        return False
    return True


def _detect_next_i_enabled(tokenizer) -> bool:
    return _tokenizer_defines_token(tokenizer, "<|next_1_pred|>")


def _detect_max_next_i_train(tokenizer) -> int:
    unk_id = getattr(tokenizer, "unk_token_id", None)
    n = 0
    while True:
        tid = tokenizer.convert_tokens_to_ids(f"<|next_{n + 1}_pred|>")
        if tid is None or (unk_id is not None and tid == unk_id):
            break
        n += 1
    return max(1, n)


def parse_checkpoint_step(name: str) -> int | None:
    m = re.search(r"checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else None


def discover_checkpoints(run_dir: Path, only_names: set[str] | None) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        if only_names is not None and child.name not in only_names:
            continue
        step = parse_checkpoint_step(child.name)
        if step is None:
            continue
        out.append((step, child))
    out.sort(key=lambda x: x[0])
    return out


def _pick_metric(doc: dict[str, Any], names: list[str]) -> tuple[float | None, str | None]:
    # First try exact keys.
    for key in names:
        if key in doc and isinstance(doc[key], (float, int)):
            val = float(doc[key])
            if np.isfinite(val):
                return val, key
    # Then try ",none" forms and key prefixes used by harness.
    for key, raw in doc.items():
        if not isinstance(raw, (float, int)):
            continue
        val = float(raw)
        if not np.isfinite(val):
            continue
        for name in names:
            if key == f"{name},none" or key.startswith(f"{name},"):
                return val, key
    return None, None


def _pick_stderr(doc: dict[str, Any], metric_key: str | None) -> float | None:
    candidates: list[str] = []
    if metric_key:
        if metric_key.endswith(",none"):
            candidates.append(metric_key.replace(",none", "_stderr,none"))
        if "," in metric_key:
            metric_name = metric_key.split(",", 1)[0]
            candidates.append(f"{metric_name}_stderr,none")
            candidates.append(f"{metric_name}_stderr")
        candidates.append(f"{metric_key}_stderr")
    for key in candidates:
        if key in doc and isinstance(doc[key], (float, int)):
            val = float(doc[key])
            if np.isfinite(val):
                return val
    for key, raw in doc.items():
        if not isinstance(raw, (float, int)):
            continue
        if "stderr" not in key:
            continue
        val = float(raw)
        if np.isfinite(val):
            return val
    return None


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def _calc_ci(values: list[float]) -> float:
    vals = [v for v in values if np.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    lower, upper = scipy.stats.t.interval(
        0.95,
        len(vals) - 1,
        loc=np.mean(vals),
        scale=scipy.stats.sem(vals),
    )
    return float((upper - lower) / 2.0)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zero-shot checkpoint eval with lm-evaluation-harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", "-r", required=True, type=Path, help="runs/<run_name> directory")
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="lm-eval task ids (space separated)",
    )
    p.add_argument(
        "--batch-size-per-device",
        "-bs",
        type=int,
        default=8,
        help="Harness batch_size argument per process",
    )
    p.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional lm-eval limit for smoke tests (fraction or count)",
    )
    p.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Few-shot examples (0 for zero-shot)",
    )
    p.add_argument(
        "--runs-per-task",
        "-n",
        type=int,
        default=1,
        help="Repeated eval passes per checkpoint/task for mean+CI aggregation",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for repeated eval passes; run k uses seed + k",
    )
    p.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional subset of checkpoint names, e.g. checkpoint-720 checkpoint-1440",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device string passed to harness (default auto from LOCAL_RANK / cuda)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Harness dtype for HF backend (e.g. float16, bfloat16, auto)",
    )
    p.add_argument(
        "--global-train-batch-size",
        type=int,
        default=512,
        help="Used only to record tokens_seen = step * global_batch * train_seq_len",
    )
    p.add_argument(
        "--train-seq-len",
        type=int,
        default=2048,
        help="Used only to record tokens_seen = step * global_batch * train_seq_len",
    )
    p.add_argument(
        "--add-l2r-token",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Install directional LM wrapper with <|l2r_pred|> layout when not using "
        "the next-1 layout, or set include_direction when using --add-next-1-pred-token "
        "(see eval_checkpoints_l2r.load_eval_model). Default: no <|l2r_pred|> prefix.",
    )
    p.add_argument(
        "--add-next-1-pred-token",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use fixed i=1 next-token eval layout (<|next_1_pred|> ...). Requires the "
        "tokenizer to define <|next_1_pred|>. Default: plain LM (no wrapper).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if _LM_EVAL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "lm-evaluation-harness is not importable. Install with "
            '`pip install "lm_eval[hf]"`.'
        ) from _LM_EVAL_IMPORT_ERROR

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir is not a directory: {run_dir}")

    only = set(args.checkpoints) if args.checkpoints else None
    checkpoints = discover_checkpoints(run_dir, only)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories under {run_dir}")

    run_name = run_dir.name
    out_dir = (args.results_root.resolve() / run_name / "zero_shot_eval")
    if get_rank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    maybe_barrier()
    summary_path = out_dir / "summary.jsonl"

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    for step, ckpt_path in checkpoints:
        ckpt_name = ckpt_path.name
        out_json = out_dir / f"{ckpt_name}.json"

        # Load existing results (if any) and determine which tasks still need running.
        existing_record: dict | None = None
        if out_json.exists():
            try:
                existing_record = json.loads(out_json.read_text())
            except Exception:
                existing_record = None
        existing_tasks = set(existing_record.get("tasks", {}).keys()) if existing_record else set()
        current_tasks = [t for t in args.tasks if t not in existing_tasks]

        if not current_tasks:
            if get_rank() == 0:
                print(f"Skip {ckpt_name} (all tasks present): {out_json}")
            maybe_barrier()
            continue

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), local_files_only=True)
        tokenizer_has_next_1 = _detect_next_i_enabled(tokenizer)
        max_next_i_train = _detect_max_next_i_train(tokenizer)
        tokenizer_has_l2r = _tokenizer_defines_token(tokenizer, "<|l2r_pred|>")

        if args.add_next_1_pred_token and not tokenizer_has_next_1:
            raise ValueError(
                "--add-next-1-pred-token was set but this tokenizer does not define a "
                "real id for <|next_1_pred|> (missing or maps to unk)."
            )
        if args.add_l2r_token and not tokenizer_has_l2r:
            raise ValueError(
                "--add-l2r-token was set but this tokenizer does not define a real id "
                "for <|l2r_pred|> (missing or maps to unk)."
            )

        if get_rank() == 0:
            print(
                f"Evaluating {ckpt_name} on {len(current_tasks)} tasks "
                f"(fewshot={args.num_fewshot}, device={device}, dtype={args.dtype}; "
                f"add_l2r_token={args.add_l2r_token}, add_next_1_pred_token={args.add_next_1_pred_token})"
            )

        # Build harness HF model. Optionally patch directional behavior on the loaded model
        # so multiple-choice likelihood scoring matches a chosen train-time token layout.
        lm = HFLM(
            pretrained=str(ckpt_path),
            tokenizer=str(ckpt_path),
            batch_size=args.batch_size_per_device,
            device=device,
            dtype=args.dtype,
        )
        if args.add_next_1_pred_token:
            to_directional_lm_model(
                lm.model,
                tokenizer,
                l2r_percent=100.0,
                max_next_i=1,
                fixed_i=1,
                include_direction=args.add_l2r_token,
            )
        elif args.add_l2r_token:
            to_directional_lm_model(
                lm.model,
                tokenizer,
                l2r_percent=100.0,
                max_next_i=1,
            )

        per_task_runs: dict[str, dict[str, list[float]]] = {
            t: {"acc_norm": [], "acc": [], "perplexity": [], "stderr": [], "centered_acc": []}
            for t in current_tasks
        }
        suite_runs: dict[str, list[float]] = {
            "mean_acc_norm": [],
            "mean_acc": [],
            "mean_centered_acc": [],
        }

        for run_idx in range(int(args.runs_per_task)):
            run_seed = int(args.seed) + run_idx
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            eval_out = simple_evaluate(
                model=lm,
                tasks=current_tasks,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size_per_device,
                limit=args.limit,
                log_samples=False,
            )
            raw_results = eval_out.get("results", {})

            run_acc_norm_values: list[float] = []
            run_acc_values: list[float] = []
            run_centered_values: list[float] = []

            for task_name in current_tasks:
                task_doc = raw_results.get(task_name, {})
                acc_norm, acc_norm_key = _pick_metric(task_doc, ["acc_norm"])
                acc, acc_key = _pick_metric(task_doc, ["acc"])
                ppl, _ppl_key = _pick_metric(task_doc, ["perplexity", "word_perplexity"])
                stderr = _pick_stderr(task_doc, acc_norm_key or acc_key)

                # If the top-level entry has no metrics, treat it as a group task and
                # average over subtasks whose key starts with "<task_name>_".
                if acc_norm is None and acc is None and ppl is None:
                    subtask_acc_norms = []
                    subtask_accs = []
                    for key, sdoc in raw_results.items():
                        if key.startswith(f"{task_name}_") and isinstance(sdoc, dict):
                            sn, _ = _pick_metric(sdoc, ["acc_norm"])
                            sa, _ = _pick_metric(sdoc, ["acc"])
                            if sn is not None:
                                subtask_acc_norms.append(sn)
                            if sa is not None:
                                subtask_accs.append(sa)
                    if subtask_acc_norms:
                        acc_norm = float(np.mean(subtask_acc_norms))
                    if subtask_accs:
                        acc = float(np.mean(subtask_accs))
                chosen_acc = acc_norm if acc_norm is not None else acc

                if acc_norm is not None:
                    per_task_runs[task_name]["acc_norm"].append(float(acc_norm))
                    run_acc_norm_values.append(float(acc_norm))
                if acc is not None:
                    per_task_runs[task_name]["acc"].append(float(acc))
                    run_acc_values.append(float(acc))
                if ppl is not None:
                    per_task_runs[task_name]["perplexity"].append(float(ppl))
                if stderr is not None:
                    per_task_runs[task_name]["stderr"].append(float(stderr))

                rb = TASK_RANDOM_BASELINE.get(task_name)
                if chosen_acc is not None and rb is not None and rb < 1.0:
                    centered = (chosen_acc - rb) / (1.0 - rb)
                    if np.isfinite(centered):
                        c = float(centered)
                        per_task_runs[task_name]["centered_acc"].append(c)
                        run_centered_values.append(c)

            mean_acc_norm_run = _safe_mean(run_acc_norm_values)
            if mean_acc_norm_run is not None:
                suite_runs["mean_acc_norm"].append(mean_acc_norm_run)
            mean_acc_run = _safe_mean(run_acc_values)
            if mean_acc_run is not None:
                suite_runs["mean_acc"].append(mean_acc_run)
            mean_centered_run = _safe_mean(run_centered_values)
            if mean_centered_run is not None:
                suite_runs["mean_centered_acc"].append(mean_centered_run)

        tasks_payload: dict[str, dict[str, Any]] = {}
        for task_name in current_tasks:
            tr = per_task_runs[task_name]
            acc_norm_mean = _safe_mean(tr["acc_norm"])
            acc_mean = _safe_mean(tr["acc"])
            ppl_mean = _safe_mean(tr["perplexity"])
            stderr_mean = _safe_mean(tr["stderr"])
            centered_mean = _safe_mean(tr["centered_acc"])
            chosen_runs = tr["acc_norm"] if tr["acc_norm"] else tr["acc"]
            tasks_payload[task_name] = {
                "acc_norm": acc_norm_mean,
                "acc_norm_ci": _calc_ci(tr["acc_norm"]),
                "acc_norm_runs": tr["acc_norm"],
                "acc": acc_mean,
                "acc_ci": _calc_ci(tr["acc"]),
                "acc_runs": tr["acc"],
                "perplexity": ppl_mean,
                "perplexity_ci": _calc_ci(tr["perplexity"]),
                "perplexity_runs": tr["perplexity"],
                "stderr": stderr_mean,
                "centered_acc": centered_mean,
                "centered_acc_ci": _calc_ci(tr["centered_acc"]),
                "centered_acc_runs": tr["centered_acc"],
                # Generic CI for primary task metric used by downstream plots.
                "ci": _calc_ci(chosen_runs),
            }

        tokens_seen = int(step) * int(args.global_train_batch_size) * int(args.train_seq_len)

        # If we loaded an existing record, merge the newly evaluated tasks into it
        # and recompute the suite-level summary across ALL tasks (old + new).
        if existing_record is not None:
            merged_tasks = dict(existing_record.get("tasks", {}))
            merged_tasks.update(tasks_payload)

            # Recompute suite summary from all tasks' acc_norm / acc runs.
            all_acc_norm_runs: list[float] = []
            all_acc_runs: list[float] = []
            all_centered_runs: list[float] = []
            for tdata in merged_tasks.values():
                all_acc_norm_runs.extend(tdata.get("acc_norm_runs") or [])
                all_acc_runs.extend(tdata.get("acc_runs") or [])
                all_centered_runs.extend(tdata.get("centered_acc_runs") or [])

            suite_runs["mean_acc_norm"] = all_acc_norm_runs
            suite_runs["mean_acc"] = all_acc_runs
            suite_runs["mean_centered_acc"] = all_centered_runs
            tasks_payload = merged_tasks
            all_tasks_evaluated = sorted(set(existing_record.get("tasks_evaluated", [])) | set(current_tasks))
        else:
            all_tasks_evaluated = list(args.tasks)

        mean_acc_norm_summary = _safe_mean(suite_runs["mean_acc_norm"])
        mean_acc_summary = _safe_mean(suite_runs["mean_acc"])
        mean_centered_summary = _safe_mean(suite_runs["mean_centered_acc"])
        record = {
            "run_dir": str(run_dir),
            "run": run_name,
            "checkpoint_path": str(ckpt_path),
            "checkpoint": ckpt_name,
            "global_step": int(step),
            "tokens_seen": int(tokens_seen),
            "num_fewshot": int(args.num_fewshot),
            "runs_per_task": int(args.runs_per_task),
            "base_seed": int(args.seed),
            "tasks_evaluated": all_tasks_evaluated,
            "tasks": tasks_payload,
            "summary": {
                "mean_acc_norm": mean_acc_norm_summary,
                "mean_acc_norm_ci": _calc_ci(suite_runs["mean_acc_norm"]),
                "mean_acc": mean_acc_summary,
                "mean_acc_ci": _calc_ci(suite_runs["mean_acc"]),
                "mean_centered_acc": mean_centered_summary,
                "mean_centered_acc_ci": _calc_ci(suite_runs["mean_centered_acc"]),
                "mean_acc_norm_runs": suite_runs["mean_acc_norm"],
                "mean_acc_runs": suite_runs["mean_acc"],
                "mean_centered_acc_runs": suite_runs["mean_centered_acc"],
                # Generic CI for default plotting summary metric.
                "ci": _calc_ci(suite_runs["mean_acc_norm"]),
            },
            "batch_size_per_device": int(args.batch_size_per_device),
            "device": device,
            "dtype": args.dtype,
            "add_l2r_token": bool(args.add_l2r_token),
            "add_next_1_pred_token": bool(args.add_next_1_pred_token),
            "tokenizer_has_l2r_pred": bool(tokenizer_has_l2r),
            "tokenizer_has_next_1_pred": bool(tokenizer_has_next_1),
            "max_next_i_train": int(max_next_i_train),
        }

        if get_rank() == 0:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
                f.write("\n")
            with open(summary_path, "a", encoding="utf-8") as sf:
                sf.write(json.dumps(record) + "\n")
            print(
                f"Wrote {out_json} "
                f"(mean_acc_norm={record['summary']['mean_acc_norm']}, "
                f"ci={record['summary']['ci']}, "
                f"mean_centered_acc={record['summary']['mean_centered_acc']})"
            )

        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        maybe_barrier()

    if get_rank() == 0:
        print(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
