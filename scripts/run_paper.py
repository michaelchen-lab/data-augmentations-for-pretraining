#!/usr/bin/env python3
"""Launch paper training runs from configs/paper_runs.json.

    python scripts/run_paper.py --list
    python scripts/run_paper.py --protocol a --dry-run --run baseline-fulle-lm
    python scripts/run_paper.py --protocol a --run random5-l2r50-i5-exp-fulle-lm --nproc 2
    python scripts/run_paper.py --protocol unique --nproc 8
    python scripts/run_paper.py --protocol seeds --run t1a_S2_C3_rand5_r2l50_i5exp_s43
    python scripts/run_paper.py --protocol decay --run random5-l2r50-i5-exp-fulle-lm-wsd-from-4896

``nproc * --bs * --ga`` must equal 512. Defaults: 2 GPUs → bs=8, ga=32; 8 GPUs → bs=8, ga=8.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "paper_runs.json"
GLOBAL_BATCH = 512
AUG_FLAGS = {
    "l2r_percent": "--l2r-percent",
    "max_next_i": "--max-next-i",
    "next_i_weighting": "--next-i-weighting",
    "mask_percent": "--mask-percent",
    "random_token_percent": "--random-token-percent",
    "psm_percent": "--psm-percent",
    "spm_percent": "--spm-percent",
    "dropout": "--dropout",
}


def load_config():
    return json.loads(CONFIG.read_text())


def iter_jobs(cfg, protocol: str):
    if protocol == "a":
        for run in cfg["protocol_a"]["runs"]:
            yield "a", run
    elif protocol == "decay":
        by_name = {r["name"]: r for r in cfg["protocol_a"]["runs"]}
        for run in cfg["decay"]["runs"]:
            yield "decay", {**run, "aug": by_name[run["stable"]]["aug"]}
    elif protocol == "unique":
        for run in cfg["protocol_b"]["unique_token_runs"]:
            yield "unique", run
    elif protocol == "seeds":
        for run in cfg["protocol_b"]["seed_runs"]:
            yield "seeds", run
    else:
        raise ValueError(protocol)


def grad_accum(nproc: int, bs: int) -> int:
    ga = GLOBAL_BATCH // (nproc * bs)
    if ga < 1 or nproc * bs * ga != GLOBAL_BATCH:
        raise ValueError(
            f"nproc={nproc} bs={bs} cannot make global batch {GLOBAL_BATCH}; "
            "adjust --nproc / --bs so nproc * bs * ga = 512."
        )
    return ga


def command_for(cfg, protocol, run, nproc, bs, ga):
    arch = cfg["architecture"]
    train_py = str(REPO / "src" / "train.py")
    cmd = [
        shutil.which("torchrun") or "torchrun",
        "--nproc-per-node", str(nproc),
        train_py,
        "-lr", "6e-4",
        "-bs", str(bs),
        "-ga", str(ga),
        "-hidden-sz", str(arch["hidden_size"]),
        "-layers", str(arch["num_layers"]),
        "-att-heads", str(arch["num_heads"]),
        "-int-sz", str(arch["intermediate_size"]),
        "--precision", "bf16",
        "--seed", str(run.get("seed", 42)),
        "--run-name", run["name"],
    ]

    if protocol == "a":
        pa = cfg["protocol_a"]
        out = REPO / "runs" / run["name"]
        cmd += [
            "--lr-schedule", "constant",
            "-e", str(pa["epochs"]),
            "-pt", str(pa["pretraining_tokens"]),
            "--training-files-no", str(pa["training_files_no"]),
            "--snapshot-interval", str(pa["snapshot_interval"]),
            "--eval-every-steps", "0",
            "-o", str(out),
        ]
    elif protocol == "decay":
        pa = cfg["protocol_a"]
        stable_dir = REPO / "runs" / run["stable"] / f"checkpoint-{run['resume_step']}"
        out = REPO / "runs" / run["name"]
        cmd += [
            "--lr-schedule", "cooldown",
            "--num-decay-steps", str(run["num_decay_steps"]),
            "-e", str(run["epochs"]),
            "-pt", str(pa["pretraining_tokens"]),
            "--training-files-no", str(pa["training_files_no"]),
            "--eval-every-steps", "0",
            "--save-final-only",
            "--resume-from-checkpoint", str(stable_dir),
            "-o", str(out),
        ]
    else:
        pb = cfg["protocol_b"]
        tokens = run.get("tokens_m", 75)
        files_no = run.get("training_files_no", 3)
        if protocol == "unique":
            out = REPO / "runs" / "t2a" / run["name"]
            res = REPO / "results" / "t2a" / f"{run['name']}.json"
        else:
            sub = "t1a" if run["seed"] == 42 else "seeds"
            out = REPO / "runs" / sub / run["name"]
            res = REPO / "results" / sub / f"{run['name']}.json"
        cmd += [
            "--lr-schedule", "constant",
            "-e", str(pb["epochs"]),
            "-pt", str(tokens),
            "--training-files-no", str(files_no),
            "--eval-every-steps", str(pb["eval_every_steps"]),
            "--save-best-only",
            "--early-stop-patience", str(pb["early_stop_patience"]),
            "--early-stop-min-evals", str(pb["early_stop_min_evals"]),
            "--torch-compile",
            "--results-path", str(res),
            "-o", str(out),
        ]

    for key, flag in AUG_FLAGS.items():
        if key in run.get("aug", {}):
            cmd += [flag, str(run["aug"][key])]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=["a", "decay", "unique", "seeds"], default="a")
    ap.add_argument("--run", action="append", default=None, help="Run name; repeatable. Default: all in protocol.")
    ap.add_argument("--nproc", type=int, default=None, help="GPUs. Default: 2 for Protocol A, 8 for Protocol B.")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    jobs = list(iter_jobs(cfg, a.protocol))
    if a.run:
        wanted = set(a.run)
        jobs = [(p, r) for p, r in jobs if r["name"] in wanted]
        found = {r["name"] for _, r in jobs}
        missing = wanted - found
        if missing:
            print(f"unknown run(s) for protocol {a.protocol}: {sorted(missing)}", file=sys.stderr)
            return 1

    if a.list:
        for _, r in jobs:
            extra = r.get("note") or ""
            print(f"{r['name']:48} {extra}")
        print(f"{len(jobs)} runs")
        return 0

    nproc = a.nproc
    if nproc is None:
        nproc = 2 if a.protocol in ("a", "decay") else 8
    ga = grad_accum(nproc, a.bs)

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    for i, (protocol, run) in enumerate(jobs, 1):
        cmd = command_for(cfg, protocol, run, nproc, a.bs, ga)
        tag = f"[{i}/{len(jobs)}] {run['name']}"
        print(tag)
        print(" ", " ".join(cmd))
        if a.dry_run:
            continue
        out_dir = Path(cmd[cmd.index("-o") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, cwd=REPO, env=env)
        if proc.returncode != 0:
            print(f"{tag} FAILED rc={proc.returncode}", file=sys.stderr)
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
