# Demystifying Training-Time Augmentation for Data-Constrained Language Model Pretraining

We study three orthogonal categories of training-time data augmentation as regularizers for autoregressive (AR) language model pretraining in the data-constrained, multi-epoch regime:

1. **Token-level noise** — masking or random token replacement
2. **Sequence permutations** — right-to-left prediction and Fill-in-the-Middle (FIM)
3. **Target offset prediction** — predicting $x_{t+i}$ for $i > 1$

All experiments use a 150M-parameter Llama-based model trained on 75M tokens from DCLM-RefinedWeb for 100 epochs. The primary metric is held-out validation loss; zero-shot benchmarks via `lm-evaluation-harness` serve as a secondary signal.

---

## Repository structure

```
data_aug_pretraining/
├── src/
│   ├── train.py                        # Main pretraining script
│   ├── model.py                        # Model definition and augmentation wrappers
│   ├── dataset.py                      # Dataset loading and preprocessing
│   ├── eval_checkpoints_l2r.py         # Primary eval: validation loss sweep
│   ├── eval_checkpoints_lmharness.py   # Secondary eval: zero-shot benchmarks
│   └── scripts/
│       ├── extract_pretraining_data.py  # Download/extract DCLM-RefinedWeb shards
│       └── count_pretraining_tokens.py  # Verify token counts in extracted data
├── pyproject.toml
├── requirements.txt
└── README.md
```

Pretraining data is **not** included but is downloaded automatically from Hugging Face on first run (see [Pretraining data](#pretraining-data)). Results and figures are generated locally by running the scripts.

---

## Setup

```bash
git clone <this-repo>
cd data_aug_pretraining
pip install -e .
```

**WandB (optional).** If you have a WandB account, create a `.env` file with `WANDB_API_KEY=<your key>`. Otherwise, training automatically disables WandB logging.

**Hardware.** All paper runs used either a 2×H100 or 4×A100 setup. The commands below use `torchrun` for multi-GPU training; adjust `--nproc-per-node` for your setup.

---

## Pretraining data

Training reads token shards from `pretraining_data/75M/shard_XXXXXXXX_processed.jsonl` and validates on `pretraining_data/val_shard_00000000_processed.jsonl`. If these files are absent, `dataset.py` downloads them automatically from the Hugging Face dataset [here](https://huggingface.co/datasets/gashingriver5963/DCLM-pretraining-dataset).

To build the data locally from DCLM-RefinedWeb (requires network access and `zstd`):

```bash
# Extract ~75M tokens into pretraining_data/75M/
python src/scripts/extract_pretraining_data.py --tokens 75

# Verify token count
python src/scripts/count_pretraining_tokens.py --data-dir pretraining_data/75M --show-per-file
```

---

## Reproducing the experiments

The following commands reproduce the best-performing configuration from the paper: **Random 5% + R2L 50% + $i \leq 5$ exp.** All commands are run from the repository root. Adjust `--nproc-per-node` to match your GPU setup.

### Step 1: Stable-phase training (100 epochs)

```bash
torchrun --nproc-per-node 2 src/train.py \
  -m-type default --lr-schedule constant -lr 6e-4 -e 100 \
  --snapshot-interval 4 -bs 8 -ga 32 \
  --random-token-percent 5 --l2r-percent 50 \
  --max-next-i 5 --next-i-weighting exp \
  -o ./runs/random5-l2r50-i5-exp-fulle-lm
```

### Step 2: Validation loss sweep

Evaluates every checkpoint saved during stable-phase training and writes per-checkpoint JSON files to `results/random5-l2r50-i5-exp-fulle-lm/l2r_n1_eval/`. We run this using a 1xRTX3090 setup.

```bash
python src/eval_checkpoints_l2r.py \
  --run-dir runs/random5-l2r50-i5-exp-fulle-lm \
  --add-l2r-token --global-eval-batch-size 512 -ga 256
```

### Step 3: WSD decay phase

Identify the checkpoint with the lowest validation loss from Step 2 (epoch 68, step 4896 in the paper), then resume training with the cooldown schedule:

```bash
torchrun --nproc-per-node 2 src/train.py \
  -m-type default --lr-schedule cooldown --num-decay-steps 979 -lr 6e-4 -e 86 \
  -bs 8 -ga 32 \
  --random-token-percent 5 --l2r-percent 50 \
  --max-next-i 5 --next-i-weighting exp \
  --resume-from-checkpoint runs/random5-l2r50-i5-exp-fulle-lm/checkpoint-4896 \
  --save-final-only \
  -o ./runs/random5-l2r50-i5-exp-fulle-lm-wsd
```

Then evaluate the decay checkpoint's validation loss, following the same command as Step 2 but pointing `--run-dir` at `runs/random5-l2r50-i5-exp-fulle-lm-wsd`.

### Step 4: Zero-shot evaluation

```bash
python src/eval_checkpoints_lmharness.py \
  --run-dir runs/random5-l2r50-i5-exp-fulle-lm-wsd/checkpoint-4896 \
  --add-l2r-token -bs 8
```

---

## Key hyperparameters

| Parameter | Value |
|---|---|
| Architecture | Llama-based, decoder-only |
| Parameters | ~150M |
| Layers / heads / width | 20 / 4 / 512 |
| Context length | 2048 tokens |
| Tokenizer | Qwen2 (vocab 151,646) |
| Training tokens | 75M (DCLM-RefinedWeb) |
| Global batch size | 512 sequences |
| Peak learning rate | 6 × 10⁻⁴ |
| LR schedule (stable) | Constant with 100-step warmup |
| LR schedule (decay) | WSD 1−√· |
| Weight decay | 0.033 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Training epochs | 100 (stable) + ~20% decay |

For full architecture and optimizer details see Appendix B of the paper.

---

## Training argument reference

Run `python src/train.py --help` for the full argument list. The most commonly used flags for replication:

| Flag | Description |
|---|---|
| `--lr-schedule constant` | Use constant LR (stable phase); required for all ablation runs |
| `-e 100` | Train for 100 epochs |
| `--snapshot-interval 4` | Save checkpoint every 4 epochs |
| `-bs 8 -ga 32` | Batch size 8 per device, 32 gradient accumulation steps → global batch 512 (2 GPUs) |
| `--l2r-percent 50` | 50% L2R + 50% R2L (Category 2 permutation) |
| `--mask-percent N` | Mask N% of input tokens (Category 1) |
| `--random-token-percent N` | Replace N% of tokens randomly (Category 1) |
| `--psm-percent N --spm-percent N` | FIM: N% PSM + N% SPM (Category 2) |
| `--max-next-i N` | Offset prediction horizon (Category 3) |
| `--next-i-weighting exp` | Exponential weighting over offsets (recommended) |
