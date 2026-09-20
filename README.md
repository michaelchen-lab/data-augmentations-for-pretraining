# Demystifying Training-Time Augmentation for Data-Constrained Language Model Pretraining

Code for the paper. We study three orthogonal categories of training-time data augmentation as regularizers for autoregressive language model pretraining in the data-constrained, multi-epoch regime:

1. **Token-level noise** — masking or random token replacement
2. **Sequence permutations** — right-to-left prediction and Fill-in-the-Middle (FIM)
3. **Target offset prediction** — predicting \(x_{t+i}\) for \(i > 1\)

All paper models are a 150M-parameter Llama (20 layers, hidden 512, 4 heads, FFN 1536, context 2048) trained on DCLM-RefinedWeb. The primary metric is held-out left-to-right validation loss. Zero-shot `lm-evaluation-harness` scores are a secondary signal.

Released checkpoints live in the Hugging Face dataset [`michaelchenkj/test-models`](https://huggingface.co/datasets/michaelchenkj/test-models) (`runs/` + `results/`). Do not mix Protocol A and Protocol B numbers: they are different training recipes (see below).

---

## Setup

```bash
git clone https://github.com/michaelchen-lab/data-augmentations-for-pretraining.git
cd data-augmentations-for-pretraining
pip install -r requirements.txt
```

Install a CUDA PyTorch build from [pytorch.org](https://pytorch.org) first. Paper runs used **PyTorch 2.11.0+cu128** and **transformers 5.15.0**. Pinning `transformers` matters: an older build was used for some original Table 1 checkpoints and moves val loss at the ~0.02 level.

**WandB (optional).** Put `WANDB_API_KEY` in a `.env` file, or training disables WandB automatically.

**Hardware.** Global batch is always **512 sequences × 2048 tokens**. Set `--nproc-per-node`, `-bs`, and `-ga` so `nproc × bs × ga = 512`.

| Protocol | Paper GPUs | Typical flags |
|---|---|---|
| A (Tables 1–3, decay) | 2×H100 or 4×A100 | `-bs 8 -ga 32` on 2 GPUs; `-bs 8 -ga 16` on 4 |
| B (§4.8, Appendix F) | 8×H100 | `-bs 8 -ga 8`, `--torch-compile` |

---

## Pretraining data

Training reads `pretraining_data/<N>M/shard_*_processed.jsonl` and validates on `pretraining_data/val_shard_00000000_processed.jsonl`. If those files are missing, `dataset.py` downloads them from [`michaelchenkj/DCLM-pretraining-dataset`](https://huggingface.co/datasets/michaelchenkj/DCLM-pretraining-dataset).

The 19M / 37M / 75M / 150M / 300M folders are nested prefixes of one shard ordering. To cut extra budgets from the released 300M set:

```bash
python src/scripts/make_corpus_subset.py --tokens 19 --tokens 150
```

To rebuild from CommonCrawl DCLM-RefinedWeb instead (needs network and `zstd`):

```bash
python src/scripts/extract_pretraining_data.py --tokens 75
python src/scripts/count_pretraining_tokens.py --data-dir pretraining_data/75M --show-per-file
```

**Tokenizer.** The paper describes a Qwen2 tokenizer. Training loads it from `Qwen/Qwen3-Embedding-0.6B` (`tokenizer_class=Qwen2Tokenizer`). That is the ID used by every released checkpoint. Do **not** switch to `Qwen/Qwen2-0.5B`: the vocab length differs (151669 vs 151646 before augmentation special tokens).

---

## Two training protocols

The paper reports two families. Commands for every table row are in [`configs/paper_runs.json`](configs/paper_runs.json). `scripts/run_paper.py` prints or launches them.

### Protocol A — Tables 1–3, WSD decay, downstream

Constant LR, **100 epochs**, checkpoint **every 4 epochs**, **no online eval**. After training, sweep snapshots with `eval_checkpoints_l2r.py` and pick the min-loss checkpoint. Decay runs resume that checkpoint with a \(1-\sqrt{\cdot}\) WSD cooldown (~20% of the resume step).

This is the original 2-GPU / 4×A100 family (`*-fulle-lm` on Hugging Face).

```bash
# List every Table 1–2 run
python scripts/run_paper.py --protocol a --list

# Best 3-category combination (Table 2)
python scripts/run_paper.py --protocol a --run random5-l2r50-i5-exp-fulle-lm --nproc 2

# Equivalent explicit command
torchrun --nproc-per-node 2 src/train.py \
  --lr-schedule constant -lr 6e-4 -e 100 \
  --snapshot-interval 4 --eval-every-steps 0 \
  -bs 8 -ga 32 \
  --random-token-percent 5 --l2r-percent 50 \
  --max-next-i 5 --next-i-weighting exp \
  -o ./runs/random5-l2r50-i5-exp-fulle-lm
```

Post-hoc validation loss on every snapshot:

```bash
python src/eval_checkpoints_l2r.py \
  --run-dir runs/random5-l2r50-i5-exp-fulle-lm \
  --add-l2r-token --global-eval-batch-size 512 -ga 256
```

WSD decay from the paper min (epoch 68, step 4896, 979 decay steps):

```bash
python scripts/run_paper.py --protocol decay \
  --run random5-l2r50-i5-exp-fulle-lm-wsd-from-4896 --nproc 2
```

Then evaluate the decay run directory the same way. Resume steps for all eight decay configs are in `configs/paper_runs.json` (Appendix decay-details table).

Residual dropout (after attention `o_proj` and MLP `down_proj`; default `--dropout 0`). Eval dropout-only with `--no-add-l2r-token`; eval the stack with `--add-l2r-token`.

```bash
python scripts/run_paper.py --protocol a --run baseline-dropout0.20 --nproc 4
python scripts/run_paper.py --protocol a --run dropout0.20-random5-l2r50-i5-exp-fulle-lm --nproc 4
```

Zero-shot (Table 4 five tasks; Appendix G ten tasks):

```bash
python src/eval_checkpoints_lmharness.py \
  --run-dir runs/random5-l2r50-i5-exp-fulle-lm-wsd-from-4896 \
  --add-l2r-token --suite paper5 -bs 8

python src/eval_checkpoints_lmharness.py \
  --run-dir runs/random5-l2r50-i5-exp-fulle-lm-wsd-from-4896 \
  --add-l2r-token --suite paper10 -bs 8
```

### Protocol B — §4.8 unique-token ladder and Appendix F seeds

**Different recipe.** 8×H100, `torch.compile`, online eval **every 72 steps**, `--save-best-only`, early-stop patience **20** after **25** evals, **100-epoch cap**. Reported “test loss” is `min_eval_loss` from the results JSON.

The 75M unique-token column in §4.8 is **spliced from Protocol A Tables 1–3**, not retrained under Protocol B. Appendix F seed 42 is a Protocol B *retrain* of the four headline configs; it is not the Table 1 run.

```bash
python scripts/run_paper.py --protocol unique --list
python scripts/run_paper.py --protocol unique --nproc 8 --dry-run

python scripts/run_paper.py --protocol seeds --list
python scripts/run_paper.py --protocol seeds --run t1a_S2_C3_rand5_r2l50_i5exp_s43 --nproc 8
```

19M / 37M / 150M / 300M unique-token data must be on disk first (`make_corpus_subset.py` or the HF dataset).

---

## Figure

```bash
python scripts/plot_scaling.py --out figures
```

Writes `figures/paper_scaling_unique.pdf` (val loss + unique-data multiplier for baseline, Random 15%, and 3-cat).

---

## Layout

```
configs/paper_runs.json          # every paper training run
scripts/run_paper.py             # launch / dry-run those runs
scripts/plot_scaling.py          # §4.8 figure
src/train.py                     # pretraining
src/model.py                     # Llama + augmentation wrappers
src/dataset.py                   # shard download, tokenize, pack
src/eval_checkpoints_l2r.py      # primary metric
src/eval_checkpoints_lmharness.py
src/scripts/extract_pretraining_data.py
src/scripts/make_corpus_subset.py
src/scripts/count_pretraining_tokens.py
src/scripts/build_caches.py
tests/test_clean_eval.py         # eval must be unaugmented L2R
```

Checkpoints and large result dumps are not in git. Download them from Hugging Face or train locally into `runs/` and `results/` (gitignored).

---

## Key hyperparameters

| Parameter | Value |
|---|---|
| Architecture | Llama decoder, tied embeddings |
| Parameters | 145.8M total / 68.2M non-embedding |
| Layers / heads / width / FFN | 20 / 4 / 512 / 1536 |
| Context | 2048 |
| Tokenizer | Qwen2 via `Qwen/Qwen3-Embedding-0.6B` |
| Unique tokens (main ablations) | 75M DCLM-RefinedWeb |
| Global batch | 512 sequences |
| Peak LR | \(6 \times 10^{-4}\) |
| Weight decay | 0.033 |
| Dropout | 0 (off). Residual \(p\in\{0.05,0.10,0.20\}\) in the dropout comparison |
| Optimizer | AdamW \(\beta=(0.9, 0.999)\), grad clip 1.0 |
| Protocol A | constant LR, 100-step warmup, snapshot every 4 epochs |
| Protocol A decay | WSD \(1-\sqrt{\cdot}\), ~20% of resume step |
| Protocol B | constant LR, eval every 72 steps, early-stop 20 / min 25 evals |

`python src/train.py --help` lists every flag.

---

## License and data

This repository is MIT licensed. Pretraining text comes from [DCLM-RefinedWeb](https://data.commoncrawl.org) / DataComp-LM; follow that dataset’s terms when redistributing derived shards. The Qwen tokenizer is subject to the Alibaba Qwen license on Hugging Face.
