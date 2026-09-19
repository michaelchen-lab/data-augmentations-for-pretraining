"""Shared identifiers for paper replication.

The paper text calls the tokenizer Qwen2. Training loads it from
``Qwen/Qwen3-Embedding-0.6B`` (``tokenizer_class=Qwen2Tokenizer``). That is
the ID used by every released checkpoint; do not substitute
``Qwen/Qwen2-0.5B`` — the vocab length differs (151669 vs 151646 before
augmentation special tokens).
"""

TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"
DATA_REPO_ID = "michaelchenkj/DCLM-pretraining-dataset"
CHECKPOINTS_REPO_ID = "michaelchenkj/test-models"

# Packed-shard counts on disk for each unique-token budget.
CORPUS_SHARDS = {19: 1, 37: 2, 75: 3, 150: 5, 300: 10}

GLOBAL_BATCH_SEQUENCES = 512
SEQ_LEN = 2048
PEAK_LR = 6e-4
WEIGHT_DECAY = 0.033
WARMUP_STEPS = 100
