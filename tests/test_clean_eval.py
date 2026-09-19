"""Checks that evaluation uses the clean left-to-right, i=1 view.

The augmented views exist to regularize training. The reported metric is
standard next-token prediction, so model.eval() must stop resampling
directions and offsets. Run with:

    PYTHONPATH=src python tests/test_clean_eval.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from constants import TOKENIZER_NAME
from model import build_model_and_tokenizer


def make_args(**overrides):
    ns = argparse.Namespace(
        tokenizer=TOKENIZER_NAME,
        model_hidden_size=128,
        model_intermediate_size=256,
        model_num_layers=2,
        model_num_attention_heads=1,
        model_max_length=64,
        l2r_percent=100.0,
        max_next_i=1,
        next_i_weighting="uniform",
        next_i_temperature=1.0,
        mask_percent=0.0,
        random_token_percent=0.0,
        psm_percent=0.0,
        spm_percent=0.0,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def capture(model, batch):
    """Return the input_ids that actually reach the transformer body."""
    seen = {}

    def hook(module, inputs):
        seen["ids"] = inputs[0].detach().clone()

    h = model.model.embed_tokens.register_forward_pre_hook(hook)
    with torch.no_grad():
        model(**batch)
    h.remove()
    return seen["ids"]


def main():
    args = make_args(l2r_percent=50.0, max_next_i=5, next_i_weighting="exp")
    model, tokenizer = build_model_and_tokenizer(args)
    ids = torch.randint(10, 100, (2, args.model_max_length))
    batch = {"input_ids": ids, "labels": ids.clone()}

    model.train()
    train_ids = capture(model, batch)
    model.eval()
    eval_ids = capture(model, batch)

    l2r_id = tokenizer.convert_tokens_to_ids("<|l2r_pred|>")
    next1_id = tokenizer.convert_tokens_to_ids("<|next_1_pred|>")
    assert eval_ids[0, 0].item() == l2r_id, eval_ids[0, 0].item()
    if next1_id is not None and next1_id != tokenizer.unk_token_id:
        assert eval_ids[0, 1].item() == next1_id, eval_ids[0, 1].item()
    print("ok: eval is L2R i=1; train ids differ from eval:", not torch.equal(train_ids, eval_ids))


if __name__ == "__main__":
    main()
