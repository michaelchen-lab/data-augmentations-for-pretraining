import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM
from sentence_transformers import SentenceTransformer
from transformers import TrainerCallback
from datasets import Dataset
import transformers, json, torch, wandb, argparse, os, time, copy, math
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from itertools import chain
_wandb_key = os.getenv('WANDB_API_KEY')
if _wandb_key:
    wandb.login(key=_wandb_key)
else:
    os.environ.setdefault("WANDB_DISABLED", "true")
os.environ["WANDB_PROJECT"] = "data-aug-pretraining"

from model import * 
from dataset import get_dataset

# Limit CPU threads to prevent over-subscription due to add_prediction_mode
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

class ConceptTrainer(transformers.Trainer):
    def get_eval_dataloader(self, eval_dataset=None):
        """Force 0 workers for evaluation to prevent VRAM spikes."""
        old_workers = self.args.dataloader_num_workers
        self.args.dataloader_num_workers = 0
        loader = super().get_eval_dataloader(eval_dataset)
        self.args.dataloader_num_workers = old_workers
        return loader

class AnalysisSnapshotCallback(TrainerCallback):
    def __init__(self, snapshot_interval=5, save_final_only=False):
        self.snapshot_interval = snapshot_interval
        self.save_final_only = bool(save_final_only)
        self.trainer = None

    def on_init_end(self, args, state, control, **kwargs):
        """Capture the trainer instance when it is initialized."""
        self.trainer = kwargs.get('trainer')

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.save_final_only:
            return
        current_epoch = int(round(state.epoch))

        if self.trainer and (current_epoch == 1 or current_epoch % self.snapshot_interval == 0):
            self.trainer._save_checkpoint(self.trainer.model, trial=None)

    def on_train_end(self, args, state, control, **kwargs):
        if self.save_final_only and self.trainer:
            self.trainer._save_checkpoint(self.trainer.model, trial=None)

def train(args):
    model, tokenizer = build_model_and_tokenizer(args)
    train_dataset, eval_dataset = get_dataset(args, tokenizer)

    if args.lr_schedule == "sine":
        lr_scheduler_type = "cosine_with_min_lr"
        lr_scheduler_kwargs = {"min_lr_rate": args.min_lr_rate}
    elif args.lr_schedule == "cooldown":
        if args.num_decay_steps <= 0:
            raise ValueError(
                "When --lr-schedule cooldown is set, --num-decay-steps must be a positive integer."
            )
        lr_scheduler_type = "warmup_stable_decay"
        lr_scheduler_kwargs = {
            "num_decay_steps": args.num_decay_steps,
            "decay_type": "1-sqrt",
            "min_lr_ratio": 0.0,
        }
    else:
        lr_scheduler_type = "constant_with_warmup"
        lr_scheduler_kwargs = {}

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,

        max_grad_norm=1.0, # default setting in TrainingArguments
        remove_unused_columns=(False if args.model_type == 'concept' else True), # for custom Dataset class

        # prediction_loss_only=True,     # Reduce memory: Do not store massive embedding outputs
        # eval_accumulation_steps=1,      # Reduce memory: Offload to CPU after every single step
        per_device_eval_batch_size=(int(args.batch_size_per_device / 2) if args.model_type == 'concept' else args.batch_size_per_device),   # Significantly lower batch size for eval
        # ASYNC SETTINGS
        dataloader_num_workers=args.dataloader_num_workers,       # Number of CPU/Background processes
        dataloader_prefetch_factor=4,   # Each worker prepares 2 batches ahead of time
        dataloader_pin_memory=True,     # Speeds up CPU -> GPU transfer for the main model

        report_to="wandb",            # Directs logs to W&B
        logging_steps=1,             # Log metrics every 10 steps
        logging_first_step=True,      # Useful to check if logging works immediately
        # run_name="concept-model-testing"
    )
    snapshot_callback = AnalysisSnapshotCallback(
        snapshot_interval=args.snapshot_interval,
        save_final_only=args.save_final_only,
    )
    trainer = transformers.trainer.Trainer(
        model=model, processing_class=tokenizer, args=training_args,
        callbacks=[snapshot_callback],
        train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    snapshot_callback.trainer = trainer
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--model-type', '-m-type', type=str, default='concept', choices=['default', 'concept', 'self-concept'])
    parser.add_argument('--model-hidden-size', '-hidden-sz', type=int, default=512)
    parser.add_argument('--model-intermediate-size', '-int-sz', type=int, default=1536) # OpenLM formula: 8/3 * hidden-sz. Round to nearest 256
    parser.add_argument('--model-num-layers', '-layers', type=int, default=20)
    parser.add_argument('--model-num-attention-heads', '-att-heads', type=int, default=4) # Keep hidden-sz / att-heads = 128
    parser.add_argument('--model-max-length', '-maxlen', type=int, default=2048) # 2048

    parser.add_argument('--embedding-model', '-embed-m', type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument('--output-embedding-size', '-output-embed-sz', type=int, default=1024)

    parser.add_argument('--output-dir', '-o', type=str, default='./training_output',
                        help='Directory where pretraining checkpoints and trainer state are saved.')
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--batch-size-per-device', '-bs', type=int, default=32) # 512 / 1024
    parser.add_argument('--gradient-accumulation', '-ga', type=int, default=8)
    parser.add_argument('--embedding-batch-size-multiplier', '-embed-bs-multi', type=int, default=16)
    parser.add_argument('--warmup-steps', '-warm', type=int, default=100) # 2000
    parser.add_argument('--learning-rate', '-lr', type=float, default=6e-4)
    parser.add_argument("--lr-schedule", "-lr-sch", type=str, default="sine", choices=["sine", "constant", "cooldown"])
    parser.add_argument("--min-lr-rate", type=float, default=0.01)
    parser.add_argument("--snapshot-interval", type=int, default=5)
    parser.add_argument("--save-final-only", action="store_true", default=False)
    parser.add_argument("--num-decay-steps", type=int, default=0)
    parser.add_argument('--weight-decay', '-wd', type=float, default=0.033)
    parser.add_argument('--training-files-no', type=int, default=3)
    parser.add_argument('--pretraining-tokens', '-pt', type=int, default=75)
    parser.add_argument('--val-files-no', type=int, default=1)
    parser.add_argument('--train-max-samples', '-train-max', type=int, default=None)
    parser.add_argument('--eval-max-samples', '-eval-max', type=int, default=None)

    parser.add_argument('--embed-loss-func', '-emb-loss', type=str, default='cosine', choices=['mse', 'cosine'])
    parser.add_argument('--embed-window-size', '-emb-win-sz', type=int, default=1)
    parser.add_argument('--l2r-percent', type=float, default=100.0)
    parser.add_argument('--max-next-i', type=int, default=1)
    parser.add_argument('--next-i-weighting', type=str, default='uniform', choices=['uniform', 'exp'])
    parser.add_argument('--next-i-temperature', type=float, default=1.0)
    parser.add_argument('--mask-percent', type=float, default=0.0)
    parser.add_argument('--random-token-percent', type=float, default=0.0)
    parser.add_argument('--psm-percent', type=float, default=0.0)
    parser.add_argument('--spm-percent', type=float, default=0.0)
    parser.add_argument('--no-embed-sliding-window-attn', '-eswa', action='store_true', default=False)
    parser.add_argument('--dataloader-num-workers', '-workers', type=int, default=4)
    parser.add_argument('--embed-dir', '-embed-dir', type=str, default=None)
    parser.add_argument('--save-embed-filename', '-save-embed-filename', type=str, default='./data/sample_embed.pt')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None)
    return parser


if __name__ == '__main__':
    transformers.logging.set_verbosity_info()
    parser = argparse.ArgumentParser(description='Pretraining', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_train_args(parser)

    args = parser.parse_args()
    
    train(args)