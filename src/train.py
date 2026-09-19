import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import argparse, json, os, time, shutil
from dotenv import load_dotenv
load_dotenv()
_wandb_key = os.getenv('WANDB_API_KEY')
_is_rank0 = int(os.environ.get('RANK', '0')) == 0
# Importing wandb starts a local service unless disabled first.
os.environ.setdefault("WANDB_DISABLED", "true")
from transformers import TrainerCallback
import transformers, torch, wandb

from constants import TOKENIZER_NAME
from model import build_model_and_tokenizer
from dataset import get_dataset

# Limit CPU threads to prevent over-subscription due to add_prediction_mode
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

class ScalingRunCallback(TrainerCallback):
    """Records the validation trajectory, keeps only the best checkpoint, and
    stops once the run has clearly passed its minimum.

    The paper's protocol is to report the minimum held-out loss over training,
    so once the loss has stayed above its running minimum for `patience`
    consecutive evaluations there is nothing further to learn from the run.
    Patience needs to be generous: the Rand 15% + R2L configuration did not
    bottom out until epoch 104.
    """

    def __init__(self, run_meta, results_path, patience=15, min_evals=20, save_best_only=True):
        self.run_meta = run_meta
        self.results_path = results_path
        self.patience = patience
        self.min_evals = min_evals
        self.save_best_only = save_best_only
        self.trainer = None
        self.history = []
        self.best_loss = float('inf')
        self.best_epoch = None
        self.best_step = None
        self.best_ckpt = None
        self.evals_since_best = 0
        self.start_time = time.time()
        self._prior_wall = 0.0
        self.stopped_early = False

    def restore_from_results(self, checkpoint_step=None):
        """Reload the recorded trajectory so a resumed run does not forget its min."""
        if not self.results_path or not os.path.isfile(self.results_path):
            return
        with open(self.results_path) as f:
            prev = json.load(f)
        history = list(prev.get('history') or [])
        if checkpoint_step is not None:
            history = [h for h in history if h.get('step', 0) <= checkpoint_step]
        if not history:
            return
        self.history = history
        self.best_loss = float(prev.get('min_eval_loss', min(h['eval_loss'] for h in history)))
        self.best_epoch = prev.get('best_epoch')
        self.best_step = prev.get('best_step')
        self.best_ckpt = prev.get('best_checkpoint')
        self._prior_wall = float(prev.get('wall_time_s') or 0)
        if checkpoint_step is not None and history[-1]['step'] < (self.best_step or 0):
            rec = min(history, key=lambda h: h['eval_loss'])
            self.best_loss = float(rec['eval_loss'])
            self.best_epoch = rec['epoch']
            self.best_step = rec['step']
        self.evals_since_best = 0
        for rec in reversed(self.history):
            if rec.get('step') == self.best_step:
                break
            self.evals_since_best += 1
        print(
            f"[resume] restored {len(self.history)} evals, best {self.best_loss:.4f} "
            f"@ ep {self.best_epoch} ({self.evals_since_best}/{self.patience} since best)",
            flush=True,
        )

    # -- persistence ---------------------------------------------------
    def _summary(self, status):
        return {
            **self.run_meta,
            'status': status,
            'min_eval_loss': None if self.best_loss == float('inf') else self.best_loss,
            'best_epoch': self.best_epoch,
            'best_step': self.best_step,
            'best_checkpoint': self.best_ckpt,
            'epochs_completed': self.history[-1]['epoch'] if self.history else 0,
            'stopped_early': self.stopped_early,
            'wall_time_s': round(self._prior_wall + (time.time() - self.start_time), 1),
            'history': self.history,
        }

    def _write(self, status):
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        tmp = self.results_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self._summary(status), f, indent=2)
        os.replace(tmp, self.results_path)

    # -- hooks ---------------------------------------------------------
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Every rank runs this body. Checkpoint saving goes through Accelerate,
        # which blocks on a collective, so a rank-0-only save deadlocks DDP.
        # The eval metrics are identical on all ranks, so the best-so-far and
        # stop decisions stay in agreement without any extra communication;
        # only filesystem writes are restricted to rank 0.
        if not metrics:
            return
        loss = metrics.get('eval_loss')
        if loss is None:
            return
        is_main = state.is_world_process_zero
        epoch = round(state.epoch or 0, 3)
        if self.history and self.history[-1].get('step') == state.global_step:
            return
        self.history.append({
            'epoch': epoch,
            'step': state.global_step,
            'eval_loss': round(float(loss), 5),
            'elapsed_s': round(self._prior_wall + (time.time() - self.start_time), 1),
        })

        if loss < self.best_loss - 1e-6:
            self.best_loss = float(loss)
            self.best_epoch = epoch
            self.best_step = state.global_step
            self.evals_since_best = 0
            if self.save_best_only and self.trainer is not None:
                prev = self.best_ckpt
                self.trainer._save_checkpoint(self.trainer.model, trial=None)
                self.best_ckpt = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
                if is_main and prev and prev != self.best_ckpt and os.path.isdir(prev):
                    shutil.rmtree(prev, ignore_errors=True)
        else:
            self.evals_since_best += 1
            if self.evals_since_best >= self.patience and len(self.history) >= self.min_evals:
                self.stopped_early = True
                control.should_training_stop = True

        if is_main:
            self._write('running')
            print(
                f"[eval] epoch {epoch} step {state.global_step} loss {loss:.4f} "
                f"(best {self.best_loss:.4f} @ ep {self.best_epoch}, "
                f"{self.evals_since_best}/{self.patience} since best)",
                flush=True,
            )

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write('stopped_early' if self.stopped_early else 'completed')


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
    if _wandb_key and _is_rank0:
        os.environ.pop("WANDB_DISABLED", None)
        os.environ["WANDB_PROJECT"] = "data-aug-pretraining"
        try:
            wandb.login(key=_wandb_key, timeout=30)
        except Exception as e:
            print(f"[wandb] login failed ({e}); continuing with logging disabled", flush=True)
            os.environ["WANDB_DISABLED"] = "true"

    model, tokenizer = build_model_and_tokenizer(args)
    train_dataset, eval_dataset = get_dataset(args, tokenizer)

    n_total = sum(p.numel() for p in model.parameters())
    n_embed = model.get_input_embeddings().weight.numel()
    run_meta = {
        'run_name': args.run_name or os.path.basename(args.output_dir.rstrip('/')),
        'output_dir': args.output_dir,
        'params_total': n_total,
        'params_non_embedding': n_total - n_embed,
        'hidden_size': args.model_hidden_size,
        'num_layers': args.model_num_layers,
        'num_heads': args.model_num_attention_heads,
        'intermediate_size': args.model_intermediate_size,
        'unique_tokens_m': args.pretraining_tokens,
        'training_files_no': args.training_files_no,
        'epochs_requested': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'precision': args.precision,
        'global_batch_sequences': (
            args.batch_size_per_device * args.gradient_accumulation
            * int(os.environ.get('WORLD_SIZE', '1'))
        ),
        'augmentation': {
            'l2r_percent': args.l2r_percent,
            'max_next_i': args.max_next_i,
            'next_i_weighting': args.next_i_weighting,
            'mask_percent': args.mask_percent,
            'random_token_percent': args.random_token_percent,
            'psm_percent': args.psm_percent,
            'spm_percent': args.spm_percent,
        },
        'n_train_blocks': len(train_dataset),
        'n_eval_blocks': len(eval_dataset),
    }

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
        eval_strategy="steps" if args.eval_every_steps > 0 else "no",
        eval_steps=(args.eval_every_steps if args.eval_every_steps > 0 else None),
        save_strategy="no",
        load_best_model_at_end=False,
        seed=args.seed,
        data_seed=args.seed,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),

        max_grad_norm=1.0,
        remove_unused_columns=True,
        per_device_eval_batch_size=args.batch_size_per_device,
        # torch.compile fuses the cross-entropy over the 151k vocab, which is
        # both the throughput and the memory bottleneck at these widths.
        torch_compile=args.torch_compile,
        # No parameter is ever skipped, so the unused-parameter search is pure
        # overhead on every backward pass.
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,

        # ASYNC SETTINGS
        dataloader_num_workers=args.dataloader_num_workers,       # Number of CPU/Background processes
        dataloader_prefetch_factor=4,   # Each worker prepares 2 batches ahead of time
        dataloader_pin_memory=True,     # Speeds up CPU -> GPU transfer for the main model
        # train.py forces the 'spawn' start method, so re-creating workers at
        # every epoch boundary costs a full interpreter start per worker. In the
        # many-epoch regime studied here an epoch can be a few dozen steps, so
        # that respawn cost lands on the critical path constantly.
        dataloader_persistent_workers=(args.dataloader_num_workers > 0),

        report_to=("wandb" if not os.environ.get("WANDB_DISABLED") else "none"),
        logging_steps=10,
        logging_first_step=True,
        run_name=(args.run_name or os.path.basename(args.output_dir.rstrip('/'))),
    )

    callbacks = []
    if args.results_path:
        scaling_callback = ScalingRunCallback(
            run_meta=run_meta,
            results_path=args.results_path,
            patience=args.early_stop_patience,
            min_evals=args.early_stop_min_evals,
            save_best_only=args.save_best_only,
        )
        callbacks.append(scaling_callback)
    else:
        scaling_callback = None

    snapshot_callback = None
    if not args.save_best_only:
        snapshot_callback = AnalysisSnapshotCallback(
            snapshot_interval=args.snapshot_interval,
            save_final_only=args.save_final_only,
        )
        callbacks.append(snapshot_callback)

    trainer = transformers.trainer.Trainer(
        model=model, processing_class=tokenizer, args=training_args,
        callbacks=callbacks,
        train_dataset=train_dataset,
        eval_dataset=(eval_dataset if args.eval_every_steps > 0 else None),
    )
    if snapshot_callback is not None:
        snapshot_callback.trainer = trainer
    if scaling_callback is not None:
        scaling_callback.trainer = trainer
        if args.resume_from_checkpoint:
            ckpt_step = None
            state_path = os.path.join(args.resume_from_checkpoint, 'trainer_state.json')
            if os.path.isfile(state_path):
                with open(state_path) as f:
                    ckpt_step = json.load(f).get('global_step')
            scaling_callback.restore_from_results(checkpoint_step=ckpt_step)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--model-hidden-size', '-hidden-sz', type=int, default=512)
    parser.add_argument('--model-intermediate-size', '-int-sz', type=int, default=1536)
    parser.add_argument('--model-num-layers', '-layers', type=int, default=20)
    parser.add_argument('--model-num-attention-heads', '-att-heads', type=int, default=4)
    parser.add_argument('--model-max-length', '-maxlen', type=int, default=2048)

    parser.add_argument('--tokenizer', type=str, default=TOKENIZER_NAME,
                        help='HF tokenizer id. Paper runs use Qwen/Qwen3-Embedding-0.6B (Qwen2Tokenizer).')

    parser.add_argument('--output-dir', '-o', type=str, default='./training_output',
                        help='Directory where pretraining checkpoints and trainer state are saved.')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch-size-per-device', '-bs', type=int, default=8)
    parser.add_argument('--gradient-accumulation', '-ga', type=int, default=32,
                        help='Set so nproc * bs * ga = 512 (the paper global batch).')
    parser.add_argument('--warmup-steps', '-warm', type=int, default=100)
    parser.add_argument('--learning-rate', '-lr', type=float, default=6e-4)
    parser.add_argument("--lr-schedule", "-lr-sch", type=str, default="constant",
                        choices=["constant", "cooldown", "sine"])
    parser.add_argument("--min-lr-rate", type=float, default=0.01)
    parser.add_argument("--snapshot-interval", type=int, default=4)
    parser.add_argument("--save-final-only", action="store_true", default=False)
    parser.add_argument("--num-decay-steps", type=int, default=0)
    parser.add_argument('--weight-decay', '-wd', type=float, default=0.033)
    parser.add_argument('--training-files-no', type=int, default=3)
    parser.add_argument('--pretraining-tokens', '-pt', type=int, default=75)
    parser.add_argument('--val-files-no', type=int, default=1)
    parser.add_argument('--train-max-samples', '-train-max', type=int, default=None)
    parser.add_argument('--eval-max-samples', '-eval-max', type=int, default=None)

    parser.add_argument('--l2r-percent', type=float, default=100.0)
    parser.add_argument('--max-next-i', type=int, default=1)
    parser.add_argument('--next-i-weighting', type=str, default='uniform', choices=['uniform', 'exp'])
    parser.add_argument('--next-i-temperature', type=float, default=1.0)
    parser.add_argument('--mask-percent', type=float, default=0.0)
    parser.add_argument('--random-token-percent', type=float, default=0.0)
    parser.add_argument('--psm-percent', type=float, default=0.0)
    parser.add_argument('--spm-percent', type=float, default=0.0)
    parser.add_argument('--dataloader-num-workers', '-workers', type=int, default=4)
    parser.add_argument('--resume-from-checkpoint', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--results-path', type=str, default=None,
                        help='JSON file to record the validation trajectory and run metadata (Protocol B).')
    parser.add_argument('--save-best-only', action='store_true', default=False,
                        help='Keep only the lowest-validation-loss checkpoint (Protocol B).')
    parser.add_argument('--early-stop-patience', type=int, default=20,
                        help='Stop after this many consecutive evals above the running minimum.')
    parser.add_argument('--early-stop-min-evals', type=int, default=25)
    parser.add_argument('--torch-compile', action='store_true', default=False)
    parser.add_argument('--eval-every-steps', type=int, default=0,
                        help='Optimizer steps between evaluations. 0 disables online eval (Protocol A). '
                             'Protocol B uses 72.')
    return parser


if __name__ == '__main__':
    transformers.logging.set_verbosity_info()
    parser = argparse.ArgumentParser(description='Pretraining', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_train_args(parser)

    args = parser.parse_args()
    
    train(args)