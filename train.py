"""Train one of the three models.

Usage:
python train.py --task baseline   # normal character-level language model
python train.py --task skeleton   # language model over the consonant skeleton
python train.py --task vocalizer  # fills vowels back into a skeleton in one pass
"""

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from data import TASKS, Corpus, load_corpus, sample_windows, task_streams, task_vocab_sizes
from model import ModelConfig, Transformer

WARMUP_STEPS = 200
MIN_LR_FRACTION = 0.1
WEIGHT_DECAY = 0.1
GRAD_CLIP_NORM = 1.0
EVAL_BATCHES = 50

Streams = tuple[torch.Tensor, torch.Tensor]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--max_chars", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def build_model_config(args: argparse.Namespace, corpus: Corpus) -> ModelConfig:
    vocab_size, n_out = task_vocab_sizes(args.task, corpus)
    return ModelConfig(
        vocab_size=vocab_size,
        n_out=n_out,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        causal=args.task != "vocalizer",
    )


def learning_rate_at(step: int, base_lr: float, max_steps: int) -> float:
    if step < WARMUP_STEPS:
        return base_lr * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, max_steps - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (MIN_LR_FRACTION + (1.0 - MIN_LR_FRACTION) * cosine)


def should_evaluate(step: int, eval_interval: int, max_steps: int) -> bool:
    """Evaluate on the interval and always on the final step, so the last progress is never lost."""
    return step % eval_interval == 0 or step == max_steps


def compute_loss(model: Transformer, inputs: torch.Tensor, targets: torch.Tensor, device: str) -> torch.Tensor:
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        logits: torch.Tensor = model(inputs)
    return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())


@torch.no_grad()
def estimate_val_loss(model: Transformer, streams: Streams, args: argparse.Namespace) -> float:
    model.eval()
    losses = [
        float(compute_loss(model, *sample_windows(*streams, args.block_size, args.batch_size), args.device).item())
        for _ in range(EVAL_BATCHES)
    ]
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model: Transformer, args: argparse.Namespace, step: int, val_loss: float) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "task": args.task,
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, out_dir / f"{args.task}.pt")


def run_training_step(
    model: Transformer, optimizer: torch.optim.Optimizer, streams: Streams, step: int, args: argparse.Namespace
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_at(step, args.lr, args.max_steps)
    inputs, targets = sample_windows(*streams, args.block_size, args.batch_size)
    loss = compute_loss(model, inputs, targets, args.device)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()


def train(args: argparse.Namespace) -> float:
    torch.manual_seed(args.seed)
    corpus = load_corpus(args.data, args.device, args.max_chars)
    train_streams = task_streams(args.task, corpus.train)
    val_streams = task_streams(args.task, corpus.val)

    model = Transformer(build_model_config(args, corpus)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    print(f"[{args.task}] {model.count_parameters():,} parameters, {len(train_streams[0]):,} training tokens")

    best_val_loss = float("inf")
    started = time.time()
    for step in range(args.max_steps + 1):
        if should_evaluate(step, args.eval_interval, args.max_steps):
            val_loss = estimate_val_loss(model, val_streams, args)
            print(f"[{args.task}] step {step:5d}  val loss {val_loss:.4f}  ({time.time() - started:.0f}s)")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, args, step, val_loss)
        run_training_step(model, optimizer, train_streams, step, args)

    print(f"[{args.task}] done, best val loss {best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    train(parse_args())
