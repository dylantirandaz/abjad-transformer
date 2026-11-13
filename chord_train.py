"""Train one arm of the chord experiment.

Usage:
python chord_train.py                                        # the BPE baseline
python chord_train.py --setting k32_r1024 --order run_first  # a chord model
"""

import argparse
import functools
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chord_model import ORDERS, RUN_FIRST, ChordConfig, ChordTransformer
from chord_prepare import BPE_STREAM, PreparedMeta, load_meta, stream_path
from chords import HOST_COLUMN, RUN_COLUMN
from model import ModelConfig, Transformer
from train import Streams, TrainingRun, add_training_arguments, compute_loss, fit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="prepared")
    parser.add_argument("--out_dir", default="chord_checkpoints")
    parser.add_argument("--setting", default=None, help="chord setting such as k32_r1024; omit for the BPE baseline")
    parser.add_argument("--order", choices=ORDERS, default=RUN_FIRST)
    add_training_arguments(parser)
    return parser.parse_args(argv)


def arm_name(setting: str | None, order: str) -> str:
    return BPE_STREAM if setting is None else f"{setting}_{order}"


def load_stream(data_dir: str, stream: str, split: str, device: str) -> torch.Tensor:
    path = stream_path(data_dir, stream, split)
    try:
        return torch.from_numpy(np.load(path)).long().to(device)
    except OSError as error:
        message = f"Missing {path}. Run chord_prepare.py first."
        raise RuntimeError(message) from error


def next_position_streams(stream: torch.Tensor) -> Streams:
    """Aligned (inputs, targets): each position predicts the one after it."""
    return stream[:-1], stream[1:]


def compute_chord_loss(
    model: ChordTransformer, chords: torch.Tensor, next_chords: torch.Tensor, device: str
) -> torch.Tensor:
    """Loss per chord: the cost of the run plus the cost of the host."""
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        run_logits, host_logits = model(chords, next_chords)
    run_loss = F.cross_entropy(run_logits.float().flatten(0, 1), next_chords[..., RUN_COLUMN].flatten())
    host_loss = F.cross_entropy(host_logits.float().flatten(0, 1), next_chords[..., HOST_COLUMN].flatten())
    return run_loss + host_loss


def build_baseline(args: argparse.Namespace, meta: PreparedMeta) -> Transformer:
    config = ModelConfig(
        vocab_size=meta["bpe_vocab_size"],
        n_out=meta["bpe_vocab_size"],
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        causal=True,
    )
    return Transformer(config).to(args.device)


def build_chord_model(args: argparse.Namespace, meta: PreparedMeta) -> ChordTransformer:
    setting = meta["settings"][args.setting]
    config = ChordConfig(
        n_runs=setting["n_runs"],
        n_hosts=setting["n_hosts"],
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        order=args.order,
    )
    return ChordTransformer(config).to(args.device)


def save_arm_checkpoint(
    model: Transformer | ChordTransformer, args: argparse.Namespace, step: int, val_loss: float
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "setting": args.setting,
        "order": args.order,
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, out_dir / f"{arm_name(args.setting, args.order)}.pt")


def train_arm(args: argparse.Namespace) -> float:
    torch.manual_seed(args.seed)
    meta = load_meta(args.data_dir)
    label = arm_name(args.setting, args.order)
    stream = BPE_STREAM if args.setting is None else args.setting
    train_streams = next_position_streams(load_stream(args.data_dir, stream, "train", args.device))
    val_streams = next_position_streams(load_stream(args.data_dir, stream, "val", args.device))
    print(f"[{label}] {len(train_streams[0]):,} training positions")

    if args.setting is None:
        baseline = build_baseline(args, meta)
        print(f"[{label}] {baseline.count_parameters():,} parameters")
        save_baseline = functools.partial(save_arm_checkpoint, baseline, args)
        return fit(TrainingRun(label, baseline, compute_loss, train_streams, val_streams, save_baseline), args)

    chord_model = build_chord_model(args, meta)
    print(f"[{label}] {chord_model.count_parameters():,} parameters")
    save_chords = functools.partial(save_arm_checkpoint, chord_model, args)
    return fit(TrainingRun(label, chord_model, compute_chord_loss, train_streams, val_streams, save_chords), args)


if __name__ == "__main__":
    train_arm(parse_args())
