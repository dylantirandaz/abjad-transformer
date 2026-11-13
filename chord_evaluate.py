"""Score every trained arm of the chord experiment the same way.

Usage:
python chord_evaluate.py --data_dir prepared --checkpoint_dir chord_checkpoints
"""

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from bpe import decode_tokens
from chord_model import ChordConfig, ChordTransformer
from chord_prepare import BPE_STREAM, load_meta, load_tokenizer, vocab_path
from chord_train import load_stream
from chords import HOST_COLUMN, RUN_COLUMN, decode_chords, load_chord_vocab
from console import configure_console
from evaluate import generate_ids, iterate_tiles, sum_bits
from model import ModelConfig, Transformer

SAMPLE_CHARS = 400
SPEED_TEST_CHARS = 4000
SPEED_WARMUP_PASSES = 16
CHARS_PER_THOUSAND = 1000

TextGenerator = Callable[[int], str]


@dataclass(frozen=True)
class LoadedArm:
    setting: str | None  # chord setting name, or None for the BPE baseline
    best_step: int
    model: Transformer | ChordTransformer


class ArmReport(TypedDict):
    parameters: int
    best_step: int
    passes_per_1k_chars: float
    bits_per_char: float
    bits_per_pass: float
    run_bits_share: float
    chars_per_second: float
    passes_per_second: float
    sample: str


class ChordEvalReport(TypedDict):
    val_chars: int
    arms: dict[str, ArmReport]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="prepared")
    parser.add_argument("--checkpoint_dir", default="chord_checkpoints")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def load_arm(path: Path, device: str) -> LoadedArm:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    setting: str | None = checkpoint["setting"]
    model: Transformer | ChordTransformer
    if setting is None:
        model = Transformer(ModelConfig(**checkpoint["model_config"]))
    else:
        model = ChordTransformer(ChordConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return LoadedArm(setting=setting, best_step=int(checkpoint["step"]), model=model.to(device).eval())


def chord_tile_nats(model: ChordTransformer, chords: torch.Tensor, next_chords: torch.Tensor) -> torch.Tensor:
    """Total cost of one batch of windows as a pair (run nats, host nats)."""
    device_type = chords.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        run_logits, host_logits = model(chords, next_chords)
    run_targets = next_chords[..., RUN_COLUMN].flatten()
    host_targets = next_chords[..., HOST_COLUMN].flatten()
    run_nats = F.cross_entropy(run_logits.float().flatten(0, 1), run_targets, reduction="sum")
    host_nats = F.cross_entropy(host_logits.float().flatten(0, 1), host_targets, reduction="sum")
    return torch.stack([run_nats, host_nats]).double()


@torch.no_grad()
def sum_chord_bits(model: ChordTransformer, chords: torch.Tensor, next_chords: torch.Tensor) -> tuple[float, float]:
    """Return (run bits, host bits) over the stream, every chord predicted exactly once."""
    tiles = iterate_tiles(chords, next_chords, model.config.block_size)
    totals = torch.stack([chord_tile_nats(model, x, y) for x, y in tiles]).sum(dim=0) / math.log(2)
    return float(totals[RUN_COLUMN].item()), float(totals[HOST_COLUMN].item())


@torch.no_grad()
def generate_chords(model: ChordTransformer, start: torch.Tensor, n_new: int, temperature: float) -> torch.Tensor:
    """Extend a (1, 1, 2) chord context by n_new chords. Returns (n_new + 1, 2)."""
    chords = start
    device_type = start.device.type
    for _ in range(n_new):
        context = chords[:, -model.config.block_size :]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            hidden = model.hidden(context)[:, -1]
        next_chord = model.sample_next(hidden.float(), temperature)
        chords = torch.cat([chords, next_chord[:, None]], dim=1)
    return chords[0]


def time_generation(generate: TextGenerator, n_passes: int) -> tuple[float, float]:
    """Return (characters per second, passes per second) for naive generation with no key-value cache.

    Detokenization is included in the time. Reporting both numbers separates how much text
    a pass yields from how long a pass takes.
    """
    generate(SPEED_WARMUP_PASSES)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    text = generate(n_passes)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return len(text) / elapsed, n_passes / elapsed


def build_report(
    arm: LoadedArm, n_passes: int, n_chars: int, bits: tuple[float, float], generate: TextGenerator
) -> ArmReport:
    """Combine the bit counts with a timed generation run and a sample."""
    total_bits = sum(bits)
    chars_per_pass = n_chars / n_passes
    chars_per_second, passes_per_second = time_generation(generate, round(SPEED_TEST_CHARS / chars_per_pass))
    return {
        "parameters": arm.model.count_parameters(),
        "best_step": arm.best_step,
        "passes_per_1k_chars": CHARS_PER_THOUSAND / chars_per_pass,
        "bits_per_char": total_bits / n_chars,
        "bits_per_pass": total_bits / n_passes,
        "run_bits_share": bits[RUN_COLUMN] / total_bits,
        "chars_per_second": chars_per_second,
        "passes_per_second": passes_per_second,
        "sample": generate(round(SAMPLE_CHARS / chars_per_pass)),
    }


def evaluate_baseline(arm: LoadedArm, model: Transformer, args: argparse.Namespace, tokenizer: Tokenizer) -> ArmReport:
    token_ids = load_stream(args.data_dir, BPE_STREAM, "val", args.device)
    n_chars = load_meta(args.data_dir)["n_val_chars"]
    bits = (0.0, sum_bits(model, token_ids[:-1], token_ids[1:]))

    def generate(n_passes: int) -> str:
        generated = generate_ids(model, int(token_ids[0].item()), n_passes, args.temperature)
        return decode_tokens(tokenizer, generated.cpu().numpy())

    return build_report(arm, len(token_ids), n_chars, bits, generate)


def evaluate_chords(
    arm: LoadedArm, model: ChordTransformer, args: argparse.Namespace, tokenizer: Tokenizer
) -> ArmReport:
    setting = str(arm.setting)
    chords = load_stream(args.data_dir, setting, "val", args.device)
    n_chars = load_meta(args.data_dir)["n_val_chars"]
    vocab = load_chord_vocab(str(vocab_path(args.data_dir, setting)))
    bits = sum_chord_bits(model, chords[:-1], chords[1:])

    def generate(n_passes: int) -> str:
        generated = generate_chords(model, chords[None, :1], n_passes, args.temperature)
        return decode_tokens(tokenizer, decode_chords(generated.cpu().numpy(), vocab))

    return build_report(arm, len(chords), n_chars, bits, generate)


def evaluate_arm(path: Path, args: argparse.Namespace, tokenizer: Tokenizer) -> ArmReport:
    arm = load_arm(path, args.device)
    if isinstance(arm.model, ChordTransformer):
        return evaluate_chords(arm, arm.model, args, tokenizer)
    return evaluate_baseline(arm, arm.model, args, tokenizer)


def evaluate(args: argparse.Namespace) -> ChordEvalReport:
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.data_dir)
    paths = sorted(Path(args.checkpoint_dir).glob("*.pt"))
    if not paths:
        message = f"No checkpoints in {args.checkpoint_dir}. Train an arm with chord_train.py first."
        raise RuntimeError(message)
    return {
        "val_chars": load_meta(args.data_dir)["n_val_chars"],
        "arms": {path.stem: evaluate_arm(path, args, tokenizer) for path in paths},
    }


def print_comparison(arms: dict[str, ArmReport], baseline: ArmReport) -> None:
    print(
        f"\n{'relative to BPE':<26}{'passes':>9}{'bits/char':>11}{'speed':>8}"
        f"{'time per pass':>15}{'run share of bits':>19}"
    )
    for name, arm in arms.items():
        if name == BPE_STREAM:
            continue
        print(
            f"{name:<26}{arm['passes_per_1k_chars'] / baseline['passes_per_1k_chars']:>9.3f}"
            f"{arm['bits_per_char'] / baseline['bits_per_char'] - 1:>+11.1%}"
            f"{arm['chars_per_second'] / baseline['chars_per_second']:>7.2f}x"
            f"{baseline['passes_per_second'] / arm['passes_per_second']:>14.2f}x{arm['run_bits_share']:>19.1%}"
        )


def print_report(report: ChordEvalReport) -> None:
    arms = report["arms"]
    print(f"\nValidation text: {report['val_chars']:,} characters\n")
    print(f"{'arm':<26}{'parameters':>12}{'passes/1k chars':>17}{'bits/char':>11}{'bits/pass':>11}{'chars/sec':>11}")
    for name, arm in arms.items():
        print(
            f"{name:<26}{arm['parameters']:>12,}{arm['passes_per_1k_chars']:>17.1f}"
            f"{arm['bits_per_char']:>11.4f}{arm['bits_per_pass']:>11.2f}{arm['chars_per_second']:>11.0f}"
        )
    if BPE_STREAM in arms:
        print_comparison(arms, arms[BPE_STREAM])
    for name, arm in arms.items():
        print(f"\n--- {name} sample (best checkpoint at step {arm['best_step']}) ---\n{arm['sample']}")


if __name__ == "__main__":
    configure_console()
    arguments = parse_args()
    evaluation = evaluate(arguments)
    print_report(evaluation)
    results_path = Path(arguments.checkpoint_dir) / "results.json"
    results_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"\nSaved {results_path}")
