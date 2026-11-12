"""Compare the abjad system (skeleton model + vocalizer) against the baseline model.

Usage:
python evaluate.py --data data/input.txt --checkpoint_dir checkpoints
"""

import argparse
import json
import math
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F

from abjad import BOS, join_text
from data import TASKS, Corpus, Split, load_corpus, task_streams
from model import ModelConfig, Transformer

EVAL_BATCH_WINDOWS = 64
SAMPLE_CHARS = 400
SPEED_TEST_CHARS = 2000
SPEED_WARMUP_CHARS = 64
PREVIEW_CHARS = 300
WHITESPACE = re.compile(r"\s")

Models = dict[str, Transformer]
TextGenerator = Callable[[Models, Corpus, int, float], str]


class BitsReport(TypedDict):
    baseline_bpc: float
    skeleton_bpc: float
    vocalizer_bpc: float
    abjad_bpc: float
    skeleton_bits_per_skeleton_token: float


class VocalizerReport(TypedDict):
    slot_accuracy: float
    word_accuracy: float
    preview_original: str
    preview_skeleton: str
    preview_reconstructed: str


class Report(TypedDict):
    val_chars: int
    n_vowel_run_classes: int
    skeleton_fraction: float
    parameters: dict[str, int]
    bits: BitsReport
    vocalizer: VocalizerReport
    chars_per_second: dict[str, float]
    samples: dict[str, str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--baseline_dir", default=None, help="load the baseline from here instead of --checkpoint_dir")
    parser.add_argument("--max_chars", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def resolve_checkpoint_dir(task: str, args: argparse.Namespace) -> str:
    """Let the baseline live in its own directory so one abjad system can be scored against several baselines."""
    if task == "baseline" and args.baseline_dir:
        return str(args.baseline_dir)
    return str(args.checkpoint_dir)


def load_model(checkpoint_dir: str, task: str, device: str) -> Transformer:
    path = Path(checkpoint_dir) / f"{task}.pt"
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except OSError as error:
        message = f"Missing checkpoint {path}. Train it with: python train.py --task {task}"
        raise RuntimeError(message) from error
    model = Transformer(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def iterate_tiles(
    inputs: torch.Tensor, targets: torch.Tensor, block_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (inputs, targets) batches of windows that cover both streams exactly once."""
    n_full = len(inputs) // block_size
    cut = n_full * block_size
    full_inputs = inputs[:cut].view(n_full, block_size)
    full_targets = targets[:cut].view(n_full, block_size)
    for start in range(0, n_full, EVAL_BATCH_WINDOWS):
        yield full_inputs[start : start + EVAL_BATCH_WINDOWS], full_targets[start : start + EVAL_BATCH_WINDOWS]
    if cut < len(inputs):
        yield inputs[cut:].unsqueeze(0), targets[cut:].unsqueeze(0)


def forward_logits(model: Transformer, inputs: torch.Tensor) -> torch.Tensor:
    device_type = inputs.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        logits: torch.Tensor = model(inputs)
    return logits.float()


@torch.no_grad()
def sum_bits(model: Transformer, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Total bits the model needs to encode every target, each predicted exactly once."""
    tile_losses = [
        F.cross_entropy(forward_logits(model, x).flatten(0, 1), y.flatten(), reduction="sum").double()
        for x, y in iterate_tiles(inputs, targets, model.config.block_size)
    ]
    return float(torch.stack(tile_losses).sum().item()) / math.log(2)


def pick_classes(logits: torch.Tensor, temperature: float | None) -> torch.Tensor:
    """Most likely class per row if temperature is None, otherwise a sample."""
    if temperature is None:
        return logits.argmax(dim=-1)
    return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(1)


@torch.no_grad()
def choose_runs(model: Transformer, skeleton_ids: torch.Tensor, temperature: float | None) -> torch.Tensor:
    """Pick a vowel run for every skeleton slot."""
    tiles = iterate_tiles(skeleton_ids, skeleton_ids, model.config.block_size)
    return torch.cat([pick_classes(forward_logits(model, x).flatten(0, 1), temperature) for x, _ in tiles])


def decode_abjad(corpus: Corpus, skeleton_ids: torch.Tensor, run_ids: torch.Tensor) -> str:
    skeleton = "".join(corpus.skeleton_vocab[index] for index in skeleton_ids.tolist())
    runs = [corpus.run_vocab[index] for index in run_ids.tolist()]
    return join_text(skeleton, runs)


def measure_word_accuracy(original: str, predicted: str) -> float:
    """Share of words rebuilt exactly.

    Whitespace survives in the skeleton, so splitting at every whitespace character keeps
    the two texts aligned even when a vowel-only word such as "a" is missed entirely.
    """
    original_pieces: list[str] = WHITESPACE.split(original)
    predicted_pieces: list[str] = WHITESPACE.split(predicted)
    pieces = zip(original_pieces, predicted_pieces, strict=True)
    word_pairs = [(expected, actual) for expected, actual in pieces if expected or actual]
    if not word_pairs:
        return 1.0
    return sum(expected == actual for expected, actual in word_pairs) / len(word_pairs)


def measure_vocalizer(vocalizer: Transformer, corpus: Corpus) -> VocalizerReport:
    split = corpus.val
    predicted_ids = choose_runs(vocalizer, split.skeleton_ids, temperature=None)
    original = decode_abjad(corpus, split.skeleton_ids, split.run_ids)
    predicted = decode_abjad(corpus, split.skeleton_ids, predicted_ids)
    empty_run_ids = torch.full_like(split.run_ids, corpus.run_vocab.index(""))
    skeleton_text = decode_abjad(corpus, split.skeleton_ids, empty_run_ids)
    skeleton_preview_chars = int(PREVIEW_CHARS * len(split.skeleton_ids) / split.n_chars)
    return {
        "slot_accuracy": float((predicted_ids == split.run_ids).float().mean().item()),
        "word_accuracy": measure_word_accuracy(original, predicted),
        "preview_original": original[:PREVIEW_CHARS],
        "preview_skeleton": skeleton_text[:skeleton_preview_chars],
        "preview_reconstructed": predicted[:PREVIEW_CHARS],
    }


def measure_bits(models: Models, split: Split) -> BitsReport:
    bits = {task: sum_bits(model, *task_streams(task, split)) for task, model in models.items()}
    n_chars = split.n_chars
    return {
        "baseline_bpc": bits["baseline"] / (n_chars - 1),
        "skeleton_bpc": bits["skeleton"] / n_chars,
        "vocalizer_bpc": bits["vocalizer"] / n_chars,
        "abjad_bpc": (bits["skeleton"] + bits["vocalizer"]) / n_chars,
        "skeleton_bits_per_skeleton_token": bits["skeleton"] / (len(split.skeleton_ids) - 1),
    }


@torch.no_grad()
def generate_ids(model: Transformer, start_id: int, n_new: int, temperature: float) -> torch.Tensor:
    ids = torch.tensor([[start_id]], device=next(model.parameters()).device)
    for _ in range(n_new):
        context = ids[:, -model.config.block_size :]
        probabilities = F.softmax(forward_logits(model, context)[:, -1] / temperature, dim=-1)
        ids = torch.cat([ids, torch.multinomial(probabilities, 1)], dim=1)
    return ids[0]


def generate_baseline(models: Models, corpus: Corpus, n_chars: int, temperature: float) -> str:
    start_id = corpus.char_vocab.index("\n") if "\n" in corpus.char_vocab else 0
    ids = generate_ids(models["baseline"], start_id, n_chars, temperature)
    return "".join(corpus.char_vocab[index] for index in ids[1:].tolist())


def generate_abjad(models: Models, corpus: Corpus, n_chars: int, temperature: float) -> str:
    skeleton_fraction = len(corpus.val.skeleton_ids) / corpus.val.n_chars
    start_id = corpus.skeleton_vocab.index(BOS)
    skeleton_ids = generate_ids(models["skeleton"], start_id, round(n_chars * skeleton_fraction), temperature)
    run_ids = choose_runs(models["vocalizer"], skeleton_ids, temperature)
    return decode_abjad(corpus, skeleton_ids, run_ids)


def time_generation(generate: TextGenerator, models: Models, corpus: Corpus, temperature: float) -> float:
    """Characters of final text produced per second (naive generation, no key-value cache)."""
    generate(models, corpus, SPEED_WARMUP_CHARS, temperature)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    text = generate(models, corpus, SPEED_TEST_CHARS, temperature)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return len(text) / (time.perf_counter() - started)


def evaluate(args: argparse.Namespace) -> Report:
    torch.manual_seed(args.seed)
    corpus = load_corpus(args.data, args.device, args.max_chars)
    models = {task: load_model(resolve_checkpoint_dir(task, args), task, args.device) for task in TASKS}
    return {
        "val_chars": corpus.val.n_chars,
        "n_vowel_run_classes": len(corpus.run_vocab),
        "skeleton_fraction": len(corpus.val.skeleton_ids) / corpus.val.n_chars,
        "parameters": {task: model.count_parameters() for task, model in models.items()},
        "bits": measure_bits(models, corpus.val),
        "vocalizer": measure_vocalizer(models["vocalizer"], corpus),
        "chars_per_second": {
            "baseline": time_generation(generate_baseline, models, corpus, args.temperature),
            "abjad": time_generation(generate_abjad, models, corpus, args.temperature),
        },
        "samples": {
            "baseline": generate_baseline(models, corpus, SAMPLE_CHARS, args.temperature),
            "abjad": generate_abjad(models, corpus, SAMPLE_CHARS, args.temperature),
        },
    }


def print_score_table(results: Report) -> None:
    parameters = results["parameters"]
    abjad_parameters = parameters["skeleton"] + parameters["vocalizer"]
    bits = results["bits"]
    speed = results["chars_per_second"]
    print(f"{'system':<22}{'parameters':>12}{'bits/char':>11}{'chars/sec':>11}")
    print(f"{'baseline':<22}{parameters['baseline']:>12,}{bits['baseline_bpc']:>11.4f}{speed['baseline']:>11.0f}")
    print(f"{'abjad (total)':<22}{abjad_parameters:>12,}{bits['abjad_bpc']:>11.4f}{speed['abjad']:>11.0f}")
    print(f"{'  skeleton model':<22}{parameters['skeleton']:>12,}{bits['skeleton_bpc']:>11.4f}")
    print(f"{'  vocalizer':<22}{parameters['vocalizer']:>12,}{bits['vocalizer_bpc']:>11.4f}")


def print_report(results: Report) -> None:
    vocalizer = results["vocalizer"]
    print(f"\nSkeleton length: {results['skeleton_fraction']:.1%} of the original text")
    print(f"Distinct vowel runs: {results['n_vowel_run_classes']}\n")
    print_score_table(results)
    print(
        f"\nVocalizer reading true skeletons: {vocalizer['slot_accuracy']:.2%} of slots, "
        f"{vocalizer['word_accuracy']:.2%} of whole words correct"
    )
    print(f"\n--- original ---\n{vocalizer['preview_original']}")
    print(f"\n--- skeleton ---\n{vocalizer['preview_skeleton']}")
    print(f"\n--- reconstructed ---\n{vocalizer['preview_reconstructed']}")
    print(f"\n--- baseline sample ---\n{results['samples']['baseline']}")
    print(f"\n--- abjad sample ---\n{results['samples']['abjad']}")


if __name__ == "__main__":
    arguments = parse_args()
    report = evaluate(arguments)
    print_report(report)
    results_path = Path(arguments.checkpoint_dir) / "results.json"
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {results_path}")
