"""Load a text file and encode it three ways: full characters, skeleton, and vowel runs."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from abjad import split_text

TRAIN_FRACTION = 0.9
TASKS = ("baseline", "skeleton", "vocalizer")

Vocabularies = tuple[list[str], list[str], list[str]]


@dataclass(frozen=True)
class Split:
    n_chars: int
    text_ids: torch.Tensor  # every character of the original text
    skeleton_ids: torch.Tensor  # BOS + text with vowels removed
    run_ids: torch.Tensor  # run_ids[i] is the vowel run following skeleton_ids[i]


@dataclass(frozen=True)
class Corpus:
    char_vocab: list[str]
    skeleton_vocab: list[str]
    run_vocab: list[str]
    train: Split
    val: Split


def _encode(symbols: Iterable[str], vocab: list[str], device: str) -> torch.Tensor:
    index_of = {symbol: index for index, symbol in enumerate(vocab)}
    return torch.tensor([index_of[symbol] for symbol in symbols], dtype=torch.long, device=device)


def _encode_split(text: str, vocabs: Vocabularies, device: str) -> Split:
    char_vocab, skeleton_vocab, run_vocab = vocabs
    skeleton, runs = split_text(text)
    return Split(
        n_chars=len(text),
        text_ids=_encode(text, char_vocab, device),
        skeleton_ids=_encode(skeleton, skeleton_vocab, device),
        run_ids=_encode(runs, run_vocab, device),
    )


def load_corpus(path: str, device: str, max_chars: int | None = None) -> Corpus:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        message = f"Could not read dataset at {path}: {error}"
        raise RuntimeError(message) from error
    text = text[:max_chars] if max_chars else text

    # Vocabularies come from the whole file so validation never contains an unknown symbol.
    skeleton, runs = split_text(text)
    vocabs = (sorted(set(text)), sorted(set(skeleton)), sorted(set(runs)))

    boundary = int(len(text) * TRAIN_FRACTION)
    return Corpus(
        char_vocab=vocabs[0],
        skeleton_vocab=vocabs[1],
        run_vocab=vocabs[2],
        train=_encode_split(text[:boundary], vocabs, device),
        val=_encode_split(text[boundary:], vocabs, device),
    )


def task_streams(task: str, split: Split) -> tuple[torch.Tensor, torch.Tensor]:
    """Aligned (inputs, targets) streams: targets[i] is what the model predicts at inputs[i]."""
    if task == "baseline":
        return split.text_ids[:-1], split.text_ids[1:]
    if task == "skeleton":
        return split.skeleton_ids[:-1], split.skeleton_ids[1:]
    if task == "vocalizer":
        return split.skeleton_ids, split.run_ids
    message = f"Unknown task: {task}"
    raise ValueError(message)


def task_vocab_sizes(task: str, corpus: Corpus) -> tuple[int, int]:
    """Return (input vocabulary size, number of output classes) for a task."""
    if task == "baseline":
        return len(corpus.char_vocab), len(corpus.char_vocab)
    if task == "skeleton":
        return len(corpus.skeleton_vocab), len(corpus.skeleton_vocab)
    if task == "vocalizer":
        return len(corpus.skeleton_vocab), len(corpus.run_vocab)
    message = f"Unknown task: {task}"
    raise ValueError(message)


def sample_windows(
    inputs: torch.Tensor, targets: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random aligned windows from two equal-length 1D streams."""
    starts = torch.randint(len(inputs) - block_size + 1, (batch_size, 1), device=inputs.device)
    positions = starts + torch.arange(block_size, device=inputs.device)
    return inputs[positions], targets[positions]
