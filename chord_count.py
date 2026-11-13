"""Count how many model passes each tokenization needs per character, before training anything.

Usage:
python chord_count.py --text data/input.txt
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
from tokenizers import Tokenizer

from bpe import TokenIds, decode_tokens, encode_lines, read_lines, train_bpe
from chords import HOST_COLUMN, RUN_COLUMN, ChordVocab, build_chord_vocab, decode_chords, encode_chords
from console import configure_console

BPE_VOCAB_SIZES = (1024, 2048, 4096, 8192, 16384)
CHORD_BPE_VOCAB_SIZE = 8192
SATELLITE_COUNTS = (16, 32, 64, 128, 256, 512)
RUN_COUNTS = (1024, 4096, 16384)
HELD_OUT_FRACTION = 0.1
EXAMPLES_SHOWN = 20


@dataclass(frozen=True)
class CountSettings:
    """Which BPE vocabularies to measure, and which chord settings to build on which of them."""

    bpe_vocab_sizes: tuple[int, ...] = BPE_VOCAB_SIZES
    chord_bpe_vocab_size: int = CHORD_BPE_VOCAB_SIZE
    satellite_counts: tuple[int, ...] = SATELLITE_COUNTS
    run_counts: tuple[int, ...] = RUN_COUNTS


DEFAULT_COUNT_SETTINGS = CountSettings()


class BpeRow(TypedDict):
    vocab_size: int
    chars_per_token: float


class ChordRow(TypedDict):
    n_satellites: int
    n_runs: int
    chars_per_chord: float
    chords_per_token: float
    no_host_share: float
    satellite_share: float


class CountReport(TypedDict):
    held_out_chars: int
    chord_bpe_vocab_size: int
    bpe: list[BpeRow]
    chords: list[ChordRow]
    best: ChordRow
    top_satellites: list[str]
    top_runs: list[str]


def measure_bpe(train_lines: list[str], held_lines: list[str], vocab_size: int) -> BpeRow:
    tokenizer = train_bpe(train_lines, vocab_size)
    token_ids, n_chars = encode_lines(tokenizer, held_lines)
    return {"vocab_size": tokenizer.get_vocab_size(), "chars_per_token": n_chars / len(token_ids)}


def measure_chords(vocab: ChordVocab, held_ids: TokenIds, held_chars: int) -> ChordRow:
    chords = encode_chords(held_ids, vocab)
    if not np.array_equal(decode_chords(chords, vocab), held_ids):
        message = f"Chords did not round-trip with {len(vocab.satellites)} satellites and {vocab.n_runs} runs"
        raise RuntimeError(message)
    return {
        "n_satellites": len(vocab.satellites),
        "n_runs": vocab.n_runs,
        "chars_per_chord": held_chars / len(chords),
        "chords_per_token": len(chords) / len(held_ids),
        "no_host_share": float((chords[:, HOST_COLUMN] == vocab.null_host).mean()),
        "satellite_share": float(np.isin(held_ids, vocab.satellites).mean()),
    }


def describe_satellites(tokenizer: Tokenizer, vocab: ChordVocab) -> list[str]:
    """Return the most frequent satellites as readable strings."""
    shown = vocab.satellites[:EXAMPLES_SHOWN]
    return [repr(decode_tokens(tokenizer, shown[index : index + 1])) for index in range(len(shown))]


def describe_runs(tokenizer: Tokenizer, vocab: ChordVocab, held_ids: TokenIds) -> list[str]:
    """Return the most frequent non-empty runs in the held-out text as readable strings."""
    run_counts = np.bincount(encode_chords(held_ids, vocab)[:, RUN_COLUMN], minlength=vocab.n_runs)
    most_frequent = [run_id for run_id in np.argsort(-run_counts).tolist() if vocab.run_lengths[run_id] > 0]
    return [
        repr(decode_tokens(tokenizer, vocab.run_table[run_id, : vocab.run_lengths[run_id]]))
        for run_id in most_frequent[:EXAMPLES_SHOWN]
    ]


def count(text_path: str, settings: CountSettings = DEFAULT_COUNT_SETTINGS) -> CountReport:
    lines = list(read_lines(text_path))
    boundary = int(len(lines) * (1 - HELD_OUT_FRACTION))
    train_lines, held_lines = lines[:boundary], lines[boundary:]

    tokenizer = train_bpe(train_lines, settings.chord_bpe_vocab_size)
    vocab_size = tokenizer.get_vocab_size()
    train_ids, _ = encode_lines(tokenizer, train_lines)
    held_ids, held_chars = encode_lines(tokenizer, held_lines)
    pairs = [(k, r) for k in settings.satellite_counts for r in settings.run_counts if r > k]
    vocabs = [build_chord_vocab(train_ids, vocab_size, k, r) for k, r in pairs]
    rows = [measure_chords(vocab, held_ids, held_chars) for vocab in vocabs]
    best_index = int(np.argmax([row["chars_per_chord"] for row in rows]))
    best_vocab = vocabs[best_index]
    return {
        "held_out_chars": held_chars,
        "chord_bpe_vocab_size": vocab_size,
        "bpe": [measure_bpe(train_lines, held_lines, size) for size in settings.bpe_vocab_sizes],
        "chords": rows,
        "best": rows[best_index],
        "top_satellites": describe_satellites(tokenizer, best_vocab),
        "top_runs": describe_runs(tokenizer, best_vocab, held_ids),
    }


def print_report(report: CountReport) -> None:
    print(f"\nHeld-out text: {report['held_out_chars']:,} characters\n")
    print(f"{'BPE vocabulary':>15}{'chars/pass':>12}")
    for bpe_row in report["bpe"]:
        print(f"{bpe_row['vocab_size']:>15,}{bpe_row['chars_per_token']:>12.2f}")

    # A bigger BPE vocabulary also cuts passes, so chords are compared with the largest one measured too.
    largest = max(report["bpe"], key=lambda bpe_row: bpe_row["vocab_size"])
    largest_label = f"vs {largest['vocab_size']:,} BPE"
    print(f"\nChords on top of the {report['chord_bpe_vocab_size']:,}-token BPE")
    print(
        f"{'satellites':>11}{'runs':>8}{'chars/pass':>12}{'passes vs BPE':>15}"
        f"{largest_label:>18}{'satellite tokens':>18}"
    )
    for row in report["chords"]:
        print(
            f"{row['n_satellites']:>11}{row['n_runs']:>8,}{row['chars_per_chord']:>12.2f}"
            f"{row['chords_per_token']:>15.3f}{largest['chars_per_token'] / row['chars_per_chord']:>18.3f}"
            f"{row['satellite_share']:>18.1%}"
        )

    best = report["best"]
    print(f"\nFewest passes: {best['n_satellites']} satellites, {best['n_runs']:,} runs")
    print(f"Its most frequent satellites: {' '.join(report['top_satellites'])}")
    print(f"Its most frequent runs: {' '.join(report['top_runs'])}")


if __name__ == "__main__":
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default="data/input.txt")
    parser.add_argument("--out", default="count.json")
    arguments = parser.parse_args()
    count_report = count(arguments.text)
    print_report(count_report)
    Path(arguments.out).write_text(json.dumps(count_report, indent=2), encoding="utf-8")
