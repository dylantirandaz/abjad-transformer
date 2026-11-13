"""Tokenize the text once and save the BPE stream and the chord stream for each setting.

Usage:
python chord_prepare.py --train_text data/train.txt --val_text data/input.txt --settings 32:1024 64:4096
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from tokenizers import Tokenizer

from bpe import TokenIds, encode_lines, read_lines, train_bpe
from chords import build_chord_vocab, decode_chords, encode_chords, save_chord_vocab

BPE_STREAM = "bpe"
TOKENIZER_FILE = "tokenizer.json"
META_FILE = "meta.json"


class ChordSettingMeta(TypedDict):
    n_satellites: int
    n_runs: int
    n_hosts: int
    train_chords: int
    val_chords: int


class PreparedMeta(TypedDict):
    n_train_chars: int
    n_val_chars: int
    bpe_vocab_size: int
    train_tokens: int
    val_tokens: int
    settings: dict[str, ChordSettingMeta]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_text", required=True)
    parser.add_argument("--val_text", required=True)
    parser.add_argument("--out_dir", default="prepared")
    parser.add_argument("--bpe_vocab_size", type=int, default=8192)
    parser.add_argument("--train_chars", type=int, default=None, help="use only this much of the training text")
    parser.add_argument("--tokenizer_chars", type=int, default=50_000_000, help="text used to train the BPE")
    parser.add_argument("--settings", nargs="+", default=["32:1024"], help="satellites:runs pairs")
    return parser.parse_args(argv)


def setting_name(n_satellites: int, n_runs: int) -> str:
    return f"k{n_satellites}_r{n_runs}"


def parse_setting(text: str) -> tuple[int, int]:
    try:
        n_satellites, n_runs = (int(part) for part in text.split(":"))
    except ValueError as error:
        message = f"Expected a setting like 32:1024 (satellites:runs), got {text!r}"
        raise ValueError(message) from error
    return n_satellites, n_runs


def stream_path(data_dir: str, stream: str, split: str) -> Path:
    return Path(data_dir) / f"{stream}_{split}.npy"


def vocab_path(data_dir: str, setting: str) -> Path:
    return Path(data_dir) / f"{setting}_vocab.npz"


def load_meta(data_dir: str) -> PreparedMeta:
    path = Path(data_dir) / META_FILE
    try:
        meta: PreparedMeta = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        message = f"Missing {path}. Run chord_prepare.py first."
        raise RuntimeError(message) from error
    return meta


def load_tokenizer(data_dir: str) -> Tokenizer:
    return Tokenizer.from_file(str(Path(data_dir) / TOKENIZER_FILE))


def prepare_setting(
    out_dir: str, setting: tuple[int, int], train_ids: TokenIds, val_ids: TokenIds, vocab_size: int
) -> ChordSettingMeta:
    n_satellites, n_runs = setting
    name = setting_name(n_satellites, n_runs)
    vocab = build_chord_vocab(train_ids, vocab_size, n_satellites, n_runs)
    train_chords = encode_chords(train_ids, vocab)
    val_chords = encode_chords(val_ids, vocab)
    if not np.array_equal(decode_chords(val_chords, vocab), val_ids):
        message = f"Chords did not round-trip for setting {name}"
        raise RuntimeError(message)
    save_chord_vocab(vocab, str(vocab_path(out_dir, name)))
    np.save(stream_path(out_dir, name, "train"), train_chords)
    np.save(stream_path(out_dir, name, "val"), val_chords)
    print(f"[{name}] {len(train_chords):,} training chords, {len(val_chords):,} validation chords")
    return {
        "n_satellites": n_satellites,
        "n_runs": vocab.n_runs,
        "n_hosts": vocab.n_hosts,
        "train_chords": len(train_chords),
        "val_chords": len(val_chords),
    }


def prepare(args: argparse.Namespace) -> PreparedMeta:
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = train_bpe(read_lines(args.train_text, args.tokenizer_chars), args.bpe_vocab_size)
    tokenizer.save(str(Path(args.out_dir) / TOKENIZER_FILE))
    vocab_size = tokenizer.get_vocab_size()

    train_ids, n_train_chars = encode_lines(tokenizer, read_lines(args.train_text, args.train_chars))
    val_ids, n_val_chars = encode_lines(tokenizer, read_lines(args.val_text))
    np.save(stream_path(args.out_dir, BPE_STREAM, "train"), train_ids)
    np.save(stream_path(args.out_dir, BPE_STREAM, "val"), val_ids)
    print(f"[{BPE_STREAM}] {len(train_ids):,} training tokens, {len(val_ids):,} validation tokens")

    settings = [parse_setting(text) for text in args.settings]
    meta: PreparedMeta = {
        "n_train_chars": n_train_chars,
        "n_val_chars": n_val_chars,
        "bpe_vocab_size": vocab_size,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "settings": {
            setting_name(*setting): prepare_setting(args.out_dir, setting, train_ids, val_ids, vocab_size)
            for setting in settings
        },
    }
    (Path(args.out_dir) / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    prepare(parse_args())
