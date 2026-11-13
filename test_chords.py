"""Checks that BPE and chord tokens are lossless, including the awkward cases."""

from pathlib import Path

import numpy as np

from bpe import END_OF_TEXT, decode_tokens, encode_lines, train_bpe
from chords import (
    HOST_COLUMN,
    MAX_RUN_LENGTH,
    RUN_COLUMN,
    ChordVocab,
    IntArray,
    build_chord_vocab,
    decode_chords,
    encode_chords,
    load_chord_vocab,
    save_chord_vocab,
)

SAMPLE_LINES = [
    "Once upon a time, there was a little girl named Lily.\n",
    "She said, “I don't want to go to the café!” 🙂\n",
    f"{END_OF_TEXT}\n",
    "The dog and the cat went to the park with the girl.\n",
    "  Two leading spaces, a tab\tand no newline at the end",
]
BPE_VOCAB_SIZE = 300
TOKEN_VOCAB_SIZE = 50


def skewed_tokens(length: int, seed: int) -> IntArray:
    """Random tokens where low ids are much more frequent, like real text."""
    weights = 1.0 / np.arange(1, TOKEN_VOCAB_SIZE + 1)
    generator = np.random.default_rng(seed)
    return generator.choice(TOKEN_VOCAB_SIZE, size=length, p=weights / weights.sum()).astype(np.int64)


def run_tokens(vocab: ChordVocab, run_id: int) -> list[int]:
    tokens: list[int] = vocab.run_table[run_id, : vocab.run_lengths[run_id]].tolist()
    return tokens


def test_bpe_round_trip_keeps_every_character() -> None:
    tokenizer = train_bpe(SAMPLE_LINES, BPE_VOCAB_SIZE)
    token_ids, n_chars = encode_lines(tokenizer, SAMPLE_LINES)
    assert decode_tokens(tokenizer, token_ids) == "".join(SAMPLE_LINES)
    assert n_chars == len("".join(SAMPLE_LINES))


def test_end_of_text_is_a_single_token() -> None:
    tokenizer = train_bpe(SAMPLE_LINES, BPE_VOCAB_SIZE)
    token_ids, _ = encode_lines(tokenizer, [END_OF_TEXT])
    assert len(token_ids) == 1


def test_chords_follow_the_definition_on_a_hand_example() -> None:
    # Tokens 0 and 1 are the two most frequent, so they are the satellites.
    tokens = np.array([0, 1, 5, 0, 6, 7, 0, 1, 5, 0, 1, 8, 1], dtype=np.int64)
    vocab = build_chord_vocab(tokens, vocab_size=10, n_satellites=2, n_runs=8)
    chords = encode_chords(tokens, vocab)
    decoded = [(run_tokens(vocab, run_id), host) for run_id, host in chords.tolist()]
    assert decoded == [([0, 1], 5), ([0], 6), ([], 7), ([0, 1], 5), ([0, 1], 8), ([1], vocab.null_host)]


def test_chords_round_trip_when_most_runs_must_be_split() -> None:
    tokens = skewed_tokens(length=5000, seed=0)
    vocab = build_chord_vocab(tokens, TOKEN_VOCAB_SIZE, n_satellites=8, n_runs=12)
    chords = encode_chords(tokens, vocab)
    assert np.array_equal(decode_chords(chords, vocab), tokens)
    assert chords[:, RUN_COLUMN].max() < vocab.n_runs
    # Split runs promote a satellite to host instead of spending a pass on a chord with no host.
    assert np.isin(chords[:, HOST_COLUMN], vocab.satellites).any()
    assert (chords[:, HOST_COLUMN] == vocab.null_host).sum() <= 1


def test_chords_round_trip_on_unseen_tokens() -> None:
    vocab = build_chord_vocab(skewed_tokens(length=5000, seed=1), TOKEN_VOCAB_SIZE, n_satellites=8, n_runs=64)
    unseen = skewed_tokens(length=3000, seed=2)
    assert np.array_equal(decode_chords(encode_chords(unseen, vocab), vocab), unseen)


def test_a_very_long_run_is_split_in_order() -> None:
    pattern = [0, 1, 2, 1, 0, 2, 2, 1]
    tokens = np.array([*pattern * (2 * MAX_RUN_LENGTH), 9], dtype=np.int64)
    vocab = build_chord_vocab(tokens, vocab_size=10, n_satellites=3, n_runs=4)
    chords = encode_chords(tokens, vocab)
    assert np.array_equal(decode_chords(chords, vocab), tokens)
    # Only single satellites are known runs, so each chord carries two satellites: a run and a promoted host.
    assert len(chords) == len(tokens) // 2 + 1


def test_text_ending_on_a_satellite_gets_a_chord_with_no_host() -> None:
    tokens = np.array([0, 0, 5, 0, 0, 6, 0, 0], dtype=np.int64)
    vocab = build_chord_vocab(tokens, vocab_size=10, n_satellites=1, n_runs=4)
    chords = encode_chords(tokens, vocab)
    assert chords[-1, HOST_COLUMN] == vocab.null_host
    assert np.array_equal(decode_chords(chords, vocab), tokens)


def test_zero_satellites_reduces_to_plain_tokens() -> None:
    tokens = skewed_tokens(length=500, seed=3)
    vocab = build_chord_vocab(tokens, TOKEN_VOCAB_SIZE, n_satellites=0, n_runs=1)
    chords = encode_chords(tokens, vocab)
    assert np.array_equal(chords[:, HOST_COLUMN], tokens)
    assert not chords[:, RUN_COLUMN].any()


def test_empty_input_gives_no_chords() -> None:
    vocab = build_chord_vocab(skewed_tokens(length=500, seed=4), TOKEN_VOCAB_SIZE, n_satellites=4, n_runs=16)
    empty = np.empty(0, dtype=np.int64)
    assert encode_chords(empty, vocab).shape == (0, 2)
    assert len(decode_chords(encode_chords(empty, vocab), vocab)) == 0


def test_chord_vocab_survives_saving_and_loading(tmp_path: Path) -> None:
    tokens = skewed_tokens(length=2000, seed=5)
    vocab = build_chord_vocab(tokens, TOKEN_VOCAB_SIZE, n_satellites=6, n_runs=40)
    path = str(tmp_path / "vocab.npz")
    save_chord_vocab(vocab, path)
    loaded = load_chord_vocab(path)
    assert np.array_equal(encode_chords(tokens, loaded), encode_chords(tokens, vocab))
    assert loaded.n_hosts == vocab.n_hosts
