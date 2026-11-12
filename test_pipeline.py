"""Checks on the properties the experiment's numbers depend on."""

import math
from pathlib import Path

import pytest
import torch

from abjad import BOS
from data import TASKS, Corpus, load_corpus, task_streams
from evaluate import decode_abjad, iterate_tiles, measure_word_accuracy, parse_args, resolve_checkpoint_dir, sum_bits
from model import ModelConfig, Transformer
from train import should_evaluate

SAMPLE_TEXT = "Once upon a time, a little girl ate an apple. I owe you one!\n" * 40
BLOCK_SIZE = 16
VOCAB_SIZE = 11
N_CLASSES = 7


def build_model(*, causal: bool) -> Transformer:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=VOCAB_SIZE,
        n_out=N_CLASSES,
        block_size=BLOCK_SIZE,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        causal=causal,
    )
    return Transformer(config).eval()


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    path = tmp_path / "input.txt"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")
    return load_corpus(str(path), "cpu")


def test_streams_are_aligned_for_every_task(corpus: Corpus) -> None:
    for task in TASKS:
        inputs, targets = task_streams(task, corpus.val)
        assert len(inputs) == len(targets), task


def test_skeleton_and_runs_rebuild_the_validation_text(corpus: Corpus) -> None:
    split = corpus.val
    boundary = len(SAMPLE_TEXT) - split.n_chars
    assert corpus.skeleton_vocab[int(split.skeleton_ids[0])] == BOS
    assert decode_abjad(corpus, split.skeleton_ids, split.run_ids) == SAMPLE_TEXT[boundary:]


def test_tiles_cover_the_stream_exactly_once() -> None:
    stream = torch.arange(1000)
    tiles = list(iterate_tiles(stream, stream, BLOCK_SIZE))
    covered = torch.cat([inputs.flatten() for inputs, _ in tiles])
    assert torch.equal(covered, stream)
    assert all(torch.equal(inputs, targets) for inputs, targets in tiles)


def test_causal_model_cannot_see_future_tokens() -> None:
    model = build_model(causal=True)
    ids = torch.randint(VOCAB_SIZE, (1, BLOCK_SIZE))
    changed = ids.clone()
    changed[0, -1] = (ids[0, -1] + 1) % VOCAB_SIZE
    with torch.no_grad():
        assert torch.allclose(model(ids)[:, :-1], model(changed)[:, :-1], atol=1e-6)


def test_bidirectional_model_sees_future_tokens() -> None:
    model = build_model(causal=False)
    ids = torch.randint(VOCAB_SIZE, (1, BLOCK_SIZE))
    changed = ids.clone()
    changed[0, -1] = (ids[0, -1] + 1) % VOCAB_SIZE
    with torch.no_grad():
        assert not torch.allclose(model(ids)[:, :-1], model(changed)[:, :-1], atol=1e-6)


def test_word_accuracy_stays_aligned_when_a_vowel_only_word_is_missed() -> None:
    assert measure_word_accuracy("I saw a cat", "I saw  cat") == pytest.approx(0.75)
    assert measure_word_accuracy("I saw a cat", "I saw a cat") == pytest.approx(1.0)
    assert measure_word_accuracy("the  cat", "the a cat") == pytest.approx(2 / 3)


def test_only_the_baseline_moves_when_a_baseline_dir_is_given() -> None:
    args = parse_args(["--checkpoint_dir", "main", "--baseline_dir", "control"])
    assert resolve_checkpoint_dir("baseline", args) == "control"
    assert resolve_checkpoint_dir("skeleton", args) == "main"
    assert resolve_checkpoint_dir("vocalizer", args) == "main"
    assert resolve_checkpoint_dir("baseline", parse_args(["--checkpoint_dir", "main"])) == "main"


def test_final_step_is_always_evaluated() -> None:
    assert should_evaluate(step=200, eval_interval=500, max_steps=200)
    assert should_evaluate(step=0, eval_interval=500, max_steps=200)
    assert should_evaluate(step=1000, eval_interval=500, max_steps=5000)
    assert not should_evaluate(step=199, eval_interval=500, max_steps=200)


def test_bit_count_matches_a_uniform_model_exactly() -> None:
    model = build_model(causal=True)
    torch.nn.init.zeros_(model.head.weight)
    n_targets = 5 * BLOCK_SIZE + 3
    inputs = torch.randint(VOCAB_SIZE, (n_targets,))
    targets = torch.randint(N_CLASSES, (n_targets,))
    assert sum_bits(model, inputs, targets) == pytest.approx(n_targets * math.log2(N_CLASSES))
