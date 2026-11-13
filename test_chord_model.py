"""Checks on the chord model's properties that the experiment's scores depend on."""

import math

import pytest
import torch

from chord_evaluate import generate_chords, sum_chord_bits
from chord_model import HOST_FIRST, INDEPENDENT, ORDERS, RUN_FIRST, ChordConfig, ChordTransformer
from chord_train import arm_name
from chords import HOST_COLUMN, RUN_COLUMN

BLOCK_SIZE = 16
N_RUNS = 5
N_HOSTS = 40


def build_model(order: str) -> ChordTransformer:
    torch.manual_seed(0)
    config = ChordConfig(
        n_runs=N_RUNS,
        n_hosts=N_HOSTS,
        block_size=BLOCK_SIZE,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        order=order,
    )
    return ChordTransformer(config).eval()


def random_chords(length: int) -> torch.Tensor:
    runs = torch.randint(N_RUNS, (1, length))
    hosts = torch.randint(N_HOSTS, (1, length))
    return torch.stack([runs, hosts], dim=-1)


def with_changed_column(chords: torch.Tensor, column: int, size: int) -> torch.Tensor:
    changed = chords.clone()
    changed[..., column] = (chords[..., column] + 1) % size
    return changed


@pytest.mark.parametrize("order", ORDERS)
def test_model_cannot_see_future_chords(order: str) -> None:
    model = build_model(order)
    chords, next_chords = random_chords(BLOCK_SIZE), random_chords(BLOCK_SIZE)
    changed = chords.clone()
    changed[0, -1] = (chords[0, -1] + 1) % N_RUNS
    with torch.no_grad():
        before = model(chords, next_chords)
        after = model(changed, next_chords)
    for logits_before, logits_after in zip(before, after, strict=True):
        assert torch.allclose(logits_before[:, :-1], logits_after[:, :-1], atol=1e-6)


def test_run_first_links_only_the_host_pick_to_the_run() -> None:
    model = build_model(RUN_FIRST)
    chords, next_chords = random_chords(BLOCK_SIZE), random_chords(BLOCK_SIZE)
    with torch.no_grad():
        run_logits, host_logits = model(chords, next_chords)
        run_after_run, host_after_run = model(chords, with_changed_column(next_chords, RUN_COLUMN, N_RUNS))
        run_after_host, host_after_host = model(chords, with_changed_column(next_chords, HOST_COLUMN, N_HOSTS))
    assert torch.equal(run_logits, run_after_run)
    assert not torch.allclose(host_logits, host_after_run)
    assert torch.equal(run_logits, run_after_host)
    assert torch.equal(host_logits, host_after_host)


def test_host_first_links_only_the_run_pick_to_the_host() -> None:
    model = build_model(HOST_FIRST)
    chords, next_chords = random_chords(BLOCK_SIZE), random_chords(BLOCK_SIZE)
    with torch.no_grad():
        run_logits, host_logits = model(chords, next_chords)
        run_after_host, host_after_host = model(chords, with_changed_column(next_chords, HOST_COLUMN, N_HOSTS))
        run_after_run, host_after_run = model(chords, with_changed_column(next_chords, RUN_COLUMN, N_RUNS))
    assert torch.equal(host_logits, host_after_host)
    assert not torch.allclose(run_logits, run_after_host)
    assert torch.equal(run_logits, run_after_run)
    assert torch.equal(host_logits, host_after_run)


def test_independent_picks_ignore_the_next_chord() -> None:
    model = build_model(INDEPENDENT)
    chords, next_chords = random_chords(BLOCK_SIZE), random_chords(BLOCK_SIZE)
    with torch.no_grad():
        before = model(chords, next_chords)
        after = model(chords, random_chords(BLOCK_SIZE))
    assert torch.equal(before[RUN_COLUMN], after[RUN_COLUMN])
    assert torch.equal(before[HOST_COLUMN], after[HOST_COLUMN])


def test_chord_bit_count_matches_a_uniform_model_exactly() -> None:
    model = build_model(RUN_FIRST)
    torch.nn.init.zeros_(model.run_head.weight)
    torch.nn.init.zeros_(model.host_head.weight)
    n_chords = 5 * BLOCK_SIZE + 3
    chords, next_chords = random_chords(n_chords)[0], random_chords(n_chords)[0]
    run_bits, host_bits = sum_chord_bits(model, chords, next_chords)
    assert run_bits == pytest.approx(n_chords * math.log2(N_RUNS))
    assert host_bits == pytest.approx(n_chords * math.log2(N_HOSTS))


@pytest.mark.parametrize("order", ORDERS)
def test_generated_chords_keep_run_and_host_in_their_columns(order: str) -> None:
    model = build_model(order)
    n_new = 3 * BLOCK_SIZE
    generated = generate_chords(model, random_chords(1), n_new, temperature=1.0)
    assert generated.shape == (n_new + 1, 2)
    # There are far fewer runs than hosts, so a swapped column would show up as an out-of-range run.
    assert generated[:, RUN_COLUMN].max() < N_RUNS
    assert generated[:, HOST_COLUMN].max() < N_HOSTS


def test_arm_names() -> None:
    assert arm_name(None, RUN_FIRST) == "bpe"
    assert arm_name("k32_r1024", HOST_FIRST) == "k32_r1024_host_first"
