"""Chord tokens: fold each run of frequent tokens into the ordinary token that follows it.

A chord is a pair (run, host). The host is one ordinary token and the run is the sequence
of frequent "satellite" tokens written immediately before it. Turning chords back into
tokens is plain concatenation, so the mapping is lossless.

The list of runs is finite. A run that is not on the list is spread over several chords:
each takes the longest known run and promotes the satellite after it to host, so no pass is
spent on a chord without a host.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

MAX_RUN_LENGTH = 6
RUN_COLUMN = 0
HOST_COLUMN = 1
PADDING = -1
# Run keys are base-(n_satellites + 1) numbers with MAX_RUN_LENGTH digits and must fit in an int64.
MAX_KEY = 2**62

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ChordVocab:
    vocab_size: int  # size of the underlying token vocabulary
    satellites: IntArray  # (n_satellites,) token ids folded into runs, most frequent first
    run_keys: IntArray  # (n_runs,) integer key of each run; the run id is the row index
    run_table: IntArray  # (n_runs, MAX_RUN_LENGTH) token ids of each run, padded with PADDING
    run_lengths: IntArray  # (n_runs,)

    @property
    def null_host(self) -> int:
        """Host id meaning "this chord has no host token"."""
        return self.vocab_size

    @property
    def n_runs(self) -> int:
        return len(self.run_keys)

    @property
    def n_hosts(self) -> int:
        return self.vocab_size + 1

    @property
    def key_base(self) -> int:
        return len(self.satellites) + 1


@dataclass(frozen=True)
class _Groups:
    starts: IntArray  # index of the first satellite of each chord's run
    lengths: IntArray  # run length of each chord
    hosts: IntArray  # host token id of each chord
    keys: IntArray  # run key, or PADDING when the run is longer than MAX_RUN_LENGTH


def satellite_ranks(vocab_size: int, satellites: IntArray) -> IntArray:
    """Map each token id to its position in the satellite list, or PADDING for an ordinary token."""
    ranks = np.full(vocab_size, PADDING, dtype=np.int64)
    ranks[satellites] = np.arange(len(satellites))
    return ranks


def _key_of(run_ranks: list[int], base: int) -> int:
    return sum((rank + 1) * base**offset for offset, rank in enumerate(run_ranks))


def _group(token_ids: IntArray, ranks: IntArray, null_host: int, base: int) -> _Groups:
    token_ranks = ranks[token_ids]
    host_positions = np.flatnonzero(token_ranks == PADDING)
    ends_with_host = len(host_positions) > 0 and host_positions[-1] == len(token_ids) - 1
    has_tail = len(token_ids) > 0 and not ends_with_host
    ends = np.append(host_positions, len(token_ids)) if has_tail else host_positions
    hosts = np.append(token_ids[host_positions], null_host) if has_tail else token_ids[host_positions]
    starts = np.concatenate([np.zeros(min(1, len(ends)), dtype=np.int64), ends[:-1] + 1])
    lengths = ends - starts

    keys = np.zeros(len(ends), dtype=np.int64)
    for offset in range(MAX_RUN_LENGTH):
        active = np.flatnonzero(lengths > offset)
        keys[active] += (token_ranks[starts[active] + offset] + 1) * base**offset
    keys[lengths > MAX_RUN_LENGTH] = PADDING
    return _Groups(starts=starts, lengths=lengths, hosts=hosts.astype(np.int64), keys=keys)


def _decode_keys(run_keys: IntArray, satellites: IntArray, base: int) -> tuple[IntArray, IntArray]:
    """Return (run_table, run_lengths) for the given keys."""
    digits = np.stack([(run_keys // base**offset) % base for offset in range(MAX_RUN_LENGTH)], axis=1)
    # The extra last entry is what an unused digit maps to.
    lookup = np.append(satellites, PADDING)
    run_table = lookup[np.where(digits > 0, digits - 1, len(satellites))]
    return run_table.astype(np.int64), (digits > 0).sum(axis=1).astype(np.int64)


def build_chord_vocab(token_ids: IntArray, vocab_size: int, n_satellites: int, n_runs: int) -> ChordVocab:
    """Choose the most frequent tokens as satellites and the most frequent runs as the run list."""
    base = n_satellites + 1
    if base**MAX_RUN_LENGTH >= MAX_KEY:
        message = f"{n_satellites} satellites is too many for runs of length {MAX_RUN_LENGTH}"
        raise ValueError(message)
    if n_runs < base:
        message = f"The run list needs at least {base} entries: the empty run and every single satellite"
        raise ValueError(message)

    counts = np.bincount(token_ids, minlength=vocab_size)
    satellites = np.argsort(-counts, kind="stable")[:n_satellites].astype(np.int64)
    groups = _group(token_ids, satellite_ranks(vocab_size, satellites), vocab_size, base)
    seen_keys, seen_counts = np.unique(groups.keys[groups.keys != PADDING], return_counts=True)
    by_frequency = seen_keys[np.argsort(-seen_counts, kind="stable")]

    # Keys below `base` are the empty run and the single satellites. They are always
    # included so that any run can be covered by known runs.
    required = np.arange(base, dtype=np.int64)
    optional = by_frequency[by_frequency >= base][: n_runs - base]
    run_keys = np.concatenate([required, optional])
    run_table, run_lengths = _decode_keys(run_keys, satellites, base)
    return ChordVocab(
        vocab_size=vocab_size,
        satellites=satellites,
        run_keys=run_keys,
        run_table=run_table,
        run_lengths=run_lengths,
    )


def _cover_run(
    run_tokens: list[int], run_ranks: list[int], run_id_of_key: dict[int, int], base: int
) -> tuple[list[tuple[int, int]], int]:
    """Spread an unknown run over several chords.

    Returns the extra chords as (run id, host token) pairs, and the id of the run that
    stays with the original host. Each extra chord takes the longest known run and
    promotes the satellite after it to host.
    """
    extra_chords: list[tuple[int, int]] = []
    position = 0
    while True:
        remaining = run_ranks[position:]
        final_key = _key_of(remaining, base) if len(remaining) <= MAX_RUN_LENGTH else PADDING
        if final_key in run_id_of_key:
            return extra_chords, run_id_of_key[final_key]
        longest = min(MAX_RUN_LENGTH, len(remaining) - 1)
        # The empty run is always known, so this always finds a length.
        length = next(
            candidate for candidate in range(longest, -1, -1) if _key_of(remaining[:candidate], base) in run_id_of_key
        )
        promoted_host = run_tokens[position + length]
        extra_chords = [*extra_chords, (run_id_of_key[_key_of(remaining[:length], base)], promoted_host)]
        position += length + 1


def _split_unknown_runs(
    token_ids: IntArray, ranks: IntArray, groups: _Groups, run_ids: IntArray, vocab: ChordVocab
) -> IntArray:
    unknown = np.flatnonzero(run_ids == PADDING)
    if len(unknown) == 0:
        return np.stack([run_ids, groups.hosts], axis=1)

    run_id_of_key = {key: run_id for run_id, key in enumerate(vocab.run_keys.tolist())}
    spans = zip(groups.starts[unknown].tolist(), groups.lengths[unknown].tolist(), strict=True)
    run_slices = [token_ids[start : start + length] for start, length in spans]
    covers = [
        _cover_run(run_slice.tolist(), ranks[run_slice].tolist(), run_id_of_key, vocab.key_base)
        for run_slice in run_slices
    ]
    # The extra chords of each cover are inserted just before the chord they came from.
    kept_run_ids = run_ids.copy()
    kept_run_ids[unknown] = [final_run_id for _, final_run_id in covers]
    extra_positions = np.repeat(unknown, [len(extra_chords) for extra_chords, _ in covers])
    extra_run_ids = np.array([run_id for extra_chords, _ in covers for run_id, _ in extra_chords], dtype=np.int64)
    extra_hosts = np.array([host for extra_chords, _ in covers for _, host in extra_chords], dtype=np.int64)
    all_run_ids = np.insert(kept_run_ids, extra_positions, extra_run_ids)
    all_hosts = np.insert(groups.hosts, extra_positions, extra_hosts)
    return np.stack([all_run_ids, all_hosts], axis=1)


def encode_chords(token_ids: IntArray, vocab: ChordVocab) -> IntArray:
    """Return an (n_chords, 2) array whose columns are run id and host id."""
    ranks = satellite_ranks(vocab.vocab_size, vocab.satellites)
    groups = _group(token_ids, ranks, vocab.null_host, vocab.key_base)
    order = np.argsort(vocab.run_keys)
    sorted_keys = vocab.run_keys[order]
    slots = np.minimum(np.searchsorted(sorted_keys, groups.keys), len(sorted_keys) - 1)
    run_ids = np.where(sorted_keys[slots] == groups.keys, order[slots], PADDING)
    return _split_unknown_runs(token_ids, ranks, groups, run_ids, vocab)


def decode_chords(chords: IntArray, vocab: ChordVocab) -> IntArray:
    """Invert encode_chords."""
    run_ids = chords[:, RUN_COLUMN]
    hosts = chords[:, HOST_COLUMN]
    lengths = vocab.run_lengths[run_ids]
    has_host = hosts != vocab.null_host
    sizes = lengths + has_host
    starts = np.cumsum(sizes) - sizes

    tokens = np.empty(int(sizes.sum()), dtype=np.int64)
    for offset in range(MAX_RUN_LENGTH):
        active = np.flatnonzero(lengths > offset)
        tokens[starts[active] + offset] = vocab.run_table[run_ids[active], offset]
    tokens[starts[has_host] + lengths[has_host]] = hosts[has_host]
    return tokens


def save_chord_vocab(vocab: ChordVocab, path: str) -> None:
    np.savez(
        path,
        vocab_size=np.int64(vocab.vocab_size),
        satellites=vocab.satellites,
        run_keys=vocab.run_keys,
        run_table=vocab.run_table,
        run_lengths=vocab.run_lengths,
    )


def load_chord_vocab(path: str) -> ChordVocab:
    try:
        with np.load(path) as stored:
            return ChordVocab(
                vocab_size=int(stored["vocab_size"]),
                satellites=stored["satellites"],
                run_keys=stored["run_keys"],
                run_table=stored["run_table"],
                run_lengths=stored["run_lengths"],
            )
    except OSError as error:
        message = f"Could not read chord vocabulary at {path}: {error}"
        raise RuntimeError(message) from error
