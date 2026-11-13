"""Byte-level BPE in the style of GPT-2, built on the `tokenizers` library."""

import itertools
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

END_OF_TEXT = "<|endoftext|>"
LINES_PER_BATCH = 20_000

TokenIds = npt.NDArray[np.int64]


def train_bpe(lines: Iterable[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[END_OF_TEXT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(lines, trainer)
    return tokenizer


def read_lines(path: str, max_chars: int | None = None) -> Iterator[str]:
    """Yield lines with their line endings kept, stopping once max_chars characters have been read."""
    try:
        with Path(path).open(encoding="utf-8", newline="") as file:
            remaining = max_chars
            for line in file:
                if remaining is not None and remaining <= 0:
                    return
                yield line
                remaining = None if remaining is None else remaining - len(line)
    except OSError as error:
        message = f"Could not read text file at {path}: {error}"
        raise RuntimeError(message) from error


def _encode_batch(tokenizer: Tokenizer, batch: Sequence[str]) -> TokenIds:
    encodings = tokenizer.encode_batch(list(batch))
    return np.concatenate([np.asarray(encoding.ids, dtype=np.int64) for encoding in encodings])


def encode_lines(tokenizer: Tokenizer, lines: Iterable[str]) -> tuple[TokenIds, int]:
    """Return (token ids, characters encoded). Lines are encoded in batches to bound memory."""
    encoded = [
        (_encode_batch(tokenizer, batch), sum(len(line) for line in batch))
        for batch in itertools.batched(lines, LINES_PER_BATCH)
    ]
    if not encoded:
        return np.empty(0, dtype=np.int64), 0
    return np.concatenate([token_ids for token_ids, _ in encoded]), sum(n_chars for _, n_chars in encoded)


def decode_tokens(tokenizer: Tokenizer, token_ids: TokenIds) -> str:
    decoded: str = tokenizer.decode(token_ids.tolist(), skip_special_tokens=False)
    return decoded
