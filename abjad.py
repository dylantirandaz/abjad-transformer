"""Abjad codec: split text into a consonant skeleton plus the vowel runs removed from it.

The split is lossless. Every skeleton character owns one slot holding the (possibly
empty) run of vowels that follows it, so text <-> (skeleton, runs) is a bijection and
P(text) = P(skeleton) * P(runs | skeleton).
"""

import re

VOWELS = "aeiouAEIOU"
# Start-of-text marker. It gives a leading vowel run a slot to live in.
BOS = "\x02"

_SLOT_PATTERN = re.compile(f"([^{VOWELS}])([{VOWELS}]*)", re.DOTALL)


def split_text(text: str) -> tuple[str, list[str]]:
    """Return (skeleton, runs) with skeleton[0] == BOS and len(runs) == len(skeleton)."""
    if BOS in text:
        message = "Text already contains the BOS marker character"
        raise ValueError(message)
    slots: list[tuple[str, str]] = _SLOT_PATTERN.findall(BOS + text)
    skeleton = "".join(consonant for consonant, _ in slots)
    runs = [run for _, run in slots]
    return skeleton, runs


def join_text(skeleton: str, runs: list[str]) -> str:
    """Invert split_text."""
    if len(skeleton) != len(runs):
        message = f"Skeleton has {len(skeleton)} slots but got {len(runs)} runs"
        raise ValueError(message)
    if not skeleton.startswith(BOS):
        message = "Skeleton must start with the BOS marker"
        raise ValueError(message)
    merged = "".join(consonant + run for consonant, run in zip(skeleton, runs, strict=True))
    return merged[len(BOS) :]
