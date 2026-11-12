"""Round-trip checks for the abjad codec."""

from abjad import BOS, join_text, split_text

ROUND_TRIP_CASES = [
    "",
    "a",
    "I",
    "rhythm",
    "aeiou",
    "The quick brown fox.",
    "Eerie queue; AEON\n\nOnce upon a time, I ate an apple.",
    "  leading spaces and trailing vowel o",
]


def test_round_trip() -> None:
    for text in ROUND_TRIP_CASES:
        skeleton, runs = split_text(text)
        assert join_text(skeleton, runs) == text, repr(text)
        assert len(skeleton) == len(runs), repr(text)


def test_skeleton_has_no_vowels() -> None:
    skeleton, runs = split_text("The quick brown fox.")
    assert skeleton == BOS + "Th qck brwn fx."
    assert runs[2] == "e"
    assert runs[4] == "ui"


def test_leading_vowel_run_lives_on_bos() -> None:
    skeleton, runs = split_text("eat")
    assert skeleton == BOS + "t"
    assert runs == ["ea", ""]


def test_vowel_only_word_leaves_adjacent_spaces() -> None:
    skeleton, _ = split_text("was a girl")
    assert skeleton == BOS + "ws  grl"
