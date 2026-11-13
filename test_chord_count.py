"""Checks that the pass count measures the settings it is given and reports them consistently."""

from pathlib import Path

import pytest

from chord_count import CountReport, CountSettings, count, print_report

SENTENCES = (
    "Once upon a time, there was a little girl named Lily.",
    "She liked to play in the park with her dog and her friends.",
    "One day, the dog ran to the big tree near the old house.",
    'Lily said, "Come back!" and the dog came back to her.',
    "They went home and had a good dinner with mom and dad.",
)
SAMPLE_TEXT = "\n".join(SENTENCES * 40) + "\n"
SMALL_SETTINGS = CountSettings(
    bpe_vocab_sizes=(280, 320),
    chord_bpe_vocab_size=320,
    satellite_counts=(4, 16),
    run_counts=(32, 128),
)


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> CountReport:
    path = tmp_path_factory.mktemp("count") / "text.txt"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")
    return count(str(path), SMALL_SETTINGS)


def test_count_measures_every_setting_it_is_given(report: CountReport) -> None:
    assert len(report["bpe"]) == len(SMALL_SETTINGS.bpe_vocab_sizes)
    assert [row["n_satellites"] for row in report["chords"]] == [4, 4, 16, 16]


def test_chords_need_fewer_passes_than_their_tokens(report: CountReport) -> None:
    assert all(row["chords_per_token"] < 1 for row in report["chords"])
    assert report["best"]["chars_per_chord"] == max(row["chars_per_chord"] for row in report["chords"])


def test_report_compares_chords_with_the_largest_bpe(report: CountReport, capsys: pytest.CaptureFixture[str]) -> None:
    print_report(report)
    largest = max(row["vocab_size"] for row in report["bpe"])
    assert f"vs {largest:,} BPE" in capsys.readouterr().out


def test_text_path_is_read_from_disk(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(RuntimeError, match="Could not read text file"):
        count(str(missing), SMALL_SETTINGS)
