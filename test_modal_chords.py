"""Checks on the parts of the Modal app that run without touching Modal."""

import gzip
import json
import unicodedata

from modal_chords import RUN_DIRS, c4_url, parse_c4_records

LINE_SEPARATOR = unicodedata.lookup("LINE SEPARATOR")


def test_c4_documents_survive_unicode_line_separators() -> None:
    documents = ["First paragraph.\nSecond paragraph.", f"A line{LINE_SEPARATOR}separator inside one document."]
    records = [json.dumps({"text": text, "url": "https://example.com/"}, ensure_ascii=False) for text in documents]
    compressed = gzip.compress(("\n".join(records) + "\n").encode("utf-8"))
    assert LINE_SEPARATOR in records[1]
    assert parse_c4_records(compressed) == documents


def test_c4_urls_name_the_validation_files() -> None:
    assert c4_url(3).endswith("/en/c4-validation.00003-of-00008.json.gz")


def test_runs_that_share_data_keep_separate_checkpoints() -> None:
    checkpoint_dirs = [checkpoint_dir for _, checkpoint_dir in RUN_DIRS.values()]
    assert len(set(checkpoint_dirs)) == len(checkpoint_dirs)
