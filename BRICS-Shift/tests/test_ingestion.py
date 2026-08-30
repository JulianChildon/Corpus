"""Tests for preview-first uploaded corpus ingestion."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from brics_shift.database import initialize_database
from brics_shift.ingestion import (
    PREPROCESSING_VERSION,
    persist_prepared_pair,
    prepare_uploaded_pair,
)
from brics_shift.segmentation import SEGMENTATION_VERSION


FIXED_TIME = datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc)
ENGLISH_TEXT = (
    "BRICS Declaration\n\n"
    "1. We met Dr. Silva. We reaffirm the 2030 Agenda.\n\n"
    "2. We welcome progress."
)
CHINESE_TEXT = (
    "金砖国家宣言\n\n"
    "1. 我们会见席尔瓦博士。我们重申《2030年议程》。\n\n"
    "2. 我们欢迎进展。"
)


def prepare_pair(**overrides):
    values = {
        "pair_id": "2025_rio",
        "year": 2025,
        "title": "Rio Declaration",
        "english_bytes": ENGLISH_TEXT.encode("utf-8"),
        "english_filename": "2025_rio_en.txt",
        "chinese_bytes": CHINESE_TEXT.encode("utf-8"),
        "chinese_filename": "2025_rio_zh.txt",
        "prepared_at": FIXED_TIME,
    }
    values.update(overrides)
    return prepare_uploaded_pair(**values)


def test_prepare_upload_cleans_parses_segments_and_aligns_without_writing() -> None:
    prepared = prepare_pair()

    assert prepared.english.document.document_id == "2025_rio_en"
    assert prepared.chinese.document.document_id == "2025_rio_zh"
    assert len(prepared.english.paragraph_result.paragraphs) == 2
    assert len(prepared.chinese.paragraph_result.paragraphs) == 2
    assert [sentence.paragraph_no for sentence in prepared.english.sentences] == [
        1,
        1,
        2,
    ]
    assert prepared.english.sentences[0].text == "We met Dr. Silva."
    assert prepared.structure_comparison.matching_paragraph_numbers == 2
    assert len(prepared.alignment_result.alignments) == 3
    assert all(
        alignment.alignment_type == "1:1"
        for alignment in prepared.alignment_result.alignments
    )


def test_preparation_is_exactly_repeatable_with_fixed_time() -> None:
    assert prepare_pair() == prepare_pair()


def test_persist_writes_complete_traceable_pipeline_atomically() -> None:
    connection = initialize_database(":memory:")
    prepared = prepare_pair()

    summary = persist_prepared_pair(connection, prepared)

    assert summary.documents == 2
    assert summary.paragraphs == 4
    assert summary.sentences == 6
    assert summary.alignments == 3
    assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0] == 4
    assert connection.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 6
    assert connection.execute("SELECT COUNT(*) FROM alignments").fetchone()[0] == 3
    assert connection.execute(
        "SELECT DISTINCT preprocessing_version FROM documents"
    ).fetchone()[0] == PREPROCESSING_VERSION
    assert connection.execute(
        "SELECT DISTINCT segmentation_version FROM sentences"
    ).fetchone()[0] == SEGMENTATION_VERSION
    assert connection.execute(
        "SELECT COUNT(*) FROM alignment_sentences"
    ).fetchone()[0] == 6
    connection.close()


def test_duplicate_pair_id_is_rejected_without_adding_partial_rows() -> None:
    connection = initialize_database(":memory:")
    prepared = prepare_pair()
    persist_prepared_pair(connection, prepared)

    with pytest.raises(ValueError, match="already exists"):
        persist_prepared_pair(connection, prepared)

    assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0] == 4
    assert connection.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 6
    assert connection.execute("SELECT COUNT(*) FROM alignments").fetchone()[0] == 3
    connection.close()


def test_failed_alignment_persistence_rolls_back_documents_and_sentences() -> None:
    connection = initialize_database(":memory:")
    prepared = prepare_pair()
    first_alignment = prepared.alignment_result.alignments[0]
    invalid_alignment = replace(first_alignment, english_text="Tampered text.")
    invalid_result = replace(
        prepared.alignment_result,
        alignments=(
            invalid_alignment,
            *prepared.alignment_result.alignments[1:],
        ),
    )
    invalid_prepared = replace(prepared, alignment_result=invalid_result)

    with pytest.raises(ValueError, match="does not match"):
        persist_prepared_pair(connection, invalid_prepared)

    for table in ("documents", "paragraphs", "sentences", "alignments"):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    connection.close()


def test_invalid_utf8_and_unnumbered_input_are_rejected_before_database_write() -> None:
    with pytest.raises(ValueError, match="not valid UTF-8"):
        prepare_pair(english_bytes=b"\xff\xfe")

    with pytest.raises(ValueError, match="No numbered paragraphs"):
        prepare_pair(english_bytes=b"Declaration without numbered paragraphs.")


def test_utf8_bom_is_not_added_to_heading_text() -> None:
    prepared = prepare_pair(
        english_bytes=b"\xef\xbb\xbf" + ENGLISH_TEXT.encode("utf-8")
    )

    assert prepared.english.paragraph_result.heading_text == "BRICS Declaration"
