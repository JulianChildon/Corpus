"""Tests for alignment quality-control reports and sampling."""

from collections import Counter

import pytest

from brics_shift.database import (
    get_next_unannotated_alignment,
    initialize_database,
    set_alignment_error_flag,
)
from brics_shift.quality_control import (
    EligibilityConfig,
    format_alignment_review_report,
    get_alignment_review_report,
    sample_alignments_for_quality_check,
)


@pytest.fixture
def review_database():
    connection = initialize_database(":memory:")
    yield connection
    connection.close()


def insert_review_alignment(
    connection,
    *,
    alignment_id: str,
    year: int,
    paragraph_no: int,
    alignment_type: str,
    confidence: float,
    confidence_band: str,
) -> None:
    connection.execute(
        """
        INSERT INTO alignments (
            alignment_id, year, paragraph_no, en_sentence_ids,
            zh_sentence_ids, en_text, zh_text, alignment_type,
            alignment_cost, alignment_confidence,
            alignment_confidence_band, alignment_algorithm_version,
            created_at
        ) VALUES (?, ?, ?, '[]', '[]', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            alignment_id,
            year,
            paragraph_no,
            f"English text for {alignment_id}.",
            f"中文文本{alignment_id}。",
            alignment_type,
            1.0 - confidence,
            confidence,
            confidence_band,
            "brics-shift-align-v0.1",
            "2026-08-30T08:00:00+00:00",
        ),
    )
    connection.commit()


def populate_filter_examples(connection) -> None:
    examples = (
        ("A_2025_P001_01", 2025, 1, "1:1", 0.92, "high"),
        ("A_2025_P002_01", 2025, 2, "1:2", 0.76, "medium"),
        ("A_2025_P003_01", 2025, 3, "2:1", 0.62, "low"),
        ("A_2024_P004_01", 2024, 4, "2:2", 0.55, "low"),
    )
    for values in examples:
        insert_review_alignment(
            connection,
            alignment_id=values[0],
            year=values[1],
            paragraph_no=values[2],
            alignment_type=values[3],
            confidence=values[4],
            confidence_band=values[5],
        )


def test_review_report_exposes_required_fields(review_database) -> None:
    populate_filter_examples(review_database)

    records = get_alignment_review_report(review_database)

    record = records[-1]
    assert record.year == 2025
    assert record.paragraph_no == 3
    assert record.alignment_id == "A_2025_P003_01"
    assert record.alignment_type == "2:1"
    assert record.alignment_confidence == 0.62
    assert record.english_text.startswith("English text")
    assert record.chinese_text.startswith("中文文本")
    report = format_alignment_review_report(records)
    assert "year\tparagraph_no\talignment_id" in report
    assert "A_2025_P003_01" in report


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        ({"confidence_band": "low"}, {"A_2025_P003_01", "A_2024_P004_01"}),
        ({"confidence_band": "medium"}, {"A_2025_P002_01"}),
        ({"alignment_type": "1:2"}, {"A_2025_P002_01"}),
        ({"alignment_type": "2:1"}, {"A_2025_P003_01"}),
        ({"alignment_type": "2:2"}, {"A_2024_P004_01"}),
        ({"paragraph_no": 2}, {"A_2025_P002_01"}),
        ({"year": 2024}, {"A_2024_P004_01"}),
    ],
)
def test_review_filters(review_database, filters, expected_ids) -> None:
    populate_filter_examples(review_database)

    records = get_alignment_review_report(review_database, **filters)

    assert {record.alignment_id for record in records} == expected_ids


def test_eligibility_and_quarantine_are_derived_without_deletion(review_database) -> None:
    populate_filter_examples(review_database)
    config = EligibilityConfig(minimum_confidence=0.70)

    eligible = get_alignment_review_report(
        review_database,
        eligibility_status="eligible",
        eligibility_config=config,
    )
    quarantined = get_alignment_review_report(
        review_database,
        eligibility_status="quarantined",
        eligibility_config=config,
    )

    assert {record.alignment_id for record in eligible} == {
        "A_2025_P001_01",
        "A_2025_P002_01",
    }
    assert {record.alignment_id for record in quarantined} == {
        "A_2025_P003_01",
        "A_2024_P004_01",
    }
    assert review_database.execute("SELECT COUNT(*) FROM alignments").fetchone()[0] == 4


def test_manual_error_flag_quarantines_alignment_and_can_be_cleared(review_database) -> None:
    insert_review_alignment(
        review_database,
        alignment_id="A_FLAGGED",
        year=2025,
        paragraph_no=1,
        alignment_type="1:1",
        confidence=0.95,
        confidence_band="high",
    )

    set_alignment_error_flag(
        review_database,
        "A_FLAGGED",
        possible_alignment_error=True,
    )
    flagged = get_alignment_review_report(review_database)[0]
    assert flagged.possible_alignment_error is True
    assert flagged.eligibility_status == "quarantined"
    assert flagged.eligibility_reasons == ("possible_alignment_error is set",)

    set_alignment_error_flag(
        review_database,
        "A_FLAGGED",
        possible_alignment_error=False,
    )
    assert get_alignment_review_report(review_database)[0].eligibility_status == "eligible"


def test_default_annotation_queue_excludes_quarantined_alignments(review_database) -> None:
    insert_review_alignment(
        review_database,
        alignment_id="A_LOW",
        year=2025,
        paragraph_no=1,
        alignment_type="1:1",
        confidence=0.60,
        confidence_band="low",
    )
    insert_review_alignment(
        review_database,
        alignment_id="A_HIGH",
        year=2025,
        paragraph_no=2,
        alignment_type="1:1",
        confidence=0.90,
        confidence_band="high",
    )

    assert get_next_unannotated_alignment(
        review_database, "researcher"
    )["alignment_id"] == "A_HIGH"

    set_alignment_error_flag(
        review_database,
        "A_HIGH",
        possible_alignment_error=True,
    )
    assert get_next_unannotated_alignment(review_database, "researcher") is None
    assert get_next_unannotated_alignment(
        review_database,
        "researcher",
        minimum_confidence=None,
        include_possible_alignment_errors=True,
    )["alignment_id"] == "A_LOW"


def test_random_sample_is_exactly_reproducible(review_database) -> None:
    for paragraph_no in range(1, 11):
        insert_review_alignment(
            review_database,
            alignment_id=f"A_SAMPLE_{paragraph_no:02d}",
            year=2025,
            paragraph_no=paragraph_no,
            alignment_type="1:1",
            confidence=0.9,
            confidence_band="high",
        )

    first = sample_alignments_for_quality_check(
        review_database, sample_size=4, random_seed=42
    )
    second = sample_alignments_for_quality_check(
        review_database, sample_size=4, random_seed=42
    )

    assert first == second
    assert len({record.alignment_id for record in first}) == 4


def test_stratified_sample_uses_configured_proportions(review_database) -> None:
    available_counts = {"1:1": 7, "1:2": 2, "2:1": 2, "2:2": 2}
    paragraph_no = 1
    for alignment_type, count in available_counts.items():
        for index in range(count):
            insert_review_alignment(
                review_database,
                alignment_id=f"A_{alignment_type.replace(':', '')}_{index}",
                year=2025,
                paragraph_no=paragraph_no,
                alignment_type=alignment_type,
                confidence=0.9,
                confidence_band="high",
            )
            paragraph_no += 1

    sample = sample_alignments_for_quality_check(
        review_database,
        sample_size=10,
        random_seed=2025,
        stratify_by_alignment_type=True,
        alignment_type_proportions={
            "1:1": 0.7,
            "1:2": 0.1,
            "2:1": 0.1,
            "2:2": 0.1,
        },
    )

    assert Counter(record.alignment_type for record in sample) == {
        "1:1": 7,
        "1:2": 1,
        "2:1": 1,
        "2:2": 1,
    }
