"""Tests for human-annotation descriptive statistics."""

from datetime import datetime, timezone

import pytest

from brics_shift.database import (
    initialize_database,
    save_annotation,
    set_alignment_error_flag,
)
from brics_shift.statistics import (
    build_descriptive_statistics,
    parse_period_configuration,
)


FIXED_TIME = datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def statistics_database():
    connection = initialize_database(":memory:")
    alignments = (
        ("A1", 2020, 1, "1:1", 0.90, "high"),
        ("A2", 2020, 2, "1:2", 0.75, "medium"),
        ("A3", 2021, 3, "2:1", 0.60, "low"),
        ("A4", 2021, 4, "2:2", 0.88, "high"),
        ("A5", 2022, 5, "1:1", 0.91, "high"),
        ("A6", 2022, 6, "1:1", 0.93, "high"),
    )
    connection.executemany(
        """
        INSERT INTO alignments (
            alignment_id, year, paragraph_no, en_sentence_ids,
            zh_sentence_ids, en_text, zh_text, alignment_type,
            alignment_cost, alignment_confidence,
            alignment_confidence_band, alignment_algorithm_version,
            created_at
        ) VALUES (?, ?, ?, '[]', '[]', ?, ?, ?, 0.1, ?, ?, 'test-v0.1', ?)
        """,
        (
            (
                alignment_id,
                year,
                paragraph_no,
                f"English {alignment_id}",
                f"中文 {alignment_id}",
                alignment_type,
                confidence,
                band,
                FIXED_TIME.isoformat(),
            )
            for alignment_id, year, paragraph_no, alignment_type, confidence, band
            in alignments
        ),
    )
    connection.commit()

    annotation_values = (
        ("A1", "N/A", "preserved", "high", False),
        ("A2", "strengthened", "neutralized", "medium", False),
        ("A3", "omitted", "weakened", "low", False),
        ("A4", "uncertain", "uncertain", "low", False),
        ("A6", "added", "strengthened", "medium", True),
    )
    for index, (alignment_id, modality, stance, confidence, error) in enumerate(
        annotation_values, start=1
    ):
        save_annotation(
            connection,
            annotation_id=f"ANN_{index}",
            alignment_id=alignment_id,
            annotator_id="pilot_researcher",
            modality_label=modality,
            stance_label=stance,
            annotator_confidence=confidence,
            notes="Synthetic pilot annotation.",
            possible_alignment_error=error,
            annotation_guideline_version="pilot-draft-v0.1",
            created_at=FIXED_TIME,
        )
    set_alignment_error_flag(
        connection, "A6", possible_alignment_error=True
    )

    yield connection
    connection.close()


def rows_by_category(distribution):
    return {row.category: row for row in distribution}


def crosstab_rows(crosstab):
    return {group: dict(zip(crosstab.columns, counts)) for group, counts in crosstab.rows}


def test_overview_separately_reports_excluded_records(statistics_database) -> None:
    report = build_descriptive_statistics(
        statistics_database,
        annotator_id="pilot_researcher",
        minimum_alignment_confidence=0.70,
    )

    assert report.overview.total_eligible_alignments == 4
    assert report.overview.total_annotated_alignments == 5
    assert report.overview.eligible_annotated_alignments == 3
    assert report.overview.annotation_completion_rate == 75.0
    assert report.overview.possible_alignment_error_count == 1
    assert report.overview.uncertain_annotation_count == 1
    assert report.overview.quarantined_alignment_count == 2
    assert report.overview.annotated_quarantined_count == 2


def test_alignment_and_confidence_distributions_use_eligible_alignments(
    statistics_database,
) -> None:
    report = build_descriptive_statistics(
        statistics_database,
        annotator_id="pilot_researcher",
    )
    alignment = rows_by_category(report.alignment_type_distribution)
    confidence = rows_by_category(report.confidence_band_distribution)

    assert alignment["1:1"].count == 2
    assert alignment["1:1"].percentage == 50.0
    assert alignment["1:2"].count == 1
    assert alignment["2:1"].count == 0
    assert alignment["2:2"].count == 1
    assert confidence["high"].count == 3
    assert confidence["medium"].count == 1
    assert confidence["low"].count == 0


def test_label_statistics_use_only_human_eligible_non_error_annotations(
    statistics_database,
) -> None:
    report = build_descriptive_statistics(
        statistics_database,
        annotator_id="pilot_researcher",
        denominator_mode="all_annotated",
    )
    modality = rows_by_category(report.modality_distribution)
    stance = rows_by_category(report.stance_distribution)

    assert report.modality_denominator == 3
    assert modality["N/A"].count == 1
    assert modality["strengthened"].count == 1
    assert modality["uncertain"].count == 1
    assert modality["omitted"].count == 0  # Low-confidence A3 is quarantined.
    assert modality["added"].count == 0  # Alignment-error A6 is quarantined.
    assert stance["preserved"].count == 1
    assert stance["neutralized"].count == 1
    assert stance["uncertain"].count == 1


def test_applicable_denominator_excludes_na_but_reports_its_raw_count(
    statistics_database,
) -> None:
    report = build_descriptive_statistics(
        statistics_database,
        annotator_id="pilot_researcher",
        denominator_mode="applicable",
    )
    modality = rows_by_category(report.modality_distribution)

    assert report.modality_denominator == 2
    assert modality["N/A"].count == 1
    assert modality["N/A"].percentage is None
    assert modality["strengthened"].percentage == 50.0
    assert modality["uncertain"].percentage == 50.0
    assert "excluded from percentages" in report.modality_denominator_description


def test_required_crosstabs_contain_human_label_counts(statistics_database) -> None:
    report = build_descriptive_statistics(
        statistics_database,
        annotator_id="pilot_researcher",
    )
    year_modality = crosstab_rows(report.year_modality)
    type_stance = crosstab_rows(report.alignment_type_stance)

    assert year_modality["2020"]["N/A"] == 1
    assert year_modality["2020"]["strengthened"] == 1
    assert year_modality["2021"]["uncertain"] == 1
    assert type_stance["1:1"]["preserved"] == 1
    assert type_stance["1:2"]["neutralized"] == 1
    assert type_stance["2:2"]["uncertain"] == 1


def test_configurable_historical_period_grouping(statistics_database) -> None:
    periods = parse_period_configuration("first,2020,2020\nsecond,2021,2022")
    report = build_descriptive_statistics(
        statistics_database,
        annotator_id="pilot_researcher",
        historical_periods=periods,
    )

    period_modality = crosstab_rows(report.period_modality)
    assert period_modality["first"]["N/A"] == 1
    assert period_modality["first"]["strengthened"] == 1
    assert period_modality["second"]["uncertain"] == 1


def test_overlapping_historical_periods_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        parse_period_configuration("one,2010,2020\ntwo,2020,2025")
