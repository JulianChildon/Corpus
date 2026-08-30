"""Descriptive statistics derived from human annotations only.

This module reports empirical distributions. It performs no significance
tests and makes no linguistic or causal interpretations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import sqlite3
from typing import Literal, Sequence

from brics_shift.annotation import MODALITY_LABELS, STANCE_LABELS
from brics_shift.database import list_alignments_for_annotation


DenominatorMode = Literal["all_annotated", "applicable"]

ALIGNMENT_TYPES = ("1:1", "1:2", "2:1", "2:2")
CONFIDENCE_BANDS = ("high", "medium", "low")


@dataclass(frozen=True, slots=True)
class HistoricalPeriod:
    """A researcher-configured, inclusive year range."""

    name: str
    start_year: int
    end_year: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("period name must not be empty")
        if self.start_year < 1 or self.end_year < self.start_year:
            raise ValueError("period years must be positive and start <= end")


@dataclass(frozen=True, slots=True)
class OverviewStatistics:
    total_eligible_alignments: int
    total_annotated_alignments: int
    eligible_annotated_alignments: int
    annotation_completion_rate: float
    possible_alignment_error_count: int
    uncertain_annotation_count: int
    quarantined_alignment_count: int
    annotated_quarantined_count: int


@dataclass(frozen=True, slots=True)
class DistributionRow:
    category: str
    count: int
    percentage: float | None


@dataclass(frozen=True, slots=True)
class Crosstab:
    row_dimension: str
    column_dimension: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class DescriptiveStatisticsReport:
    annotator_id: str
    minimum_alignment_confidence: float
    denominator_mode: DenominatorMode
    overview: OverviewStatistics
    alignment_type_distribution: tuple[DistributionRow, ...]
    confidence_band_distribution: tuple[DistributionRow, ...]
    modality_distribution: tuple[DistributionRow, ...]
    stance_distribution: tuple[DistributionRow, ...]
    modality_denominator: int
    stance_denominator: int
    modality_denominator_description: str
    stance_denominator_description: str
    year_modality: Crosstab
    year_stance: Crosstab
    declaration_modality: Crosstab
    declaration_stance: Crosstab
    alignment_type_modality: Crosstab
    alignment_type_stance: Crosstab
    period_modality: Crosstab | None
    period_stance: Crosstab | None


def parse_period_configuration(text: str) -> tuple[HistoricalPeriod, ...]:
    """Parse ``name,start_year,end_year`` lines supplied by the researcher."""
    periods: list[HistoricalPeriod] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Period line {line_number} must use name,start_year,end_year"
            )
        name, start_text, end_text = parts
        try:
            period = HistoricalPeriod(name, int(start_text), int(end_text))
        except ValueError as error:
            raise ValueError(f"Invalid period line {line_number}: {error}") from error
        periods.append(period)

    names = [period.name for period in periods]
    if len(names) != len(set(names)):
        raise ValueError("historical period names must be unique")
    ordered = sorted(periods, key=lambda period: (period.start_year, period.end_year))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start_year <= previous.end_year:
            raise ValueError(
                f"historical periods overlap: {previous.name} and {current.name}"
            )
    return tuple(periods)


def _percentage(count: int, denominator: int) -> float | None:
    return None if denominator == 0 else round((count / denominator) * 100, 2)


def _distribution(
    values: Sequence[str],
    categories: Sequence[str],
    *,
    denominator: int | None = None,
    excluded_categories: frozenset[str] = frozenset(),
) -> tuple[DistributionRow, ...]:
    counts = Counter(values)
    active_denominator = len(values) if denominator is None else denominator
    return tuple(
        DistributionRow(
            category=category,
            count=counts[category],
            percentage=(
                None
                if category in excluded_categories
                else _percentage(counts[category], active_denominator)
            ),
        )
        for category in categories
    )


def _crosstab(
    records: Sequence[dict],
    *,
    row_dimension: str,
    column_dimension: str,
    columns: Sequence[str],
    row_value,
    column_field: str,
) -> Crosstab:
    counts: dict[str, Counter] = defaultdict(Counter)
    row_order: list[str] = []
    for record in records:
        group = str(row_value(record))
        if group not in counts:
            row_order.append(group)
        counts[group][record[column_field]] += 1
    return Crosstab(
        row_dimension=row_dimension,
        column_dimension=column_dimension,
        columns=tuple(columns),
        rows=tuple(
            (group, tuple(counts[group][column] for column in columns))
            for group in row_order
        ),
    )


def _period_name(year: int, periods: Sequence[HistoricalPeriod]) -> str:
    for period in periods:
        if period.start_year <= year <= period.end_year:
            return period.name
    return "Unassigned"


def build_descriptive_statistics(
    connection: sqlite3.Connection,
    *,
    annotator_id: str,
    minimum_alignment_confidence: float = 0.70,
    denominator_mode: DenominatorMode = "all_annotated",
    historical_periods: Sequence[HistoricalPeriod] = (),
) -> DescriptiveStatisticsReport:
    """Build transparent descriptive distributions for one annotator."""
    if not annotator_id.strip():
        raise ValueError("annotator_id must not be empty")
    if not 0 <= minimum_alignment_confidence <= 1:
        raise ValueError("minimum_alignment_confidence must be between 0 and 1")
    if denominator_mode not in {"all_annotated", "applicable"}:
        raise ValueError(f"Unsupported denominator mode: {denominator_mode}")
    # Reuse validation for names/ranges and reject overlapping programmatic input.
    parse_period_configuration(
        "\n".join(
            f"{period.name},{period.start_year},{period.end_year}"
            for period in historical_periods
        )
    )

    records = list_alignments_for_annotation(
        connection,
        annotator_id,
        minimum_confidence=None,
        include_possible_alignment_errors=True,
    )
    annotated_records = [record for record in records if record["annotation_id"]]

    def has_possible_error(record: dict) -> bool:
        return bool(record["possible_alignment_error"]) or bool(
            record["annotation_possible_alignment_error"]
        )

    def is_eligible(record: dict) -> bool:
        return (
            record["alignment_confidence"] >= minimum_alignment_confidence
            and not has_possible_error(record)
        )

    eligible_records = [record for record in records if is_eligible(record)]
    included_annotations = [
        record for record in annotated_records if is_eligible(record)
    ]
    possible_error_count = sum(has_possible_error(record) for record in records)
    quarantined_count = len(records) - len(eligible_records)
    annotated_quarantined_count = sum(
        not is_eligible(record) for record in annotated_records
    )
    uncertain_count = sum(
        record["modality_label"] == "uncertain"
        or record["stance_label"] == "uncertain"
        for record in annotated_records
    )
    completion_rate = _percentage(
        len(included_annotations), len(eligible_records)
    ) or 0.0

    overview = OverviewStatistics(
        total_eligible_alignments=len(eligible_records),
        total_annotated_alignments=len(annotated_records),
        eligible_annotated_alignments=len(included_annotations),
        annotation_completion_rate=completion_rate,
        possible_alignment_error_count=possible_error_count,
        uncertain_annotation_count=uncertain_count,
        quarantined_alignment_count=quarantined_count,
        annotated_quarantined_count=annotated_quarantined_count,
    )

    alignment_types = [record["alignment_type"] for record in eligible_records]
    confidence_bands = [
        record["alignment_confidence_band"] for record in eligible_records
    ]
    modality_values = [record["modality_label"] for record in included_annotations]
    stance_values = [record["stance_label"] for record in included_annotations]

    if denominator_mode == "all_annotated":
        modality_denominator = len(modality_values)
        stance_denominator = len(stance_values)
        excluded = frozenset()
        denominator_description = (
            "Eligible, non-error alignments annotated by the selected annotator; "
            "N/A and uncertain labels are included."
        )
        modality_description = denominator_description
        stance_description = denominator_description
    else:
        modality_denominator = sum(label != "N/A" for label in modality_values)
        stance_denominator = sum(label != "N/A" for label in stance_values)
        excluded = frozenset({"N/A"})
        modality_description = (
            "Eligible, non-error human annotations whose modality label is not "
            "N/A. The N/A raw count is shown but excluded from percentages."
        )
        stance_description = (
            "Eligible, non-error human annotations whose stance label is not "
            "N/A. The N/A raw count is shown but excluded from percentages."
        )

    year_value = lambda record: record["year"]
    declaration_value = lambda record: record["document_title"]
    alignment_type_value = lambda record: record["alignment_type"]

    period_modality = None
    period_stance = None
    if historical_periods:
        period_value = lambda record: _period_name(
            record["year"], historical_periods
        )
        period_modality = _crosstab(
            included_annotations,
            row_dimension="historical_period",
            column_dimension="modality_label",
            columns=MODALITY_LABELS,
            row_value=period_value,
            column_field="modality_label",
        )
        period_stance = _crosstab(
            included_annotations,
            row_dimension="historical_period",
            column_dimension="stance_label",
            columns=STANCE_LABELS,
            row_value=period_value,
            column_field="stance_label",
        )

    return DescriptiveStatisticsReport(
        annotator_id=annotator_id,
        minimum_alignment_confidence=minimum_alignment_confidence,
        denominator_mode=denominator_mode,
        overview=overview,
        alignment_type_distribution=_distribution(
            alignment_types, ALIGNMENT_TYPES
        ),
        confidence_band_distribution=_distribution(
            confidence_bands, CONFIDENCE_BANDS
        ),
        modality_distribution=_distribution(
            modality_values,
            MODALITY_LABELS,
            denominator=modality_denominator,
            excluded_categories=excluded,
        ),
        stance_distribution=_distribution(
            stance_values,
            STANCE_LABELS,
            denominator=stance_denominator,
            excluded_categories=excluded,
        ),
        modality_denominator=modality_denominator,
        stance_denominator=stance_denominator,
        modality_denominator_description=modality_description,
        stance_denominator_description=stance_description,
        year_modality=_crosstab(
            included_annotations,
            row_dimension="year",
            column_dimension="modality_label",
            columns=MODALITY_LABELS,
            row_value=year_value,
            column_field="modality_label",
        ),
        year_stance=_crosstab(
            included_annotations,
            row_dimension="year",
            column_dimension="stance_label",
            columns=STANCE_LABELS,
            row_value=year_value,
            column_field="stance_label",
        ),
        declaration_modality=_crosstab(
            included_annotations,
            row_dimension="declaration",
            column_dimension="modality_label",
            columns=MODALITY_LABELS,
            row_value=declaration_value,
            column_field="modality_label",
        ),
        declaration_stance=_crosstab(
            included_annotations,
            row_dimension="declaration",
            column_dimension="stance_label",
            columns=STANCE_LABELS,
            row_value=declaration_value,
            column_field="stance_label",
        ),
        alignment_type_modality=_crosstab(
            included_annotations,
            row_dimension="alignment_type",
            column_dimension="modality_label",
            columns=MODALITY_LABELS,
            row_value=alignment_type_value,
            column_field="modality_label",
        ),
        alignment_type_stance=_crosstab(
            included_annotations,
            row_dimension="alignment_type",
            column_dimension="stance_label",
            columns=STANCE_LABELS,
            row_value=alignment_type_value,
            column_field="stance_label",
        ),
        period_modality=period_modality,
        period_stance=period_stance,
    )
