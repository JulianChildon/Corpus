"""Transparent alignment quality-control reports and reproducible sampling.

Quality-control status is derived from configuration. Alignments are never
deleted or silently excluded from storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import sqlite3
from typing import Literal, Mapping

from brics_shift.alignment import AlignmentType, ConfidenceBand


EligibilityStatus = Literal["eligible", "quarantined"]

_ALIGNMENT_TYPES: tuple[AlignmentType, ...] = ("1:1", "1:2", "2:1", "2:2")
_CONFIDENCE_BANDS: tuple[ConfidenceBand, ...] = ("high", "medium", "low")


@dataclass(frozen=True, slots=True)
class EligibilityConfig:
    """Transparent rules for admission to the default annotation queue."""

    minimum_confidence: float = 0.70
    quarantine_possible_alignment_errors: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class AlignmentReviewRecord:
    """One alignment row prepared for human quality inspection."""

    year: int
    paragraph_no: int
    alignment_id: str
    alignment_type: AlignmentType
    alignment_confidence: float
    alignment_confidence_band: ConfidenceBand
    english_text: str
    chinese_text: str
    possible_alignment_error: bool
    eligibility_status: EligibilityStatus
    eligibility_reasons: tuple[str, ...]


def derive_eligibility(
    *,
    alignment_confidence: float,
    possible_alignment_error: bool,
    config: EligibilityConfig | None = None,
) -> tuple[EligibilityStatus, tuple[str, ...]]:
    """Derive eligibility and reasons without mutating the alignment."""
    active_config = config or EligibilityConfig()
    if not 0 <= alignment_confidence <= 1:
        raise ValueError("alignment_confidence must be between 0 and 1")

    reasons: list[str] = []
    if alignment_confidence < active_config.minimum_confidence:
        reasons.append(
            f"confidence {alignment_confidence:.6f} is below threshold "
            f"{active_config.minimum_confidence:.6f}"
        )
    if (
        active_config.quarantine_possible_alignment_errors
        and possible_alignment_error
    ):
        reasons.append("possible_alignment_error is set")

    status: EligibilityStatus = "quarantined" if reasons else "eligible"
    return status, tuple(reasons)


def get_alignment_review_report(
    connection: sqlite3.Connection,
    *,
    confidence_band: ConfidenceBand | None = None,
    alignment_type: AlignmentType | None = None,
    paragraph_no: int | None = None,
    year: int | None = None,
    eligibility_status: EligibilityStatus | None = None,
    eligibility_config: EligibilityConfig | None = None,
) -> tuple[AlignmentReviewRecord, ...]:
    """Return inspectable alignment rows matching explicit quality filters.

    Use ``confidence_band='low'`` or ``'medium'`` for the corresponding
    confidence-only filters, and ``alignment_type`` for 1:2, 2:1, or 2:2.
    """
    if confidence_band is not None and confidence_band not in _CONFIDENCE_BANDS:
        raise ValueError(f"Unsupported confidence band: {confidence_band}")
    if alignment_type is not None and alignment_type not in _ALIGNMENT_TYPES:
        raise ValueError(f"Unsupported alignment type: {alignment_type}")
    if paragraph_no is not None and paragraph_no < 1:
        raise ValueError("paragraph_no must be positive")
    if year is not None and year < 1:
        raise ValueError("year must be positive")
    if eligibility_status not in {None, "eligible", "quarantined"}:
        raise ValueError(f"Unsupported eligibility status: {eligibility_status}")

    sql = """
        SELECT year, paragraph_no, alignment_id, alignment_type,
               alignment_confidence, alignment_confidence_band, en_text, zh_text,
               possible_alignment_error
        FROM alignments
        WHERE 1 = 1
    """
    parameters: list[object] = []
    if confidence_band is not None:
        sql += " AND alignment_confidence_band = ?"
        parameters.append(confidence_band)
    if alignment_type is not None:
        sql += " AND alignment_type = ?"
        parameters.append(alignment_type)
    if paragraph_no is not None:
        sql += " AND paragraph_no = ?"
        parameters.append(paragraph_no)
    if year is not None:
        sql += " AND year = ?"
        parameters.append(year)
    sql += " ORDER BY year, paragraph_no, alignment_id"

    active_config = eligibility_config or EligibilityConfig()
    records: list[AlignmentReviewRecord] = []
    for row in connection.execute(sql, parameters):
        possible_error = bool(row["possible_alignment_error"])
        status, reasons = derive_eligibility(
            alignment_confidence=row["alignment_confidence"],
            possible_alignment_error=possible_error,
            config=active_config,
        )
        if eligibility_status is not None and status != eligibility_status:
            continue
        records.append(
            AlignmentReviewRecord(
                year=row["year"],
                paragraph_no=row["paragraph_no"],
                alignment_id=row["alignment_id"],
                alignment_type=row["alignment_type"],
                alignment_confidence=row["alignment_confidence"],
                alignment_confidence_band=row["alignment_confidence_band"],
                english_text=row["en_text"],
                chinese_text=row["zh_text"],
                possible_alignment_error=possible_error,
                eligibility_status=status,
                eligibility_reasons=reasons,
            )
        )
    return tuple(records)


def format_alignment_review_report(
    records: tuple[AlignmentReviewRecord, ...],
) -> str:
    """Format review records as a deterministic tab-separated text report."""
    header = (
        "year\tparagraph_no\talignment_id\talignment_type\tconfidence\t"
        "confidence_band\tstatus\teligibility_reasons\t"
        "possible_alignment_error\tenglish_text\tchinese_text"
    )
    lines = [header]
    for record in records:
        english_text = record.english_text.replace("\t", " ").replace("\n", "\\n")
        chinese_text = record.chinese_text.replace("\t", " ").replace("\n", "\\n")
        reasons = "; ".join(record.eligibility_reasons)
        lines.append(
            f"{record.year}\t{record.paragraph_no}\t{record.alignment_id}\t"
            f"{record.alignment_type}\t{record.alignment_confidence:.6f}\t"
            f"{record.alignment_confidence_band}\t{record.eligibility_status}\t"
            f"{reasons}\t"
            f"{str(record.possible_alignment_error).lower()}\t"
            f"{english_text}\t{chinese_text}"
        )
    return "\n".join(lines)


def _stratified_sample_sizes(
    sample_size: int,
    proportions: Mapping[AlignmentType, float],
) -> dict[AlignmentType, int]:
    """Allocate integer sample sizes by deterministic largest remainder."""
    unsupported = set(proportions) - set(_ALIGNMENT_TYPES)
    if unsupported:
        raise ValueError(f"Unsupported alignment types: {sorted(unsupported)}")

    weights = {kind: float(proportions.get(kind, 0.0)) for kind in _ALIGNMENT_TYPES}
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("sampling proportions must not be negative")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("at least one sampling proportion must be positive")

    exact = {
        kind: sample_size * weight / total_weight for kind, weight in weights.items()
    }
    counts = {kind: math.floor(value) for kind, value in exact.items()}
    remainder = sample_size - sum(counts.values())
    ranked_types = sorted(
        _ALIGNMENT_TYPES,
        key=lambda kind: (-(exact[kind] - counts[kind]), _ALIGNMENT_TYPES.index(kind)),
    )
    for kind in ranked_types[:remainder]:
        counts[kind] += 1
    return counts


def sample_alignments_for_quality_check(
    connection: sqlite3.Connection,
    *,
    sample_size: int,
    random_seed: int,
    stratify_by_alignment_type: bool = False,
    alignment_type_proportions: Mapping[AlignmentType, float] | None = None,
    year: int | None = None,
    eligibility_status: EligibilityStatus | None = None,
    eligibility_config: EligibilityConfig | None = None,
) -> tuple[AlignmentReviewRecord, ...]:
    """Draw a reproducible quality-check sample without changing the database."""
    if sample_size < 0:
        raise ValueError("sample_size must not be negative")

    candidates = get_alignment_review_report(
        connection,
        year=year,
        eligibility_status=eligibility_status,
        eligibility_config=eligibility_config,
    )
    if sample_size > len(candidates):
        raise ValueError(
            f"sample_size {sample_size} exceeds {len(candidates)} available alignments"
        )

    generator = random.Random(random_seed)
    if not stratify_by_alignment_type:
        if alignment_type_proportions is not None:
            raise ValueError(
                "alignment_type_proportions requires stratify_by_alignment_type=True"
            )
        return tuple(generator.sample(list(candidates), sample_size))

    proportions = alignment_type_proportions or {
        alignment_type: 1.0 for alignment_type in _ALIGNMENT_TYPES
    }
    requested_counts = _stratified_sample_sizes(sample_size, proportions)
    candidates_by_type = {
        alignment_type: [
            record
            for record in candidates
            if record.alignment_type == alignment_type
        ]
        for alignment_type in _ALIGNMENT_TYPES
    }

    selected: list[AlignmentReviewRecord] = []
    for alignment_type in _ALIGNMENT_TYPES:
        requested = requested_counts[alignment_type]
        available = len(candidates_by_type[alignment_type])
        if requested > available:
            raise ValueError(
                f"Stratum {alignment_type} requests {requested} alignments "
                f"but only {available} are available"
            )
        selected.extend(
            generator.sample(candidates_by_type[alignment_type], requested)
        )
    generator.shuffle(selected)
    return tuple(selected)
