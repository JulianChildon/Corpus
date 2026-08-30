"""Independent double-annotation sampling and agreement statistics.

Agreement samples contain alignment IDs only. Each queue query joins annotations
for the requesting annotator and never selects the other annotator's labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import sqlite3
from typing import Any, Literal, Mapping, Sequence


SAMPLING_ALGORITHM_VERSION = "brics-shift-agreement-sample-v0.1"
LabelDimension = Literal["modality", "stance"]


class DuplicateAgreementStudyError(ValueError):
    """Raised when a study ID is already present."""


class AgreementStudyNotFoundError(LookupError):
    """Raised when an agreement study does not exist."""


class AgreementStudyIncompleteError(RuntimeError):
    """Raised when labels are requested before independent annotation is complete."""


@dataclass(frozen=True, slots=True)
class AgreementStudy:
    study_id: str
    study_name: str
    annotator_a_id: str
    annotator_b_id: str
    sample_size: int
    random_seed: int
    stratify_by_year: bool
    stratify_by_alignment_type: bool
    minimum_alignment_confidence: float | None
    include_possible_alignment_errors: bool
    sample_definition: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class AgreementProgress:
    sample_size: int
    annotator_a_count: int
    annotator_b_count: int
    doubly_annotated_count: int

    @property
    def is_complete(self) -> bool:
        return (
            self.annotator_a_count == self.sample_size
            and self.annotator_b_count == self.sample_size
        )


@dataclass(frozen=True, slots=True)
class LabelAgreement:
    dimension: LabelDimension
    doubly_annotated_units: int
    agreement_count: int
    raw_agreement: float | None
    cohen_kappa: float | None


@dataclass(frozen=True, slots=True)
class AgreementDisagreement:
    dimension: LabelDimension
    alignment_id: str
    english_text: str
    chinese_text: str
    annotator_a_label: str
    annotator_b_label: str


@dataclass(frozen=True, slots=True)
class AgreementReport:
    study: AgreementStudy
    progress: AgreementProgress
    modality: LabelAgreement
    stance: LabelAgreement
    disagreements: tuple[AgreementDisagreement, ...]


def _timestamp(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_at must include timezone information")
    return timestamp.astimezone(timezone.utc).isoformat()


def _study_from_row(row: sqlite3.Row) -> AgreementStudy:
    return AgreementStudy(
        study_id=row["study_id"],
        study_name=row["study_name"],
        annotator_a_id=row["annotator_a_id"],
        annotator_b_id=row["annotator_b_id"],
        sample_size=row["sample_size"],
        random_seed=row["random_seed"],
        stratify_by_year=bool(row["stratify_by_year"]),
        stratify_by_alignment_type=bool(row["stratify_by_alignment_type"]),
        minimum_alignment_confidence=row["minimum_alignment_confidence"],
        include_possible_alignment_errors=bool(
            row["include_possible_alignment_errors"]
        ),
        sample_definition=json.loads(row["sample_definition"]),
        created_at=row["created_at"],
    )


def get_agreement_study(
    connection: sqlite3.Connection, study_id: str
) -> AgreementStudy:
    row = connection.execute(
        "SELECT * FROM agreement_studies WHERE study_id = ?", (study_id,)
    ).fetchone()
    if row is None:
        raise AgreementStudyNotFoundError(f"Unknown agreement study: {study_id}")
    return _study_from_row(row)


def list_agreement_studies(
    connection: sqlite3.Connection,
) -> tuple[AgreementStudy, ...]:
    rows = connection.execute(
        "SELECT * FROM agreement_studies ORDER BY created_at, study_id"
    ).fetchall()
    return tuple(_study_from_row(row) for row in rows)


def _stratum_key(
    row: sqlite3.Row,
    *,
    stratify_by_year: bool,
    stratify_by_alignment_type: bool,
) -> str:
    values: dict[str, Any] = {}
    if stratify_by_year:
        values["year"] = row["year"]
    if stratify_by_alignment_type:
        values["alignment_type"] = row["alignment_type"]
    if not values:
        values["all"] = "all"
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _proportional_allocations(
    group_sizes: Mapping[str, int], sample_size: int
) -> dict[str, int]:
    population_size = sum(group_sizes.values())
    allocations = {
        key: sample_size * size // population_size
        for key, size in group_sizes.items()
    }
    remaining = sample_size - sum(allocations.values())
    remainder_order = sorted(
        group_sizes,
        key=lambda key: (
            -(sample_size * group_sizes[key] % population_size),
            key,
        ),
    )
    for key in remainder_order[:remaining]:
        allocations[key] += 1
    return allocations


def create_agreement_study(
    connection: sqlite3.Connection,
    *,
    study_id: str,
    study_name: str,
    annotator_a_id: str,
    annotator_b_id: str,
    sample_size: int,
    random_seed: int,
    stratify_by_year: bool = False,
    stratify_by_alignment_type: bool = False,
    minimum_alignment_confidence: float | None = 0.70,
    include_possible_alignment_errors: bool = False,
    created_at: datetime | None = None,
) -> AgreementStudy:
    """Create and persist one exact fixed-seed agreement sample.

    Stratification is proportional to the represented population. Largest
    remainders are assigned deterministically by the serialized stratum key.
    """
    text_fields = {
        "study_id": study_id,
        "study_name": study_name,
        "annotator_a_id": annotator_a_id,
        "annotator_b_id": annotator_b_id,
    }
    for field, value in text_fields.items():
        if not value.strip():
            raise ValueError(f"{field} must not be blank")
    if annotator_a_id == annotator_b_id:
        raise ValueError("agreement annotators must be different")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if minimum_alignment_confidence is not None and not (
        0 <= minimum_alignment_confidence <= 1
    ):
        raise ValueError("minimum_alignment_confidence must be between 0 and 1")

    sql = """
        SELECT alignment_id, year, alignment_type
        FROM alignments
        WHERE 1 = 1
    """
    parameters: list[Any] = []
    if minimum_alignment_confidence is not None:
        sql += " AND alignment_confidence >= ?"
        parameters.append(minimum_alignment_confidence)
    if not include_possible_alignment_errors:
        sql += " AND possible_alignment_error = 0"
    sql += " ORDER BY year, paragraph_no, alignment_id"
    candidates = connection.execute(sql, parameters).fetchall()
    if sample_size > len(candidates):
        raise ValueError(
            f"sample_size {sample_size} exceeds the eligible population "
            f"of {len(candidates)}"
        )

    groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for candidate in candidates:
        key = _stratum_key(
            candidate,
            stratify_by_year=stratify_by_year,
            stratify_by_alignment_type=stratify_by_alignment_type,
        )
        groups[key].append(candidate)
    allocations = _proportional_allocations(
        {key: len(rows) for key, rows in groups.items()}, sample_size
    )
    generator = random.Random(random_seed)
    selected: list[tuple[sqlite3.Row, str]] = []
    for key in sorted(groups):
        count = allocations[key]
        picked = generator.sample(groups[key], count)
        selected.extend((row, key) for row in picked)
    generator.shuffle(selected)

    population_ids = [row["alignment_id"] for row in candidates]
    population_hash = hashlib.sha256(
        "\n".join(population_ids).encode("utf-8")
    ).hexdigest()
    definition = {
        "sampling_algorithm_version": SAMPLING_ALGORITHM_VERSION,
        "sample_size": sample_size,
        "random_seed": random_seed,
        "stratify_by_year": stratify_by_year,
        "stratify_by_alignment_type": stratify_by_alignment_type,
        "stratification_method": "proportional_largest_remainder",
        "minimum_alignment_confidence": minimum_alignment_confidence,
        "include_possible_alignment_errors": include_possible_alignment_errors,
        "eligible_population_size": len(candidates),
        "eligible_population_alignment_ids_sha256": population_hash,
        "stratum_population_counts": {
            key: len(groups[key]) for key in sorted(groups)
        },
        "stratum_sample_allocations": {
            key: allocations[key] for key in sorted(allocations)
        },
    }
    definition_json = json.dumps(
        definition, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    created_at_text = _timestamp(created_at)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO agreement_studies (
                    study_id, study_name, annotator_a_id, annotator_b_id,
                    sample_size, random_seed, stratify_by_year,
                    stratify_by_alignment_type, minimum_alignment_confidence,
                    include_possible_alignment_errors, sample_definition,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study_id, study_name, annotator_a_id, annotator_b_id,
                    sample_size, random_seed, int(stratify_by_year),
                    int(stratify_by_alignment_type), minimum_alignment_confidence,
                    int(include_possible_alignment_errors), definition_json,
                    created_at_text,
                ),
            )
            connection.executemany(
                """
                INSERT INTO agreement_sample_items (
                    study_id, alignment_id, sample_order, stratum_key
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (study_id, row["alignment_id"], position, key)
                    for position, (row, key) in enumerate(selected, start=1)
                ),
            )
    except sqlite3.IntegrityError as error:
        duplicate = connection.execute(
            "SELECT 1 FROM agreement_studies WHERE study_id = ?", (study_id,)
        ).fetchone()
        if duplicate is not None:
            raise DuplicateAgreementStudyError(
                f"Agreement study {study_id!r} already exists"
            ) from error
        raise
    return get_agreement_study(connection, study_id)


def get_agreement_sample_ids(
    connection: sqlite3.Connection, study_id: str
) -> tuple[str, ...]:
    get_agreement_study(connection, study_id)
    rows = connection.execute(
        """
        SELECT alignment_id FROM agreement_sample_items
        WHERE study_id = ? ORDER BY sample_order
        """,
        (study_id,),
    ).fetchall()
    return tuple(row["alignment_id"] for row in rows)


def _require_study_annotator(study: AgreementStudy, annotator_id: str) -> None:
    if annotator_id not in {study.annotator_a_id, study.annotator_b_id}:
        raise ValueError(f"{annotator_id!r} is not an annotator in {study.study_id!r}")


def list_agreement_annotation_queue(
    connection: sqlite3.Connection,
    study_id: str,
    annotator_id: str,
    *,
    annotation_status: Literal["all", "annotated", "unannotated"] = "all",
) -> tuple[dict[str, Any], ...]:
    """Return sample units joined only to the requesting annotator's record."""
    study = get_agreement_study(connection, study_id)
    _require_study_annotator(study, annotator_id)
    if annotation_status not in {"all", "annotated", "unannotated"}:
        raise ValueError(f"Unsupported annotation status: {annotation_status}")
    sql = """
        SELECT asi.sample_order, asi.stratum_key, al.*,
               own.annotation_id, own.modality_label, own.stance_label,
               own.annotator_confidence, own.notes,
               own.possible_alignment_error AS annotation_possible_alignment_error,
               own.annotation_guideline_version,
               COALESCE(
                   (
                       SELECT d.title FROM alignment_sentences AS links
                       JOIN sentences AS s ON s.sentence_id = links.sentence_id
                       JOIN documents AS d ON d.document_id = s.document_id
                       WHERE links.alignment_id = al.alignment_id
                       ORDER BY links.language, links.sentence_position LIMIT 1
                   ),
                   '[Title unavailable]'
               ) AS document_title
        FROM agreement_sample_items AS asi
        JOIN alignments AS al ON al.alignment_id = asi.alignment_id
        LEFT JOIN annotations AS own
          ON own.alignment_id = al.alignment_id AND own.annotator_id = ?
        WHERE asi.study_id = ?
    """
    if annotation_status == "annotated":
        sql += " AND own.annotation_id IS NOT NULL"
    elif annotation_status == "unannotated":
        sql += " AND own.annotation_id IS NULL"
    sql += " ORDER BY asi.sample_order"
    records: list[dict[str, Any]] = []
    for row in connection.execute(sql, (annotator_id, study_id)):
        record = dict(row)
        record["en_sentence_ids"] = json.loads(record["en_sentence_ids"])
        record["zh_sentence_ids"] = json.loads(record["zh_sentence_ids"])
        record["possible_alignment_error"] = bool(
            record["possible_alignment_error"]
        )
        own_error = record["annotation_possible_alignment_error"]
        record["annotation_possible_alignment_error"] = (
            bool(own_error) if own_error is not None else None
        )
        records.append(record)
    return tuple(records)


def get_agreement_progress(
    connection: sqlite3.Connection, study_id: str
) -> AgreementProgress:
    study = get_agreement_study(connection, study_id)
    row = connection.execute(
        """
        SELECT
            COUNT(*) AS sample_size,
            SUM(CASE WHEN a.annotation_id IS NOT NULL THEN 1 ELSE 0 END) AS a_count,
            SUM(CASE WHEN b.annotation_id IS NOT NULL THEN 1 ELSE 0 END) AS b_count,
            SUM(CASE WHEN a.annotation_id IS NOT NULL
                      AND b.annotation_id IS NOT NULL THEN 1 ELSE 0 END) AS double_count
        FROM agreement_sample_items AS sample
        LEFT JOIN annotations AS a
          ON a.alignment_id = sample.alignment_id AND a.annotator_id = ?
        LEFT JOIN annotations AS b
          ON b.alignment_id = sample.alignment_id AND b.annotator_id = ?
        WHERE sample.study_id = ?
        """,
        (study.annotator_a_id, study.annotator_b_id, study_id),
    ).fetchone()
    return AgreementProgress(
        sample_size=row["sample_size"],
        annotator_a_count=row["a_count"] or 0,
        annotator_b_count=row["b_count"] or 0,
        doubly_annotated_count=row["double_count"] or 0,
    )


def calculate_cohens_kappa(
    annotator_a_labels: Sequence[str], annotator_b_labels: Sequence[str]
) -> tuple[int, float | None, float | None]:
    """Return agreement count, raw agreement, and Cohen's kappa.

    ``None`` is returned for rates when no paired labels exist, and for kappa
    when expected agreement is exactly one (the denominator is zero).
    """
    if len(annotator_a_labels) != len(annotator_b_labels):
        raise ValueError("label sequences must have the same length")
    total = len(annotator_a_labels)
    if total == 0:
        return 0, None, None
    agreement_count = sum(
        left == right
        for left, right in zip(annotator_a_labels, annotator_b_labels, strict=True)
    )
    observed = agreement_count / total
    counts_a = Counter(annotator_a_labels)
    counts_b = Counter(annotator_b_labels)
    labels = set(counts_a) | set(counts_b)
    expected = sum(
        (counts_a[label] / total) * (counts_b[label] / total)
        for label in labels
    )
    denominator = 1.0 - expected
    kappa = None if math.isclose(denominator, 0.0, abs_tol=1e-15) else (
        observed - expected
    ) / denominator
    return agreement_count, observed, kappa


def _label_agreement(
    rows: Sequence[sqlite3.Row], dimension: LabelDimension
) -> LabelAgreement:
    column = f"{dimension}_label"
    labels_a = [row[f"a_{column}"] for row in rows]
    labels_b = [row[f"b_{column}"] for row in rows]
    count, raw, kappa = calculate_cohens_kappa(labels_a, labels_b)
    return LabelAgreement(
        dimension=dimension,
        doubly_annotated_units=len(rows),
        agreement_count=count,
        raw_agreement=raw,
        cohen_kappa=kappa,
    )


def build_agreement_report(
    connection: sqlite3.Connection,
    study_id: str,
    *,
    reveal_before_complete: bool = False,
) -> AgreementReport:
    """Calculate separate modality and stance agreement for paired records.

    Results are hidden by default until both annotators finish the full sample,
    preserving independent annotation in the application workflow.
    """
    study = get_agreement_study(connection, study_id)
    progress = get_agreement_progress(connection, study_id)
    if not progress.is_complete and not reveal_before_complete:
        raise AgreementStudyIncompleteError(
            "Agreement labels remain hidden until both annotators complete the sample"
        )
    rows = connection.execute(
        """
        SELECT al.alignment_id, al.en_text, al.zh_text,
               a.modality_label AS a_modality_label,
               b.modality_label AS b_modality_label,
               a.stance_label AS a_stance_label,
               b.stance_label AS b_stance_label
        FROM agreement_sample_items AS sample
        JOIN alignments AS al ON al.alignment_id = sample.alignment_id
        JOIN annotations AS a
          ON a.alignment_id = sample.alignment_id AND a.annotator_id = ?
        JOIN annotations AS b
          ON b.alignment_id = sample.alignment_id AND b.annotator_id = ?
        WHERE sample.study_id = ?
        ORDER BY sample.sample_order
        """,
        (study.annotator_a_id, study.annotator_b_id, study_id),
    ).fetchall()
    disagreements: list[AgreementDisagreement] = []
    for row in rows:
        for dimension in ("modality", "stance"):
            left = row[f"a_{dimension}_label"]
            right = row[f"b_{dimension}_label"]
            if left != right:
                disagreements.append(
                    AgreementDisagreement(
                        dimension=dimension,
                        alignment_id=row["alignment_id"],
                        english_text=row["en_text"],
                        chinese_text=row["zh_text"],
                        annotator_a_label=left,
                        annotator_b_label=right,
                    )
                )
    return AgreementReport(
        study=study,
        progress=progress,
        modality=_label_agreement(rows, "modality"),
        stance=_label_agreement(rows, "stance"),
        disagreements=tuple(disagreements),
    )
