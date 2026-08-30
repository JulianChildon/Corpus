"""Deterministic exports for BRICS-Shift research data.

The complete alignment and annotation exports are kept separate. The joined
research dataset is an analysis-ready view and may apply an explicitly recorded
confidence/error policy. Raw source documents are never included.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from brics_shift import __version__
from brics_shift.database import list_alignments_for_annotation


ALIGNED_CORPUS_COLUMNS: tuple[str, ...] = (
    "alignment_id", "year", "paragraph_no", "alignment_type",
    "alignment_confidence", "en_sentence_ids", "zh_sentence_ids",
    "en_text", "zh_text",
)

ANNOTATION_EXPORT_COLUMNS: tuple[str, ...] = (
    "alignment_id", "annotator_id", "modality_label", "stance_label",
    "annotator_confidence", "notes", "possible_alignment_error",
    "guideline_version", "created_at", "updated_at",
)

RESEARCH_DATASET_COLUMNS: tuple[str, ...] = (
    *ALIGNED_CORPUS_COLUMNS,
    "alignment_confidence_band", "alignment_algorithm_version",
    "annotation_id", "annotator_id", "modality_label", "stance_label",
    "annotator_confidence", "notes", "possible_alignment_error",
    "guideline_version", "annotation_created_at", "annotation_updated_at",
    "en_document_ids", "zh_document_ids", "en_paragraph_ids",
    "zh_paragraph_ids", "en_source_filenames", "zh_source_filenames",
    "en_document_checksums", "zh_document_checksums",
    "preprocessing_versions", "segmentation_versions",
)

PILOT_EXPORT_COLUMNS: tuple[str, ...] = (
    "alignment_id", "year", "paragraph_no", "alignment_type",
    "alignment_confidence", "english_text", "chinese_text",
    "modality_label", "stance_label", "annotator_confidence", "notes",
    "possible_alignment_error", "guideline_version",
)


class TraceabilityError(RuntimeError):
    """Raised when an exported alignment lacks an enforceable source link."""


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Transparent selection policy for the joined research dataset."""

    alignment_confidence_threshold: float = 0.70
    annotator_id: str | None = None
    include_possible_alignment_errors: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.alignment_confidence_threshold <= 1:
            raise ValueError("alignment_confidence_threshold must be between 0 and 1")
        if self.annotator_id is not None and not self.annotator_id.strip():
            raise ValueError("annotator_id must not be blank")


def _csv_text(columns: Sequence[str], records: Iterable[Mapping[str, Any]]) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def _json_list(value: str | Sequence[str]) -> str:
    decoded = json.loads(value) if isinstance(value, str) else list(value)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise TraceabilityError("sentence ID fields must contain JSON arrays of strings")
    return json.dumps(decoded, ensure_ascii=False, separators=(",", ":"))


def _json_values(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def _alignment_records(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT alignment_id, year, paragraph_no, alignment_type,
               alignment_confidence, en_sentence_ids, zh_sentence_ids,
               en_text, zh_text
        FROM alignments
        ORDER BY year, paragraph_no, alignment_id
        """
    ).fetchall()
    return [
        {
            **dict(row),
            "en_sentence_ids": _json_list(row["en_sentence_ids"]),
            "zh_sentence_ids": _json_list(row["zh_sentence_ids"]),
        }
        for row in rows
    ]


def export_aligned_corpus_csv(connection: sqlite3.Connection) -> str:
    """Export every derived alignment, including quarantined alignments."""
    return _csv_text(ALIGNED_CORPUS_COLUMNS, _alignment_records(connection))


def _annotation_records(
    connection: sqlite3.Connection,
    *,
    annotator_id: str | None = None,
) -> list[dict[str, Any]]:
    if annotator_id is not None and not annotator_id.strip():
        raise ValueError("annotator_id must not be blank")
    rows = connection.execute(
        """
        SELECT alignment_id, annotator_id, modality_label, stance_label,
               annotator_confidence, notes, possible_alignment_error,
               annotation_guideline_version, created_at, updated_at
        FROM annotations
        WHERE (? IS NULL OR annotator_id = ?)
        ORDER BY alignment_id, annotator_id
        """,
        (annotator_id, annotator_id),
    ).fetchall()
    return [
        {
            "alignment_id": row["alignment_id"],
            "annotator_id": row["annotator_id"],
            "modality_label": row["modality_label"],
            "stance_label": row["stance_label"],
            "annotator_confidence": row["annotator_confidence"],
            "notes": row["notes"],
            "possible_alignment_error": str(bool(row["possible_alignment_error"])).lower(),
            "guideline_version": row["annotation_guideline_version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def export_annotations_csv(
    connection: sqlite3.Connection,
    *,
    annotator_id: str | None = None,
) -> str:
    """Export complete human annotations without alignment-derived filtering."""
    return _csv_text(
        ANNOTATION_EXPORT_COLUMNS,
        _annotation_records(connection, annotator_id=annotator_id),
    )


def _traceability_by_alignment(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows = connection.execute(
        """
        SELECT als.alignment_id, als.language, als.sentence_position,
               s.sentence_id, s.sentence_no, s.sentence_text,
               s.segmentation_version, p.paragraph_id, p.paragraph_no,
               p.original_order, d.document_id, d.year, d.title,
               d.source_filename, d.source_url, d.checksum,
               d.preprocessing_version
        FROM alignment_sentences AS als
        JOIN sentences AS s ON s.sentence_id = als.sentence_id
        JOIN paragraphs AS p ON p.paragraph_id = s.paragraph_id
        JOIN documents AS d ON d.document_id = s.document_id
        ORDER BY als.alignment_id, als.language, als.sentence_position
        """
    ).fetchall()
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        alignment = result.setdefault(row["alignment_id"], {"EN": [], "ZH": []})
        alignment[row["language"]].append(
            {
                "sentence_id": row["sentence_id"],
                "sentence_no": row["sentence_no"],
                "sentence_text": row["sentence_text"],
                "segmentation_version": row["segmentation_version"],
                "paragraph_id": row["paragraph_id"],
                "paragraph_no": row["paragraph_no"],
                "paragraph_original_order": row["original_order"],
                "document_id": row["document_id"],
                "document_year": row["year"],
                "document_title": row["title"],
                "source_filename": row["source_filename"],
                "source_url": row["source_url"],
                "document_checksum": row["checksum"],
                "preprocessing_version": row["preprocessing_version"],
            }
        )
    return result


def _selected_research_rows(
    connection: sqlite3.Connection,
    config: ExportConfig,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT al.*, an.annotation_id, an.annotator_id, an.modality_label,
               an.stance_label, an.annotator_confidence, an.notes,
               an.possible_alignment_error AS annotation_alignment_error,
               an.annotation_guideline_version,
               an.created_at AS annotation_created_at,
               an.updated_at AS annotation_updated_at
        FROM alignments AS al
        JOIN annotations AS an ON an.alignment_id = al.alignment_id
        WHERE al.alignment_confidence >= ?
          AND (? IS NULL OR an.annotator_id = ?)
          AND (? = 1 OR (al.possible_alignment_error = 0
                         AND an.possible_alignment_error = 0))
        ORDER BY al.year, al.paragraph_no, al.alignment_id, an.annotator_id
        """,
        (
            config.alignment_confidence_threshold,
            config.annotator_id,
            config.annotator_id,
            int(config.include_possible_alignment_errors),
        ),
    ).fetchall()


def _research_objects(
    connection: sqlite3.Connection,
    config: ExportConfig,
) -> list[dict[str, Any]]:
    traces = _traceability_by_alignment(connection)
    objects: list[dict[str, Any]] = []
    for row in _selected_research_rows(connection, config):
        trace = traces.get(row["alignment_id"], {"EN": [], "ZH": []})
        if not trace["EN"] or not trace["ZH"]:
            raise TraceabilityError(
                f"Alignment {row['alignment_id']} has incomplete sentence traceability"
            )
        expected_en = json.loads(row["en_sentence_ids"])
        expected_zh = json.loads(row["zh_sentence_ids"])
        linked_en = [item["sentence_id"] for item in trace["EN"]]
        linked_zh = [item["sentence_id"] for item in trace["ZH"]]
        if expected_en != linked_en or expected_zh != linked_zh:
            raise TraceabilityError(
                f"Alignment {row['alignment_id']} sentence IDs disagree with its links"
            )
        linked_en_text = " ".join(item["sentence_text"] for item in trace["EN"])
        linked_zh_text = " ".join(item["sentence_text"] for item in trace["ZH"])
        if row["en_text"] != linked_en_text or row["zh_text"] != linked_zh_text:
            raise TraceabilityError(
                f"Alignment {row['alignment_id']} text disagrees with its linked sentences"
            )

        effective_error = bool(
            row["possible_alignment_error"] or row["annotation_alignment_error"]
        )
        objects.append(
            {
                "alignment": {
                    "alignment_id": row["alignment_id"],
                    "year": row["year"],
                    "paragraph_no": row["paragraph_no"],
                    "alignment_type": row["alignment_type"],
                    "alignment_confidence": row["alignment_confidence"],
                    "alignment_confidence_band": row["alignment_confidence_band"],
                    "alignment_algorithm_version": row["alignment_algorithm_version"],
                    "en_sentence_ids": expected_en,
                    "zh_sentence_ids": expected_zh,
                    "en_text": row["en_text"],
                    "zh_text": row["zh_text"],
                },
                "annotation": {
                    "annotation_id": row["annotation_id"],
                    "annotator_id": row["annotator_id"],
                    "modality_label": row["modality_label"],
                    "stance_label": row["stance_label"],
                    "annotator_confidence": row["annotator_confidence"],
                    "notes": row["notes"],
                    "possible_alignment_error": effective_error,
                    "guideline_version": row["annotation_guideline_version"],
                    "created_at": row["annotation_created_at"],
                    "updated_at": row["annotation_updated_at"],
                },
                "traceability": {
                    "english_sentences": trace["EN"],
                    "chinese_sentences": trace["ZH"],
                },
            }
        )
    return objects


def _flatten_research_object(record: Mapping[str, Any]) -> dict[str, Any]:
    alignment = record["alignment"]
    annotation = record["annotation"]
    english = record["traceability"]["english_sentences"]
    chinese = record["traceability"]["chinese_sentences"]
    all_sentences = [*english, *chinese]
    return {
        "alignment_id": alignment["alignment_id"],
        "year": alignment["year"],
        "paragraph_no": alignment["paragraph_no"],
        "alignment_type": alignment["alignment_type"],
        "alignment_confidence": alignment["alignment_confidence"],
        "en_sentence_ids": json.dumps(alignment["en_sentence_ids"], ensure_ascii=False, separators=(",", ":")),
        "zh_sentence_ids": json.dumps(alignment["zh_sentence_ids"], ensure_ascii=False, separators=(",", ":")),
        "en_text": alignment["en_text"],
        "zh_text": alignment["zh_text"],
        "alignment_confidence_band": alignment["alignment_confidence_band"],
        "alignment_algorithm_version": alignment["alignment_algorithm_version"],
        "annotation_id": annotation["annotation_id"],
        "annotator_id": annotation["annotator_id"],
        "modality_label": annotation["modality_label"],
        "stance_label": annotation["stance_label"],
        "annotator_confidence": annotation["annotator_confidence"],
        "notes": annotation["notes"],
        "possible_alignment_error": str(annotation["possible_alignment_error"]).lower(),
        "guideline_version": annotation["guideline_version"],
        "annotation_created_at": annotation["created_at"],
        "annotation_updated_at": annotation["updated_at"],
        "en_document_ids": _json_values(x["document_id"] for x in english),
        "zh_document_ids": _json_values(x["document_id"] for x in chinese),
        "en_paragraph_ids": _json_values(x["paragraph_id"] for x in english),
        "zh_paragraph_ids": _json_values(x["paragraph_id"] for x in chinese),
        "en_source_filenames": _json_values(x["source_filename"] for x in english),
        "zh_source_filenames": _json_values(x["source_filename"] for x in chinese),
        "en_document_checksums": _json_values(x["document_checksum"] for x in english),
        "zh_document_checksums": _json_values(x["document_checksum"] for x in chinese),
        "preprocessing_versions": _json_values(x["preprocessing_version"] for x in all_sentences),
        "segmentation_versions": _json_values(x["segmentation_version"] for x in all_sentences),
    }


def export_research_dataset_csv(
    connection: sqlite3.Connection,
    config: ExportConfig | None = None,
) -> str:
    """Export the policy-selected alignment/annotation join as CSV."""
    objects = _research_objects(connection, config or ExportConfig())
    return _csv_text(
        RESEARCH_DATASET_COLUMNS,
        (_flatten_research_object(record) for record in objects),
    )


def export_research_dataset_jsonl(
    connection: sqlite3.Connection,
    config: ExportConfig | None = None,
) -> str:
    """Export nested, source-traceable research records as UTF-8 JSONL."""
    records = _research_objects(connection, config or ExportConfig())
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )


def _utc_timestamp(value: datetime | None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("export_timestamp must include timezone information")
    return timestamp.astimezone(timezone.utc).isoformat()


def _distinct_values(
    connection: sqlite3.Connection, table: str, column: str
) -> list[str]:
    # Identifiers are private constants at every call site.
    rows = connection.execute(
        f"SELECT DISTINCT {column} FROM {table} ORDER BY {column}"
    ).fetchall()
    return [row[0] for row in rows]


def build_export_metadata(
    connection: sqlite3.Connection,
    config: ExportConfig | None = None,
    *,
    export_timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build machine-readable versions, policy, and row-count metadata."""
    selected = config or ExportConfig()
    annotation_filter = " WHERE annotator_id = ?" if selected.annotator_id else ""
    annotation_parameters: tuple[Any, ...] = (
        (selected.annotator_id,) if selected.annotator_id else ()
    )
    alignment_count = connection.execute("SELECT COUNT(*) FROM alignments").fetchone()[0]
    annotation_count = connection.execute(
        f"SELECT COUNT(*) FROM annotations{annotation_filter}", annotation_parameters
    ).fetchone()[0]
    research_count = len(_selected_research_rows(connection, selected))
    quarantined_count = connection.execute(
        "SELECT COUNT(*) FROM alignments WHERE alignment_confidence < ?",
        (selected.alignment_confidence_threshold,),
    ).fetchone()[0]
    error_count = connection.execute(
        """
        SELECT COUNT(*) FROM annotations AS an
        JOIN alignments AS al ON al.alignment_id = an.alignment_id
        WHERE (al.possible_alignment_error = 1 OR an.possible_alignment_error = 1)
          AND (? IS NULL OR an.annotator_id = ?)
        """,
        (selected.annotator_id, selected.annotator_id),
    ).fetchone()[0]
    guideline_sql = "SELECT DISTINCT annotation_guideline_version FROM annotations"
    guideline_parameters: tuple[Any, ...] = ()
    if selected.annotator_id:
        guideline_sql += " WHERE annotator_id = ?"
        guideline_parameters = (selected.annotator_id,)
    guideline_sql += " ORDER BY annotation_guideline_version"

    return {
        "project_version": __version__,
        "preprocessing_version": _distinct_values(connection, "documents", "preprocessing_version"),
        "segmentation_version": _distinct_values(connection, "sentences", "segmentation_version"),
        "alignment_algorithm_version": _distinct_values(connection, "alignments", "alignment_algorithm_version"),
        "annotation_guideline_version": [
            row[0] for row in connection.execute(guideline_sql, guideline_parameters).fetchall()
        ],
        "export_timestamp": _utc_timestamp(export_timestamp),
        "alignment_confidence_threshold": selected.alignment_confidence_threshold,
        "annotator_filter": selected.annotator_id,
        "possible_alignment_error_policy": (
            "included" if selected.include_possible_alignment_errors
            else "excluded_from_research_dataset"
        ),
        "raw_source_files_included": False,
        "text_encoding": "UTF-8",
        "row_counts": {
            "aligned_corpus": alignment_count,
            "annotations": annotation_count,
            "research_dataset": research_count,
            "alignments_below_confidence_threshold": quarantined_count,
            "annotations_flagged_possible_alignment_error": error_count,
        },
    }


def export_metadata_json(
    connection: sqlite3.Connection,
    config: ExportConfig | None = None,
    *,
    export_timestamp: datetime | None = None,
) -> str:
    """Serialize reproducibility metadata with stable key ordering."""
    return json.dumps(
        build_export_metadata(connection, config, export_timestamp=export_timestamp),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def build_reproducible_export_files(
    connection: sqlite3.Connection,
    config: ExportConfig | None = None,
    *,
    export_timestamp: datetime | None = None,
) -> dict[str, str]:
    """Return the five-file reproducible export bundle in a stable order."""
    selected = config or ExportConfig()
    return {
        "aligned_corpus.csv": export_aligned_corpus_csv(connection),
        "annotations.csv": export_annotations_csv(connection, annotator_id=selected.annotator_id),
        "research_dataset.csv": export_research_dataset_csv(connection, selected),
        "research_dataset.jsonl": export_research_dataset_jsonl(connection, selected),
        "export_metadata.json": export_metadata_json(
            connection, selected, export_timestamp=export_timestamp
        ),
    }


def write_reproducible_export(
    connection: sqlite3.Connection,
    output_directory: str | Path,
    config: ExportConfig | None = None,
    *,
    export_timestamp: datetime | None = None,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write a bundle to disk, refusing implicit replacement by default."""
    files = build_reproducible_export_files(
        connection, config, export_timestamp=export_timestamp
    )
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / filename for filename in files)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing export files: {names}")
    for path, content in zip(paths, files.values(), strict=True):
        path.write_text(content, encoding="utf-8", newline="")
    return paths


def export_pilot_annotations_csv(
    connection: sqlite3.Connection,
    annotator_id: str,
    *,
    year: int | None = None,
    alignment_type: str | None = None,
    confidence_band: str | None = None,
    possible_alignment_error: bool | None = None,
    modality_label: str | None = None,
    stance_label: str | None = None,
    notes_search: str | None = None,
) -> str:
    """Return one annotator's filtered pilot review CSV (legacy v0.1 view)."""
    records = list_alignments_for_annotation(
        connection, annotator_id, year=year, minimum_confidence=None,
        include_possible_alignment_errors=True, alignment_type=alignment_type,
        confidence_band=confidence_band, annotation_status="annotated",
        possible_alignment_error=possible_alignment_error,
        modality_label=modality_label, stance_label=stance_label,
        notes_search=notes_search,
    )
    rows = (
        {
            "alignment_id": record["alignment_id"],
            "year": record["year"],
            "paragraph_no": record["paragraph_no"],
            "alignment_type": record["alignment_type"],
            "alignment_confidence": record["alignment_confidence"],
            "english_text": record["en_text"],
            "chinese_text": record["zh_text"],
            "modality_label": record["modality_label"],
            "stance_label": record["stance_label"],
            "annotator_confidence": record["annotator_confidence"],
            "notes": record["notes"],
            "possible_alignment_error": str(bool(record["annotation_possible_alignment_error"])).lower(),
            "guideline_version": record["annotation_guideline_version"],
        }
        for record in records
    )
    return _csv_text(PILOT_EXPORT_COLUMNS, rows)
