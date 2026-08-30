"""SQLite schema and persistence helpers for corpus records.

This module deliberately uses the standard :mod:`sqlite3` API. Inserts never
use ``OR REPLACE`` or an implicit upsert, so existing research records cannot
be silently overwritten.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from os import PathLike
from pathlib import Path
import sqlite3
from typing import Any, Sequence

from brics_shift.alignment import AlignmentUnit, SentenceRecord
from brics_shift.annotation import (
    AnnotatorConfidence,
    ModalityLabel,
    StanceLabel,
    validate_annotation_values,
)
from brics_shift.cleaning import ImportedDocument
from brics_shift.paragraphs import ParsedParagraph


SCHEMA_VERSION = "0.1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    year INTEGER NOT NULL CHECK (year > 0),
    title TEXT NOT NULL CHECK (length(trim(title)) > 0),
    language TEXT NOT NULL CHECK (language IN ('EN', 'ZH')),
    source_filename TEXT NOT NULL CHECK (length(trim(source_filename)) > 0),
    source_url TEXT,
    imported_at TEXT NOT NULL,
    checksum TEXT NOT NULL CHECK (length(trim(checksum)) > 0),
    preprocessing_version TEXT NOT NULL
        CHECK (length(trim(preprocessing_version)) > 0),
    UNIQUE (document_id, language)
);

CREATE TABLE IF NOT EXISTS paragraphs (
    paragraph_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    paragraph_no INTEGER NOT NULL CHECK (paragraph_no > 0),
    paragraph_text TEXT NOT NULL,
    original_order INTEGER NOT NULL CHECK (original_order > 0),
    UNIQUE (document_id, paragraph_no),
    UNIQUE (document_id, original_order),
    UNIQUE (paragraph_id, document_id, paragraph_no),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS sentences (
    sentence_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    paragraph_id TEXT NOT NULL,
    paragraph_no INTEGER NOT NULL CHECK (paragraph_no > 0),
    sentence_no INTEGER NOT NULL CHECK (sentence_no > 0),
    language TEXT NOT NULL CHECK (language IN ('EN', 'ZH')),
    sentence_text TEXT NOT NULL CHECK (length(trim(sentence_text)) > 0),
    segmentation_version TEXT NOT NULL
        CHECK (length(trim(segmentation_version)) > 0),
    UNIQUE (document_id, paragraph_no, sentence_no),
    UNIQUE (sentence_id, language),
    FOREIGN KEY (document_id, language)
        REFERENCES documents(document_id, language)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (paragraph_id, document_id, paragraph_no)
        REFERENCES paragraphs(paragraph_id, document_id, paragraph_no)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS alignments (
    alignment_id TEXT PRIMARY KEY,
    year INTEGER NOT NULL CHECK (year > 0),
    paragraph_no INTEGER NOT NULL CHECK (paragraph_no > 0),
    en_sentence_ids TEXT NOT NULL,
    zh_sentence_ids TEXT NOT NULL,
    en_text TEXT NOT NULL,
    zh_text TEXT NOT NULL,
    alignment_type TEXT NOT NULL
        CHECK (alignment_type IN ('1:1', '1:2', '2:1', '2:2')),
    alignment_cost REAL NOT NULL CHECK (alignment_cost >= 0),
    alignment_confidence REAL NOT NULL
        CHECK (alignment_confidence >= 0 AND alignment_confidence <= 1),
    alignment_confidence_band TEXT NOT NULL
        CHECK (alignment_confidence_band IN ('high', 'medium', 'low')),
    alignment_algorithm_version TEXT NOT NULL
        CHECK (length(trim(alignment_algorithm_version)) > 0),
    possible_alignment_error INTEGER NOT NULL DEFAULT 0
        CHECK (possible_alignment_error IN (0, 1)),
    created_at TEXT NOT NULL
);

-- JSON sentence lists remain in alignments for convenient v0.1 export. This
-- junction table makes every listed relationship enforceable and queryable.
CREATE TABLE IF NOT EXISTS alignment_sentences (
    alignment_id TEXT NOT NULL,
    sentence_id TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('EN', 'ZH')),
    sentence_position INTEGER NOT NULL CHECK (sentence_position > 0),
    PRIMARY KEY (alignment_id, sentence_id),
    UNIQUE (alignment_id, language, sentence_position),
    FOREIGN KEY (alignment_id) REFERENCES alignments(alignment_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (sentence_id, language) REFERENCES sentences(sentence_id, language)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id TEXT PRIMARY KEY,
    alignment_id TEXT NOT NULL,
    annotator_id TEXT NOT NULL CHECK (length(trim(annotator_id)) > 0),
    modality_label TEXT NOT NULL
        CHECK (
            modality_label IN (
                'N/A', 'preserved', 'strengthened', 'weakened',
                'added', 'omitted', 'uncertain'
            )
        ),
    stance_label TEXT NOT NULL
        CHECK (
            stance_label IN (
                'N/A', 'preserved', 'strengthened', 'weakened',
                'neutralized', 'uncertain'
            )
        ),
    annotator_confidence TEXT NOT NULL
        CHECK (annotator_confidence IN ('high', 'medium', 'low')),
    notes TEXT NOT NULL DEFAULT '',
    possible_alignment_error INTEGER NOT NULL DEFAULT 0
        CHECK (possible_alignment_error IN (0, 1)),
    annotation_guideline_version TEXT NOT NULL
        CHECK (length(trim(annotation_guideline_version)) > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (alignment_id, annotator_id),
    FOREIGN KEY (alignment_id) REFERENCES alignments(alignment_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS agreement_studies (
    study_id TEXT PRIMARY KEY,
    study_name TEXT NOT NULL CHECK (length(trim(study_name)) > 0),
    annotator_a_id TEXT NOT NULL CHECK (length(trim(annotator_a_id)) > 0),
    annotator_b_id TEXT NOT NULL CHECK (length(trim(annotator_b_id)) > 0),
    sample_size INTEGER NOT NULL CHECK (sample_size > 0),
    random_seed INTEGER NOT NULL,
    stratify_by_year INTEGER NOT NULL CHECK (stratify_by_year IN (0, 1)),
    stratify_by_alignment_type INTEGER NOT NULL
        CHECK (stratify_by_alignment_type IN (0, 1)),
    minimum_alignment_confidence REAL
        CHECK (
            minimum_alignment_confidence IS NULL
            OR (minimum_alignment_confidence >= 0
                AND minimum_alignment_confidence <= 1)
        ),
    include_possible_alignment_errors INTEGER NOT NULL DEFAULT 0
        CHECK (include_possible_alignment_errors IN (0, 1)),
    sample_definition TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (annotator_a_id <> annotator_b_id)
);

CREATE TABLE IF NOT EXISTS agreement_sample_items (
    study_id TEXT NOT NULL,
    alignment_id TEXT NOT NULL,
    sample_order INTEGER NOT NULL CHECK (sample_order > 0),
    stratum_key TEXT NOT NULL,
    PRIMARY KEY (study_id, alignment_id),
    UNIQUE (study_id, sample_order),
    FOREIGN KEY (study_id) REFERENCES agreement_studies(study_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (alignment_id) REFERENCES alignments(alignment_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_paragraphs_document
    ON paragraphs(document_id, original_order);
CREATE INDEX IF NOT EXISTS idx_sentences_paragraph
    ON sentences(paragraph_id, sentence_no);
CREATE INDEX IF NOT EXISTS idx_alignments_year_paragraph
    ON alignments(year, paragraph_no, alignment_id);
CREATE INDEX IF NOT EXISTS idx_annotations_annotator
    ON annotations(annotator_id, alignment_id);
CREATE INDEX IF NOT EXISTS idx_agreement_sample_order
    ON agreement_sample_items(study_id, sample_order);
"""


class DuplicateAnnotationError(ValueError):
    """Raised when an annotator tries to insert a second annotation."""


class RecordNotFoundError(LookupError):
    """Raised when an explicit update target does not exist."""


@dataclass(frozen=True, slots=True)
class AnnotationProgress:
    """Annotation completion counts for one annotator."""

    annotator_id: str
    total_alignments: int
    annotated_alignments: int
    remaining_alignments: int
    completion_percentage: float


@dataclass(frozen=True, slots=True)
class CorpusOverview:
    """Basic corpus counts used by the Corpus page."""

    document_count: int
    years: tuple[int, ...]
    english_sentence_count: int
    chinese_sentence_count: int
    aligned_unit_count: int
    count_1_1: int
    count_1_2: int
    count_2_1: int
    count_2_2: int
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int


def initialize_database(
    database_path: str | PathLike[str] = "data/processed/brics_shift.sqlite3",
) -> sqlite3.Connection:
    """Open a database, enable foreign keys, and create the v0.1 schema."""
    path_text = str(database_path)
    if path_text != ":memory:":
        Path(path_text).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path_text)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    with connection:
        connection.executescript(_SCHEMA)
        _ensure_v01_columns(connection)
    return connection


def _ensure_v01_columns(connection: sqlite3.Connection) -> None:
    """Apply small additive v0.1 migrations to databases already created."""
    alignment_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(alignments)")
    }
    if "possible_alignment_error" not in alignment_columns:
        connection.execute(
            """
            ALTER TABLE alignments
            ADD COLUMN possible_alignment_error INTEGER NOT NULL DEFAULT 0
                CHECK (possible_alignment_error IN (0, 1))
            """
        )

    annotation_columns = {
        row["name"]: row["type"].upper()
        for row in connection.execute("PRAGMA table_info(annotations)")
    }
    if annotation_columns.get("annotator_confidence") != "TEXT":
        raise RuntimeError(
            "This database uses the earlier numeric annotator-confidence "
            "schema. No automatic conversion was attempted because mapping "
            "numeric values to high/medium/low requires a researcher decision."
        )


def calculate_file_checksum(
    path: str | PathLike[str],
    *,
    algorithm: str = "sha256",
) -> str:
    """Return a hexadecimal content hash of the original source bytes."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamps must include timezone information")
    return timestamp.astimezone(timezone.utc).isoformat()


def paragraph_id_for(document_id: str, paragraph_no: int) -> str:
    """Create the deterministic v0.1 paragraph identifier."""
    if not document_id.strip() or paragraph_no < 1:
        raise ValueError("document_id and a positive paragraph_no are required")
    return f"{document_id}_P{paragraph_no:03d}"


def insert_document(
    connection: sqlite3.Connection,
    document: ImportedDocument,
    *,
    checksum: str,
    preprocessing_version: str,
    commit: bool = True,
) -> None:
    """Insert one imported document without overwriting an existing row."""
    with connection if commit else nullcontext():
        connection.execute(
            """
            INSERT INTO documents (
                document_id, year, title, language, source_filename, source_url,
                imported_at, checksum, preprocessing_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.document_id,
                document.year,
                document.title,
                document.language,
                document.source_filename,
                document.source_url,
                _timestamp(document.imported_at),
                checksum,
                preprocessing_version,
            ),
        )


def insert_paragraphs(
    connection: sqlite3.Connection,
    paragraphs: Sequence[ParsedParagraph],
    *,
    commit: bool = True,
) -> tuple[str, ...]:
    """Insert a batch of parsed paragraphs atomically."""
    paragraph_ids: list[str] = []
    with connection if commit else nullcontext():
        for paragraph in paragraphs:
            paragraph_id = paragraph_id_for(
                paragraph.document_id, paragraph.paragraph_no
            )
            connection.execute(
                """
                INSERT INTO paragraphs (
                    paragraph_id, document_id, paragraph_no,
                    paragraph_text, original_order
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    paragraph_id,
                    paragraph.document_id,
                    paragraph.paragraph_no,
                    paragraph.cleaned_paragraph_text,
                    paragraph.original_order,
                ),
            )
            paragraph_ids.append(paragraph_id)
    return tuple(paragraph_ids)


def insert_sentences(
    connection: sqlite3.Connection,
    sentences: Sequence[SentenceRecord],
    *,
    segmentation_version: str,
    commit: bool = True,
) -> None:
    """Insert already-segmented sentences atomically."""
    with connection if commit else nullcontext():
        for sentence in sentences:
            connection.execute(
                """
                INSERT INTO sentences (
                    sentence_id, document_id, paragraph_id, paragraph_no,
                    sentence_no, language, sentence_text, segmentation_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sentence.sentence_id,
                    sentence.document_id,
                    paragraph_id_for(sentence.document_id, sentence.paragraph_no),
                    sentence.paragraph_no,
                    sentence.sentence_no,
                    sentence.language,
                    sentence.text,
                    segmentation_version,
                ),
            )


def _validate_alignment_sentence(
    connection: sqlite3.Connection,
    sentence_id: str,
    *,
    expected_language: str,
    expected_paragraph_no: int,
    expected_year: int,
) -> tuple[str, str]:
    row = connection.execute(
        """
        SELECT s.language, s.paragraph_no, s.sentence_text,
               s.document_id, d.year
        FROM sentences AS s
        JOIN documents AS d ON d.document_id = s.document_id
        WHERE s.sentence_id = ?
        """,
        (sentence_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown sentence_id in alignment: {sentence_id}")
    if row["language"] != expected_language:
        raise ValueError(f"Sentence {sentence_id} is not {expected_language}")
    if row["paragraph_no"] != expected_paragraph_no:
        raise ValueError(
            f"Sentence {sentence_id} is not in paragraph {expected_paragraph_no}"
        )
    if row["year"] != expected_year:
        raise ValueError(f"Sentence {sentence_id} is not from year {expected_year}")
    existing_link = connection.execute(
        """
        SELECT alignment_id FROM alignment_sentences
        WHERE sentence_id = ? LIMIT 1
        """,
        (sentence_id,),
    ).fetchone()
    if existing_link is not None:
        raise ValueError(
            f"Sentence {sentence_id} is already linked to alignment "
            f"{existing_link['alignment_id']}"
        )
    return row["sentence_text"], row["document_id"]


def insert_alignments(
    connection: sqlite3.Connection,
    alignments: Sequence[AlignmentUnit],
    *,
    year: int,
    created_at: datetime | None = None,
    commit: bool = True,
) -> None:
    """Insert alignments and their sentence links in one transaction."""
    created_at_text = _timestamp(created_at)
    with connection if commit else nullcontext():
        for alignment in alignments:
            if not alignment.english_sentence_ids or not alignment.chinese_sentence_ids:
                raise ValueError("alignment sentence ID lists must not be empty")
            if len(set(alignment.english_sentence_ids)) != len(
                alignment.english_sentence_ids
            ) or len(set(alignment.chinese_sentence_ids)) != len(
                alignment.chinese_sentence_ids
            ):
                raise ValueError("alignment sentence ID lists must not contain duplicates")

            expected_type = (
                f"{len(alignment.english_sentence_ids)}:"
                f"{len(alignment.chinese_sentence_ids)}"
            )
            if alignment.alignment_type != expected_type:
                raise ValueError(
                    f"Alignment {alignment.alignment_id} is labeled "
                    f"{alignment.alignment_type} but contains {expected_type} sentences"
                )

            english_evidence = [
                _validate_alignment_sentence(
                    connection,
                    sentence_id,
                    expected_language="EN",
                    expected_paragraph_no=alignment.paragraph_no,
                    expected_year=year,
                )
                for sentence_id in alignment.english_sentence_ids
            ]
            chinese_evidence = [
                _validate_alignment_sentence(
                    connection,
                    sentence_id,
                    expected_language="ZH",
                    expected_paragraph_no=alignment.paragraph_no,
                    expected_year=year,
                )
                for sentence_id in alignment.chinese_sentence_ids
            ]
            if len({document_id for _, document_id in english_evidence}) != 1:
                raise ValueError(
                    f"Alignment {alignment.alignment_id} mixes English documents"
                )
            if len({document_id for _, document_id in chinese_evidence}) != 1:
                raise ValueError(
                    f"Alignment {alignment.alignment_id} mixes Chinese documents"
                )
            expected_english_text = " ".join(text for text, _ in english_evidence)
            expected_chinese_text = " ".join(text for text, _ in chinese_evidence)
            if alignment.english_text != expected_english_text:
                raise ValueError(
                    f"Alignment {alignment.alignment_id} English text does not "
                    "match its linked sentences"
                )
            if alignment.chinese_text != expected_chinese_text:
                raise ValueError(
                    f"Alignment {alignment.alignment_id} Chinese text does not "
                    "match its linked sentences"
                )

            connection.execute(
                """
                INSERT INTO alignments (
                    alignment_id, year, paragraph_no, en_sentence_ids,
                    zh_sentence_ids, en_text, zh_text, alignment_type,
                    alignment_cost, alignment_confidence,
                    alignment_confidence_band, alignment_algorithm_version,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alignment.alignment_id,
                    year,
                    alignment.paragraph_no,
                    json.dumps(list(alignment.english_sentence_ids), ensure_ascii=False),
                    json.dumps(list(alignment.chinese_sentence_ids), ensure_ascii=False),
                    alignment.english_text,
                    alignment.chinese_text,
                    alignment.alignment_type,
                    alignment.raw_alignment_cost,
                    alignment.normalized_confidence,
                    alignment.confidence_band,
                    alignment.algorithm_version,
                    created_at_text,
                ),
            )

            for language, sentence_ids in (
                ("EN", alignment.english_sentence_ids),
                ("ZH", alignment.chinese_sentence_ids),
            ):
                connection.executemany(
                    """
                    INSERT INTO alignment_sentences (
                        alignment_id, sentence_id, language, sentence_position
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        (alignment.alignment_id, sentence_id, language, position)
                        for position, sentence_id in enumerate(sentence_ids, start=1)
                    ),
                )


def _alignment_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    result["en_sentence_ids"] = json.loads(result["en_sentence_ids"])
    result["zh_sentence_ids"] = json.loads(result["zh_sentence_ids"])
    result["possible_alignment_error"] = bool(result["possible_alignment_error"])
    return result


def get_alignment(
    connection: sqlite3.Connection,
    alignment_id: str,
) -> dict[str, Any] | None:
    """Retrieve one alignment with decoded sentence-ID lists."""
    row = connection.execute(
        "SELECT * FROM alignments WHERE alignment_id = ?", (alignment_id,)
    ).fetchone()
    return _alignment_dict(row)


def get_available_years(connection: sqlite3.Connection) -> tuple[int, ...]:
    """Return represented document years in ascending order."""
    rows = connection.execute(
        "SELECT DISTINCT year FROM documents ORDER BY year"
    ).fetchall()
    return tuple(row["year"] for row in rows)


def get_corpus_overview(
    connection: sqlite3.Connection,
    *,
    year: int | None = None,
) -> CorpusOverview:
    """Return basic corpus counts, optionally restricted to one year."""
    document_where = " WHERE year = ?" if year is not None else ""
    document_parameters: tuple[Any, ...] = (year,) if year is not None else ()
    document_count = connection.execute(
        f"SELECT COUNT(*) FROM documents{document_where}",
        document_parameters,
    ).fetchone()[0]
    years = tuple(
        row["year"]
        for row in connection.execute(
            f"SELECT DISTINCT year FROM documents{document_where} ORDER BY year",
            document_parameters,
        )
    )

    sentence_row = connection.execute(
        """
        SELECT
            SUM(CASE WHEN s.language = 'EN' THEN 1 ELSE 0 END) AS en_count,
            SUM(CASE WHEN s.language = 'ZH' THEN 1 ELSE 0 END) AS zh_count
        FROM sentences AS s
        JOIN documents AS d ON d.document_id = s.document_id
        WHERE (? IS NULL OR d.year = ?)
        """,
        (year, year),
    ).fetchone()
    alignment_row = connection.execute(
        """
        SELECT
            COUNT(*) AS total_count,
            SUM(CASE WHEN alignment_type = '1:1' THEN 1 ELSE 0 END) AS count_1_1,
            SUM(CASE WHEN alignment_type = '1:2' THEN 1 ELSE 0 END) AS count_1_2,
            SUM(CASE WHEN alignment_type = '2:1' THEN 1 ELSE 0 END) AS count_2_1,
            SUM(CASE WHEN alignment_type = '2:2' THEN 1 ELSE 0 END) AS count_2_2,
            SUM(CASE WHEN alignment_confidence_band = 'high' THEN 1 ELSE 0 END)
                AS high_count,
            SUM(CASE WHEN alignment_confidence_band = 'medium' THEN 1 ELSE 0 END)
                AS medium_count,
            SUM(CASE WHEN alignment_confidence_band = 'low' THEN 1 ELSE 0 END)
                AS low_count
        FROM alignments
        WHERE (? IS NULL OR year = ?)
        """,
        (year, year),
    ).fetchone()

    return CorpusOverview(
        document_count=document_count,
        years=years,
        english_sentence_count=sentence_row["en_count"] or 0,
        chinese_sentence_count=sentence_row["zh_count"] or 0,
        aligned_unit_count=alignment_row["total_count"],
        count_1_1=alignment_row["count_1_1"] or 0,
        count_1_2=alignment_row["count_1_2"] or 0,
        count_2_1=alignment_row["count_2_1"] or 0,
        count_2_2=alignment_row["count_2_2"] or 0,
        high_confidence_count=alignment_row["high_count"] or 0,
        medium_confidence_count=alignment_row["medium_count"] or 0,
        low_confidence_count=alignment_row["low_count"] or 0,
    )


_ALIGNMENT_TITLE_EXPRESSION = """
    COALESCE(
        (
            SELECT d.title
            FROM alignment_sentences AS als
            JOIN sentences AS s ON s.sentence_id = als.sentence_id
            JOIN documents AS d ON d.document_id = s.document_id
            WHERE als.alignment_id = al.alignment_id AND als.language = 'EN'
            ORDER BY als.sentence_position
            LIMIT 1
        ),
        (
            SELECT d.title
            FROM alignment_sentences AS als
            JOIN sentences AS s ON s.sentence_id = als.sentence_id
            JOIN documents AS d ON d.document_id = s.document_id
            WHERE als.alignment_id = al.alignment_id AND als.language = 'ZH'
            ORDER BY als.sentence_position
            LIMIT 1
        ),
        '[Title unavailable]'
    )
"""


def list_alignment_preview(
    connection: sqlite3.Connection,
    *,
    year: int | None = None,
    limit: int = 100,
) -> tuple[dict[str, Any], ...]:
    """Return compact alignment rows for the Corpus page preview."""
    if limit < 1:
        raise ValueError("limit must be positive")
    sql = f"""
        SELECT al.year, al.paragraph_no, al.alignment_id, al.alignment_type,
               al.alignment_confidence, al.alignment_confidence_band,
               al.en_text, al.zh_text,
               {_ALIGNMENT_TITLE_EXPRESSION} AS document_title
        FROM alignments AS al
        WHERE (? IS NULL OR al.year = ?)
        ORDER BY al.year, al.paragraph_no, al.alignment_id
        LIMIT ?
    """
    rows = connection.execute(sql, (year, year, limit)).fetchall()
    return tuple(dict(row) for row in rows)


def list_alignments_for_annotation(
    connection: sqlite3.Connection,
    annotator_id: str,
    *,
    year: int | None = None,
    minimum_confidence: float | None = 0.70,
    include_possible_alignment_errors: bool = False,
    alignment_type: str | None = None,
    confidence_band: str | None = None,
    annotation_status: str = "all",
    possible_alignment_error: bool | None = None,
    modality_label: str | None = None,
    stance_label: str | None = None,
    notes_search: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return the ordered annotation queue under explicit pilot filters."""
    if not annotator_id.strip():
        raise ValueError("annotator_id must not be empty")
    if minimum_confidence is not None and not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between 0 and 1")
    if alignment_type not in {None, "1:1", "1:2", "2:1", "2:2"}:
        raise ValueError(f"Unsupported alignment type: {alignment_type}")
    if confidence_band not in {None, "high", "medium", "low"}:
        raise ValueError(f"Unsupported confidence band: {confidence_band}")
    if annotation_status not in {"all", "annotated", "unannotated"}:
        raise ValueError(f"Unsupported annotation status: {annotation_status}")
    if modality_label is not None and modality_label not in {
        "N/A", "preserved", "strengthened", "weakened",
        "added", "omitted", "uncertain",
    }:
        raise ValueError(f"Unsupported modality label: {modality_label}")
    if stance_label is not None and stance_label not in {
        "N/A", "preserved", "strengthened", "weakened",
        "neutralized", "uncertain",
    }:
        raise ValueError(f"Unsupported stance label: {stance_label}")

    sql = f"""
        SELECT al.*, an.annotation_id, an.modality_label, an.stance_label,
               an.annotator_confidence, an.notes,
               an.possible_alignment_error AS annotation_possible_alignment_error,
               an.annotation_guideline_version, an.created_at AS annotation_created_at,
               an.updated_at AS annotation_updated_at,
               {_ALIGNMENT_TITLE_EXPRESSION} AS document_title
        FROM alignments AS al
        LEFT JOIN annotations AS an
          ON an.alignment_id = al.alignment_id
         AND an.annotator_id = ?
        WHERE (? IS NULL OR al.year = ?)
    """
    parameters: list[Any] = [annotator_id, year, year]
    if minimum_confidence is not None:
        sql += " AND al.alignment_confidence >= ?"
        parameters.append(minimum_confidence)
    if not include_possible_alignment_errors:
        sql += (
            " AND al.possible_alignment_error = 0"
            " AND COALESCE(an.possible_alignment_error, 0) = 0"
        )
    if alignment_type is not None:
        sql += " AND al.alignment_type = ?"
        parameters.append(alignment_type)
    if confidence_band is not None:
        sql += " AND al.alignment_confidence_band = ?"
        parameters.append(confidence_band)
    if annotation_status == "annotated":
        sql += " AND an.annotation_id IS NOT NULL"
    elif annotation_status == "unannotated":
        sql += " AND an.annotation_id IS NULL"
    if possible_alignment_error is not None:
        sql += (
            " AND (al.possible_alignment_error = 1 "
            "OR COALESCE(an.possible_alignment_error, 0) = 1) = ?"
        )
        parameters.append(int(possible_alignment_error))
    if modality_label is not None:
        sql += " AND an.modality_label = ?"
        parameters.append(modality_label)
    if stance_label is not None:
        sql += " AND an.stance_label = ?"
        parameters.append(stance_label)
    if notes_search and notes_search.strip():
        escaped_search = (
            notes_search.strip()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        sql += " AND an.notes LIKE ? ESCAPE '\\'"
        parameters.append(f"%{escaped_search}%")
    sql += " ORDER BY al.year, al.paragraph_no, al.alignment_id"

    records: list[dict[str, Any]] = []
    for row in connection.execute(sql, parameters):
        record = _alignment_dict(row)
        if record is not None:
            annotation_error = record["annotation_possible_alignment_error"]
            record["annotation_possible_alignment_error"] = (
                bool(annotation_error) if annotation_error is not None else None
            )
            records.append(record)
    return tuple(records)


def get_next_unannotated_alignment(
    connection: sqlite3.Connection,
    annotator_id: str,
    *,
    year: int | None = None,
    minimum_confidence: float | None = 0.70,
    include_possible_alignment_errors: bool = False,
) -> dict[str, Any] | None:
    """Return the next eligible alignment not annotated by this annotator.

    Pass ``minimum_confidence=None`` and
    ``include_possible_alignment_errors=True`` to include quarantined rows.
    """
    if minimum_confidence is not None and not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between 0 and 1")
    sql = """
        SELECT al.*
        FROM alignments AS al
        LEFT JOIN annotations AS an
          ON an.alignment_id = al.alignment_id
         AND an.annotator_id = ?
        WHERE an.annotation_id IS NULL
    """
    parameters: list[Any] = [annotator_id]
    if year is not None:
        sql += " AND al.year = ?"
        parameters.append(year)
    if minimum_confidence is not None:
        sql += " AND al.alignment_confidence >= ?"
        parameters.append(minimum_confidence)
    if not include_possible_alignment_errors:
        sql += " AND al.possible_alignment_error = 0"
    sql += " ORDER BY al.year, al.paragraph_no, al.alignment_id LIMIT 1"
    return _alignment_dict(connection.execute(sql, parameters).fetchone())


def set_alignment_error_flag(
    connection: sqlite3.Connection,
    alignment_id: str,
    *,
    possible_alignment_error: bool,
) -> None:
    """Explicitly mark or clear an alignment-level quality-control flag."""
    with connection:
        cursor = connection.execute(
            """
            UPDATE alignments
            SET possible_alignment_error = ?
            WHERE alignment_id = ?
            """,
            (int(possible_alignment_error), alignment_id),
        )
        if cursor.rowcount != 1:
            raise RecordNotFoundError(f"Unknown alignment_id: {alignment_id}")


def save_annotation(
    connection: sqlite3.Connection,
    *,
    annotation_id: str,
    alignment_id: str,
    annotator_id: str,
    modality_label: ModalityLabel,
    stance_label: StanceLabel,
    annotator_confidence: AnnotatorConfidence,
    notes: str = "",
    possible_alignment_error: bool = False,
    annotation_guideline_version: str,
    created_at: datetime | None = None,
) -> None:
    """Create a new annotation; never overwrite an existing annotation."""
    for field_name, value in (
        ("annotation_id", annotation_id),
        ("alignment_id", alignment_id),
        ("annotator_id", annotator_id),
        ("annotation_guideline_version", annotation_guideline_version),
    ):
        if not value.strip():
            raise ValueError(f"{field_name} must not be empty")
    validate_annotation_values(
        modality_label, stance_label, annotator_confidence
    )
    timestamp = _timestamp(created_at)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO annotations (
                    annotation_id, alignment_id, annotator_id, modality_label,
                    stance_label, annotator_confidence, notes,
                    possible_alignment_error, annotation_guideline_version,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation_id,
                    alignment_id,
                    annotator_id,
                    modality_label,
                    stance_label,
                    annotator_confidence,
                    notes,
                    int(possible_alignment_error),
                    annotation_guideline_version,
                    timestamp,
                    timestamp,
                ),
            )
    except sqlite3.IntegrityError as error:
        duplicate = connection.execute(
            """
            SELECT 1 FROM annotations
            WHERE alignment_id = ? AND annotator_id = ?
            """,
            (alignment_id, annotator_id),
        ).fetchone()
        if duplicate is not None:
            raise DuplicateAnnotationError(
                f"Annotator {annotator_id!r} already annotated {alignment_id!r}; "
                "use update_annotation() for an explicit update."
            ) from error
        raise


def update_annotation(
    connection: sqlite3.Connection,
    annotation_id: str,
    *,
    modality_label: ModalityLabel,
    stance_label: StanceLabel,
    annotator_confidence: AnnotatorConfidence,
    notes: str,
    possible_alignment_error: bool,
    annotation_guideline_version: str,
    updated_at: datetime | None = None,
) -> None:
    """Explicitly replace the editable fields of an existing annotation."""
    validate_annotation_values(
        modality_label, stance_label, annotator_confidence
    )
    with connection:
        cursor = connection.execute(
            """
            UPDATE annotations
            SET modality_label = ?, stance_label = ?, annotator_confidence = ?,
                notes = ?, possible_alignment_error = ?,
                annotation_guideline_version = ?, updated_at = ?
            WHERE annotation_id = ?
            """,
            (
                modality_label,
                stance_label,
                annotator_confidence,
                notes,
                int(possible_alignment_error),
                annotation_guideline_version,
                _timestamp(updated_at),
                annotation_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RecordNotFoundError(f"Unknown annotation_id: {annotation_id}")


def get_annotation(
    connection: sqlite3.Connection,
    *,
    alignment_id: str,
    annotator_id: str,
) -> dict[str, Any] | None:
    """Retrieve one annotator's annotation for one alignment."""
    row = connection.execute(
        """
        SELECT * FROM annotations
        WHERE alignment_id = ? AND annotator_id = ?
        """,
        (alignment_id, annotator_id),
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["possible_alignment_error"] = bool(result["possible_alignment_error"])
    return result


def get_annotation_progress(
    connection: sqlite3.Connection,
    annotator_id: str,
    *,
    year: int | None = None,
    minimum_confidence: float | None = 0.70,
    include_possible_alignment_errors: bool = False,
) -> AnnotationProgress:
    """Return progress within the same configurable queue eligibility rules."""
    if minimum_confidence is not None and not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between 0 and 1")

    conditions: list[str] = []
    alignment_parameters: list[Any] = []
    if year is not None:
        conditions.append("year = ?")
        alignment_parameters.append(year)
    if minimum_confidence is not None:
        conditions.append("alignment_confidence >= ?")
        alignment_parameters.append(minimum_confidence)
    if not include_possible_alignment_errors:
        conditions.append("possible_alignment_error = 0")
    alignment_where = " WHERE " + " AND ".join(conditions) if conditions else ""
    total = connection.execute(
        f"SELECT COUNT(*) FROM alignments{alignment_where}",
        alignment_parameters,
    ).fetchone()[0]

    annotated_sql = """
        SELECT COUNT(*)
        FROM annotations AS an
        JOIN alignments AS al ON al.alignment_id = an.alignment_id
        WHERE an.annotator_id = ?
    """
    annotated_parameters: list[Any] = [annotator_id]
    if year is not None:
        annotated_sql += " AND al.year = ?"
        annotated_parameters.append(year)
    if minimum_confidence is not None:
        annotated_sql += " AND al.alignment_confidence >= ?"
        annotated_parameters.append(minimum_confidence)
    if not include_possible_alignment_errors:
        annotated_sql += " AND al.possible_alignment_error = 0"
    annotated = connection.execute(
        annotated_sql, annotated_parameters
    ).fetchone()[0]
    percentage = 0.0 if total == 0 else round((annotated / total) * 100, 2)

    return AnnotationProgress(
        annotator_id=annotator_id,
        total_alignments=total,
        annotated_alignments=annotated,
        remaining_alignments=total - annotated,
        completion_percentage=percentage,
    )
