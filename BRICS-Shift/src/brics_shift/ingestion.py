"""Preview-first import of one official English-Chinese declaration pair."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sqlite3

from brics_shift.alignment import (
    AlignmentConfig,
    AlignmentResult,
    SentenceRecord,
    align_documents,
)
from brics_shift.cleaning import ImportedDocument, clean_text
from brics_shift.database import (
    insert_alignments,
    insert_document,
    insert_paragraphs,
    insert_sentences,
)
from brics_shift.paragraphs import (
    ParagraphParseResult,
    ParagraphStructureComparison,
    compare_paragraph_structures,
    parse_numbered_paragraphs,
)
from brics_shift.segmentation import SEGMENTATION_VERSION, segment_paragraph


PREPROCESSING_VERSION = "brics-shift-clean-v0.1"
_SAFE_PAIR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    """One decoded document and its deterministic derived structure."""

    document: ImportedDocument
    raw_bytes: bytes
    checksum: str
    paragraph_result: ParagraphParseResult
    sentences: tuple[SentenceRecord, ...]


@dataclass(frozen=True, slots=True)
class PreparedCorpusPair:
    """A previewable import that has not yet changed persistent state."""

    pair_id: str
    year: int
    title: str
    english: PreparedDocument
    chinese: PreparedDocument
    structure_comparison: ParagraphStructureComparison
    alignment_result: AlignmentResult
    prepared_at: datetime

    @property
    def sentence_count(self) -> int:
        return len(self.english.sentences) + len(self.chinese.sentences)


@dataclass(frozen=True, slots=True)
class ImportSummary:
    documents: int
    paragraphs: int
    sentences: int
    alignments: int
    raw_files_saved: tuple[str, ...]


def _decode_uploaded_text(raw_bytes: bytes, filename: str) -> str:
    if not raw_bytes:
        raise ValueError(f"{filename} is empty")
    # UTF-8 BOM is transport metadata rather than declaration content.
    encoding = "utf-8-sig" if raw_bytes.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        return raw_bytes.decode(encoding, errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{filename} is not valid UTF-8: {error}") from error


def _source_filename(filename: str) -> str:
    safe_name = Path(filename).name
    if not safe_name or safe_name.lower().endswith(".txt") is False:
        raise ValueError("uploaded source filenames must end with .txt")
    return safe_name


def _build_sentences(
    pair_id: str,
    document_id: str,
    language: str,
    paragraph_result: ParagraphParseResult,
) -> tuple[SentenceRecord, ...]:
    sentences: list[SentenceRecord] = []
    for paragraph in paragraph_result.paragraphs:
        sentence_texts = segment_paragraph(
            language, paragraph.cleaned_paragraph_text  # type: ignore[arg-type]
        )
        if not sentence_texts:
            raise ValueError(
                f"{language} paragraph {paragraph.paragraph_no} has no sentence text"
            )
        for sentence_no, sentence_text in enumerate(sentence_texts, start=1):
            sentences.append(
                SentenceRecord(
                    sentence_id=(
                        f"{language}_{pair_id}_P{paragraph.paragraph_no:03d}_"
                        f"S{sentence_no:02d}"
                    ),
                    document_id=document_id,
                    language=language,  # type: ignore[arg-type]
                    paragraph_no=paragraph.paragraph_no,
                    sentence_no=sentence_no,
                    text=sentence_text,
                )
            )
    return tuple(sentences)


def _prepare_document(
    *,
    pair_id: str,
    raw_bytes: bytes,
    filename: str,
    year: int,
    title: str,
    language: str,
    source_url: str | None,
    imported_at: datetime,
) -> PreparedDocument:
    source_filename = _source_filename(filename)
    raw_text = _decode_uploaded_text(raw_bytes, source_filename)
    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        raise ValueError(f"{source_filename} contains no text after cleaning")
    document_id = f"{pair_id}_{language.lower()}"
    document = ImportedDocument(
        document_id=document_id,
        year=year,
        title=title,
        language=language,  # type: ignore[arg-type]
        source_filename=source_filename,
        source_url=source_url.strip() if source_url and source_url.strip() else None,
        imported_at=imported_at,
        cleaned_text=cleaned_text,
    )
    paragraph_result = parse_numbered_paragraphs(document_id, cleaned_text)
    if not paragraph_result.paragraphs:
        raise ValueError(
            f"No numbered paragraphs were detected in {source_filename}"
        )
    paragraph_numbers = [
        paragraph.paragraph_no for paragraph in paragraph_result.paragraphs
    ]
    if len(paragraph_numbers) != len(set(paragraph_numbers)):
        raise ValueError(
            f"Duplicate paragraph numbers in {source_filename}; review the source "
            "before import"
        )
    sentences = _build_sentences(
        pair_id, document_id, language, paragraph_result
    )
    return PreparedDocument(
        document=document,
        raw_bytes=raw_bytes,
        checksum=hashlib.sha256(raw_bytes).hexdigest(),
        paragraph_result=paragraph_result,
        sentences=sentences,
    )


def prepare_uploaded_pair(
    *,
    pair_id: str,
    year: int,
    title: str,
    english_bytes: bytes,
    english_filename: str,
    chinese_bytes: bytes,
    chinese_filename: str,
    english_source_url: str | None = None,
    chinese_source_url: str | None = None,
    prepared_at: datetime | None = None,
    alignment_config: AlignmentConfig | None = None,
) -> PreparedCorpusPair:
    """Decode, clean, parse, segment, and align without writing anything."""
    normalized_pair_id = pair_id.strip()
    if not _SAFE_PAIR_ID.fullmatch(normalized_pair_id):
        raise ValueError(
            "pair_id must start with a letter or number and contain only "
            "letters, numbers, underscores, or hyphens"
        )
    if isinstance(year, bool) or not isinstance(year, int) or year < 1:
        raise ValueError("year must be a positive integer")
    if not title.strip():
        raise ValueError("title must not be empty")
    if _source_filename(english_filename) == _source_filename(chinese_filename):
        raise ValueError("English and Chinese source filenames must be different")
    timestamp = prepared_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("prepared_at must include timezone information")
    timestamp = timestamp.astimezone(timezone.utc)

    english = _prepare_document(
        pair_id=normalized_pair_id,
        raw_bytes=english_bytes,
        filename=english_filename,
        year=year,
        title=title.strip(),
        language="EN",
        source_url=english_source_url,
        imported_at=timestamp,
    )
    chinese = _prepare_document(
        pair_id=normalized_pair_id,
        raw_bytes=chinese_bytes,
        filename=chinese_filename,
        year=year,
        title=title.strip(),
        language="ZH",
        source_url=chinese_source_url,
        imported_at=timestamp,
    )
    comparison = compare_paragraph_structures(
        english.paragraph_result, chinese.paragraph_result
    )
    alignment_result = align_documents(
        english.sentences,
        chinese.sentences,
        alignment_key=normalized_pair_id,
        config=alignment_config,
    )
    return PreparedCorpusPair(
        pair_id=normalized_pair_id,
        year=year,
        title=title.strip(),
        english=english,
        chinese=chinese,
        structure_comparison=comparison,
        alignment_result=alignment_result,
        prepared_at=timestamp,
    )


def _preflight_database(
    connection: sqlite3.Connection, prepared: PreparedCorpusPair
) -> None:
    document_ids = (
        prepared.english.document.document_id,
        prepared.chinese.document.document_id,
    )
    for document_id in document_ids:
        if connection.execute(
            "SELECT 1 FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone():
            raise ValueError(
                f"Document {document_id!r} already exists; no data were overwritten"
            )


def _save_raw_files(
    prepared: PreparedCorpusPair, raw_directory: Path
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    raw_directory.mkdir(parents=True, exist_ok=True)
    items = (prepared.english, prepared.chinese)
    destinations = [
        raw_directory / item.document.source_filename for item in items
    ]
    for destination, item in zip(destinations, items, strict=True):
        if destination.exists() and destination.read_bytes() != item.raw_bytes:
            raise FileExistsError(
                f"Raw file {destination.name!r} already exists with different content"
            )

    created: list[Path] = []
    try:
        for destination, item in zip(destinations, items, strict=True):
            if not destination.exists():
                with destination.open("xb") as output:
                    output.write(item.raw_bytes)
                created.append(destination)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return tuple(destinations), tuple(created)


def persist_prepared_pair(
    connection: sqlite3.Connection,
    prepared: PreparedCorpusPair,
    *,
    raw_directory: str | Path | None = None,
) -> ImportSummary:
    """Persist one preview atomically, with no implicit overwrite."""
    _preflight_database(connection, prepared)
    if not prepared.alignment_result.alignments:
        raise ValueError("No alignment units were produced; import was not saved")

    raw_paths: tuple[Path, ...] = ()
    created_paths: tuple[Path, ...] = ()
    try:
        if raw_directory is not None:
            raw_paths, created_paths = _save_raw_files(
                prepared, Path(raw_directory)
            )
        with connection:
            for item in (prepared.english, prepared.chinese):
                insert_document(
                    connection,
                    item.document,
                    checksum=item.checksum,
                    preprocessing_version=PREPROCESSING_VERSION,
                    commit=False,
                )
                insert_paragraphs(
                    connection, item.paragraph_result.paragraphs, commit=False
                )
            all_sentences = (
                *prepared.english.sentences,
                *prepared.chinese.sentences,
            )
            insert_sentences(
                connection,
                all_sentences,
                segmentation_version=SEGMENTATION_VERSION,
                commit=False,
            )
            insert_alignments(
                connection,
                prepared.alignment_result.alignments,
                year=prepared.year,
                created_at=prepared.prepared_at,
                commit=False,
            )
    except Exception:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise

    return ImportSummary(
        documents=2,
        paragraphs=(
            len(prepared.english.paragraph_result.paragraphs)
            + len(prepared.chinese.paragraph_result.paragraphs)
        ),
        sentences=prepared.sentence_count,
        alignments=len(prepared.alignment_result.alignments),
        raw_files_saved=tuple(str(path) for path in raw_paths),
    )
