"""Conservative, deterministic document import and text cleaning.

The functions in this module normalize document structure without attempting
to correct, translate, tokenize, or interpret the source text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
import re
import unicodedata
from typing import Literal


Language = Literal["EN", "ZH"]

_SUPPORTED_LANGUAGES = frozenset({"EN", "ZH"})
_LINE_BREAKS = re.compile(r"\r\n|\r|\u0085|\u2028|\u2029")
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")


@dataclass(frozen=True, slots=True)
class ImportedDocument:
    """A cleaned declaration together with its source metadata."""

    document_id: str
    year: int
    title: str
    language: Language
    source_filename: str
    source_url: str | None
    imported_at: datetime
    cleaned_text: str

    def __post_init__(self) -> None:
        """Reject incomplete metadata rather than silently repairing it."""
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if isinstance(self.year, bool) or not isinstance(self.year, int) or self.year < 1:
            raise ValueError("year must be a positive integer")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if self.language not in _SUPPORTED_LANGUAGES:
            raise ValueError("language must be 'EN' or 'ZH'")
        if not self.source_filename.strip():
            raise ValueError("source_filename must not be empty")
        if self.imported_at.tzinfo is None or self.imported_at.utcoffset() is None:
            raise ValueError("imported_at must include timezone information")


def read_utf8_text(path: str | PathLike[str]) -> str:
    """Read *path* as strict UTF-8 without guessing or replacing characters.

    Invalid UTF-8 raises :class:`UnicodeDecodeError`, making source-data
    problems visible to the researcher.
    """
    return Path(path).read_text(encoding="utf-8", errors="strict")


def normalize_unicode(text: str) -> str:
    """Return canonically equivalent NFC text.

    NFC is intentionally used instead of NFKC so that compatibility
    characters, including full-width forms, are not rewritten.
    """
    return unicodedata.normalize("NFC", text)


def normalize_line_breaks(text: str) -> str:
    """Convert common platform and Unicode line endings to ``\n``."""
    return _LINE_BREAKS.sub("\n", text)


def normalize_whitespace(text: str) -> str:
    """Normalize horizontal whitespace while retaining every line boundary.

    Runs of spaces, tabs, and other horizontal whitespace become one ordinary
    space. Leading and trailing whitespace on each line is removed. No
    non-whitespace character is removed or changed.
    """
    normalized = normalize_line_breaks(text)
    lines = normalized.split("\n")
    return "\n".join(_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in lines)


def remove_repeated_blank_lines(text: str) -> str:
    """Collapse consecutive blank lines to one preserved paragraph break."""
    result: list[str] = []
    previous_was_blank = False

    for line in normalize_line_breaks(text).split("\n"):
        is_blank = not line.strip()
        if is_blank and previous_was_blank:
            continue
        result.append("" if is_blank else line)
        previous_was_blank = is_blank

    return "\n".join(result)


def clean_text(text: str) -> str:
    """Apply the complete conservative cleaning pipeline to *text*.

    Paragraph and line boundaries are retained in their original order;
    repeated blank lines are represented by one blank line (``\n\n``).
    """
    cleaned = normalize_unicode(text)
    cleaned = normalize_line_breaks(cleaned)
    cleaned = normalize_whitespace(cleaned)
    cleaned = remove_repeated_blank_lines(cleaned)
    return cleaned.strip()


def import_document(
    path: str | PathLike[str],
    *,
    document_id: str,
    year: int,
    title: str,
    language: Language,
    source_url: str | None = None,
    imported_at: datetime | None = None,
) -> ImportedDocument:
    """Read, clean, and describe one English or Chinese declaration file."""
    source_path = Path(path)
    raw_text = read_utf8_text(source_path)

    return ImportedDocument(
        document_id=document_id,
        year=year,
        title=title,
        language=language,
        source_filename=source_path.name,
        source_url=source_url,
        imported_at=imported_at or datetime.now(timezone.utc),
        cleaned_text=clean_text(raw_text),
    )
