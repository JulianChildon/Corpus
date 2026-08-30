"""Deterministic detection of numbered diplomatic paragraphs.

Paragraph numbers are structural evidence. This module records them as found
and reports structural irregularities without renumbering or reordering text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Literal, Sequence, TextIO

from brics_shift.cleaning import clean_text, normalize_line_breaks


WarningCode = Literal[
    "no_numbered_paragraphs",
    "missing_paragraph_numbers",
    "duplicate_paragraph_number",
    "non_monotonic_paragraph_number",
    "large_numbering_gap",
]

# Supported markers are deliberately narrow: 1., 1．, 1、, 1), and (1).
# A marker must begin a line. Unspaced text is accepted (common in Chinese),
# but an unspaced digit is rejected so that values such as ``1.2`` are not
# mistaken for paragraph markers.
_NUMBERED_LINE = re.compile(
    r"^[ \t]*(?:\((?P<parenthesized>[1-9]\d*)\)|"
    r"(?P<plain>[1-9]\d*)[.．、)])"
    r"(?:[ \t\u00a0\u3000]+(?P<spaced_text>.*)|"
    r"(?P<unspaced_text>(?!\d).*))$"
)


@dataclass(frozen=True, slots=True)
class ParsedParagraph:
    """One numbered paragraph in its original document order."""

    document_id: str
    paragraph_no: int
    raw_paragraph_text: str
    cleaned_paragraph_text: str
    original_order: int


@dataclass(frozen=True, slots=True)
class ParagraphWarning:
    """A structural issue found without modifying the source structure."""

    code: WarningCode
    message: str
    paragraph_numbers: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ParagraphParseResult:
    """Numbered paragraphs, pre-paragraph heading text, and warnings."""

    document_id: str
    heading_text: str
    paragraphs: tuple[ParsedParagraph, ...]
    warnings: tuple[ParagraphWarning, ...]


@dataclass(frozen=True, slots=True)
class ParagraphStructureComparison:
    """Number-based comparison of English and Chinese document structures."""

    english_numbered_paragraphs: int
    chinese_numbered_paragraphs: int
    matching_paragraph_numbers: int
    missing_in_english: tuple[int, ...]
    missing_in_chinese: tuple[int, ...]


def _trim_blank_edge_lines(lines: Sequence[str]) -> str:
    """Join lines after removing only blank lines at the block edges."""
    start = 0
    end = len(lines)

    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1

    return "\n".join(lines[start:end])


def _structure_warnings(
    paragraph_numbers: Sequence[int],
    *,
    large_gap_threshold: int,
) -> tuple[ParagraphWarning, ...]:
    """Describe numbering anomalies while leaving the sequence unchanged."""
    if large_gap_threshold < 1:
        raise ValueError("large_gap_threshold must be at least 1")

    if not paragraph_numbers:
        return (
            ParagraphWarning(
                code="no_numbered_paragraphs",
                message="No numbered paragraphs were detected.",
            ),
        )

    warnings: list[ParagraphWarning] = []
    counts = Counter(paragraph_numbers)

    for paragraph_no in dict.fromkeys(paragraph_numbers):
        if counts[paragraph_no] > 1:
            warnings.append(
                ParagraphWarning(
                    code="duplicate_paragraph_number",
                    message=(
                        f"Paragraph number {paragraph_no} occurs "
                        f"{counts[paragraph_no]} times."
                    ),
                    paragraph_numbers=(paragraph_no,),
                )
            )

    observed = set(paragraph_numbers)
    missing = tuple(number for number in range(1, max(observed) + 1) if number not in observed)
    if missing:
        warnings.append(
            ParagraphWarning(
                code="missing_paragraph_numbers",
                message=f"Missing paragraph numbers: {list(missing)}.",
                paragraph_numbers=missing,
            )
        )

    for previous, current in zip(paragraph_numbers, paragraph_numbers[1:]):
        if current < previous:
            warnings.append(
                ParagraphWarning(
                    code="non_monotonic_paragraph_number",
                    message=(
                        f"Paragraph number {current} follows {previous}; "
                        "source order was preserved."
                    ),
                    paragraph_numbers=(previous, current),
                )
            )

        missing_between = current - previous - 1
        if missing_between >= large_gap_threshold:
            warnings.append(
                ParagraphWarning(
                    code="large_numbering_gap",
                    message=(
                        f"Large numbering gap from {previous} to {current} "
                        f"({missing_between} numbers missing)."
                    ),
                    paragraph_numbers=(previous, current),
                )
            )

    return tuple(warnings)


def parse_numbered_paragraphs(
    document_id: str,
    text: str,
    *,
    large_gap_threshold: int = 5,
) -> ParagraphParseResult:
    """Extract explicitly numbered paragraphs from *text*.

    Unnumbered text before the first marker is retained as ``heading_text``.
    After the first marker, unnumbered lines—including trailing lines—remain
    part of the current paragraph because there is no deterministic basis for
    treating them as a new paragraph.
    """
    if not document_id.strip():
        raise ValueError("document_id must not be empty")

    lines = normalize_line_breaks(text).split("\n")
    heading_lines: list[str] = []
    paragraph_blocks: list[tuple[int, list[str]]] = []
    current_number: int | None = None
    current_lines: list[str] = []

    for line in lines:
        match = _NUMBERED_LINE.fullmatch(line)
        if match:
            if current_number is not None:
                paragraph_blocks.append((current_number, current_lines))

            number_text = match.group("parenthesized") or match.group("plain")
            current_number = int(number_text)
            marker_text = match.group("spaced_text")
            if marker_text is None:
                marker_text = match.group("unspaced_text")
            current_lines = [marker_text]
        elif current_number is None:
            heading_lines.append(line)
        else:
            current_lines.append(line)

    if current_number is not None:
        paragraph_blocks.append((current_number, current_lines))

    paragraphs = tuple(
        ParsedParagraph(
            document_id=document_id,
            paragraph_no=paragraph_no,
            raw_paragraph_text=(raw_text := _trim_blank_edge_lines(block_lines)),
            cleaned_paragraph_text=clean_text(raw_text),
            original_order=original_order,
        )
        for original_order, (paragraph_no, block_lines) in enumerate(
            paragraph_blocks, start=1
        )
    )
    numbers = tuple(paragraph.paragraph_no for paragraph in paragraphs)

    return ParagraphParseResult(
        document_id=document_id,
        heading_text=_trim_blank_edge_lines(heading_lines),
        paragraphs=paragraphs,
        warnings=_structure_warnings(
            numbers,
            large_gap_threshold=large_gap_threshold,
        ),
    )


def compare_paragraph_structures(
    english: ParagraphParseResult,
    chinese: ParagraphParseResult,
) -> ParagraphStructureComparison:
    """Compare English and Chinese structures using paragraph numbers only."""
    english_numbers = {paragraph.paragraph_no for paragraph in english.paragraphs}
    chinese_numbers = {paragraph.paragraph_no for paragraph in chinese.paragraphs}

    return ParagraphStructureComparison(
        english_numbered_paragraphs=len(english.paragraphs),
        chinese_numbered_paragraphs=len(chinese.paragraphs),
        matching_paragraph_numbers=len(english_numbers & chinese_numbers),
        missing_in_english=tuple(sorted(chinese_numbers - english_numbers)),
        missing_in_chinese=tuple(sorted(english_numbers - chinese_numbers)),
    )


def format_structure_validation_report(
    comparison: ParagraphStructureComparison,
) -> str:
    """Format a concise, stable EN-ZH structure-validation report."""
    return "\n".join(
        (
            "Structure validation report",
            (
                "English numbered paragraphs: "
                f"{comparison.english_numbered_paragraphs}"
            ),
            (
                "Chinese numbered paragraphs: "
                f"{comparison.chinese_numbered_paragraphs}"
            ),
            f"Matching paragraph numbers: {comparison.matching_paragraph_numbers}",
            f"Missing in English: {list(comparison.missing_in_english)}",
            f"Missing in Chinese: {list(comparison.missing_in_chinese)}",
        )
    )


def print_structure_validation_report(
    comparison: ParagraphStructureComparison,
    *,
    file: TextIO | None = None,
) -> None:
    """Print the concise EN-ZH structure-validation report."""
    print(format_structure_validation_report(comparison), file=file)
