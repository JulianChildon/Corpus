"""Deterministic, paragraph-bounded English-Chinese sentence alignment.

Equal sentence counts use ordered 1:1 alignment. Unequal counts use dynamic
programming without semantic models, machine learning, or skip operations.

Alignment confidence is a heuristic quality score, not a calibrated
probability.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from statistics import median
from typing import Literal, Sequence, TextIO


ALGORITHM_VERSION = "brics-shift-align-v0.1"

Language = Literal["EN", "ZH"]
AlignmentType = Literal["1:1", "1:2", "2:1", "2:2"]
ConfidenceBand = Literal["high", "medium", "low"]

_ENGLISH_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*")
_ARABIC_NUMERAL = re.compile(r"\d+(?:[.,]\d+)*")
_SAFE_ALIGNMENT_KEY = re.compile(r"[^A-Za-z0-9_-]+")
_OPERATIONS: tuple[tuple[int, int, AlignmentType], ...] = (
    (1, 1, "1:1"),
    (1, 2, "1:2"),
    (2, 1, "2:1"),
    (2, 2, "2:2"),
)


@dataclass(frozen=True, slots=True)
class SentenceRecord:
    """One already-segmented sentence supplied to the aligner."""

    sentence_id: str
    document_id: str
    language: Language
    paragraph_no: int
    sentence_no: int
    text: str

    def __post_init__(self) -> None:
        if not self.sentence_id.strip():
            raise ValueError("sentence_id must not be empty")
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if self.language not in {"EN", "ZH"}:
            raise ValueError("language must be 'EN' or 'ZH'")
        if self.paragraph_no < 1:
            raise ValueError("paragraph_no must be positive")
        if self.sentence_no < 1:
            raise ValueError("sentence_no must be positive")
        if not self.text.strip():
            raise ValueError("sentence text must not be empty")


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    """All tunable assumptions used by the v0.1 alignment algorithm."""

    fallback_zh_chars_per_en_word: float = 1.6
    length_ratio_override: float | None = None
    estimate_length_ratio: bool = True
    minimum_ratio_samples: int = 3
    length_weight: float = 1.0
    numeral_weight: float = 0.6
    order_weight: float = 0.1
    penalty_1_1: float = 0.0
    penalty_1_2: float = 0.15
    penalty_2_1: float = 0.15
    penalty_2_2: float = 0.3
    confidence_cost_scale: float = 1.0
    high_confidence_threshold: float = 0.85
    medium_confidence_threshold: float = 0.70
    unusual_sentence_count_threshold: int = 8
    unusual_count_difference_threshold: int = 3

    def __post_init__(self) -> None:
        if self.fallback_zh_chars_per_en_word <= 0:
            raise ValueError("fallback length ratio must be positive")
        if self.length_ratio_override is not None and self.length_ratio_override <= 0:
            raise ValueError("length ratio override must be positive")
        if self.minimum_ratio_samples < 1:
            raise ValueError("minimum_ratio_samples must be positive")
        if min(self.length_weight, self.numeral_weight, self.order_weight) < 0:
            raise ValueError("feature weights must not be negative")
        if min(
            self.penalty_1_1,
            self.penalty_1_2,
            self.penalty_2_1,
            self.penalty_2_2,
        ) < 0:
            raise ValueError("operation penalties must not be negative")
        if self.confidence_cost_scale <= 0:
            raise ValueError("confidence_cost_scale must be positive")
        if not (
            0 <= self.medium_confidence_threshold
            <= self.high_confidence_threshold
            <= 1
        ):
            raise ValueError("confidence thresholds must satisfy 0 <= medium <= high <= 1")
        if self.unusual_sentence_count_threshold < 1:
            raise ValueError("unusual sentence count threshold must be positive")
        if self.unusual_count_difference_threshold < 1:
            raise ValueError("unusual count difference threshold must be positive")


@dataclass(frozen=True, slots=True)
class AlignmentUnit:
    """One transparent EN-ZH sentence alignment unit."""

    alignment_id: str
    english_sentence_ids: tuple[str, ...]
    chinese_sentence_ids: tuple[str, ...]
    english_text: str
    chinese_text: str
    paragraph_no: int
    alignment_type: AlignmentType
    raw_alignment_cost: float
    normalized_confidence: float
    confidence_band: ConfidenceBand
    algorithm_version: str = ALGORITHM_VERSION


@dataclass(frozen=True, slots=True)
class ParagraphAlignmentFailure:
    """A paragraph that could not be aligned under the allowed operations."""

    paragraph_no: int
    english_sentence_count: int
    chinese_sentence_count: int
    reason: str


@dataclass(frozen=True, slots=True)
class AlignmentDiagnostics:
    """Counts and paragraph-level review signals for one alignment run."""

    total_aligned_units: int
    count_1_1: int
    count_1_2: int
    count_2_1: int
    count_2_2: int
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    failed_paragraphs: tuple[ParagraphAlignmentFailure, ...]
    unusual_sentence_count_paragraphs: tuple[int, ...]
    low_confidence_paragraphs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Complete deterministic output for one bilingual document pair."""

    alignments: tuple[AlignmentUnit, ...]
    diagnostics: AlignmentDiagnostics
    expected_zh_chars_per_en_word: float
    algorithm_version: str = ALGORITHM_VERSION


def english_word_count(text: str) -> int:
    """Count transparent English-style word and number tokens."""
    return len(_ENGLISH_TOKEN.findall(text))


def chinese_character_count(text: str) -> int:
    """Count Chinese-side letters and numbers, excluding whitespace/punctuation."""
    return sum(character.isalnum() for character in text)


def extract_arabic_numerals(text: str) -> tuple[str, ...]:
    """Extract Arabic numeral strings in source order."""
    return tuple(_ARABIC_NUMERAL.findall(text))


def _group_by_paragraph(
    sentences: Sequence[SentenceRecord],
    *,
    expected_language: Language,
) -> dict[int, tuple[SentenceRecord, ...]]:
    """Group records without changing their supplied order."""
    grouped: dict[int, list[SentenceRecord]] = {}
    seen_ids: set[str] = set()
    last_sentence_no: dict[int, int] = {}
    document_id: str | None = None

    for sentence in sentences:
        if sentence.language != expected_language:
            raise ValueError(
                f"Expected {expected_language} sentence, got {sentence.language}: "
                f"{sentence.sentence_id}"
            )
        if sentence.sentence_id in seen_ids:
            raise ValueError(f"Duplicate sentence_id: {sentence.sentence_id}")
        if document_id is None:
            document_id = sentence.document_id
        elif sentence.document_id != document_id:
            raise ValueError(
                f"{expected_language} input mixes documents {document_id!r} and "
                f"{sentence.document_id!r}"
            )
        if sentence.sentence_no <= last_sentence_no.get(sentence.paragraph_no, 0):
            raise ValueError(
                "sentence_no values must increase within each paragraph in "
                "the supplied order"
            )

        seen_ids.add(sentence.sentence_id)
        last_sentence_no[sentence.paragraph_no] = sentence.sentence_no
        grouped.setdefault(sentence.paragraph_no, []).append(sentence)

    return {number: tuple(records) for number, records in grouped.items()}


def _estimated_length_ratio(
    english_by_paragraph: dict[int, tuple[SentenceRecord, ...]],
    chinese_by_paragraph: dict[int, tuple[SentenceRecord, ...]],
    config: AlignmentConfig,
) -> float:
    """Estimate the ratio from obvious ordered 1:1 pairs when sufficient."""
    if config.length_ratio_override is not None:
        return config.length_ratio_override

    samples: list[float] = []
    if config.estimate_length_ratio:
        for paragraph_no, english_sentences in english_by_paragraph.items():
            chinese_sentences = chinese_by_paragraph.get(paragraph_no)
            if not chinese_sentences or len(english_sentences) != len(chinese_sentences):
                continue

            for english, chinese in zip(english_sentences, chinese_sentences):
                english_length = english_word_count(english.text)
                chinese_length = chinese_character_count(chinese.text)
                if english_length and chinese_length:
                    samples.append(chinese_length / english_length)

    if len(samples) >= config.minimum_ratio_samples:
        return float(median(samples))
    return config.fallback_zh_chars_per_en_word


def _numeral_disagreement(english_text: str, chinese_text: str) -> float:
    """Return 0 for full numeric agreement and up to 1 for disagreement."""
    english_numbers = Counter(extract_arabic_numerals(english_text))
    chinese_numbers = Counter(extract_arabic_numerals(chinese_text))
    if not english_numbers and not chinese_numbers:
        return 0.0

    matches = sum((english_numbers & chinese_numbers).values())
    total = max(sum(english_numbers.values()), sum(chinese_numbers.values()))
    return 1.0 - (matches / total)


def _operation_penalty(alignment_type: AlignmentType, config: AlignmentConfig) -> float:
    return {
        "1:1": config.penalty_1_1,
        "1:2": config.penalty_1_2,
        "2:1": config.penalty_2_1,
        "2:2": config.penalty_2_2,
    }[alignment_type]


def _alignment_cost(
    english_sentences: Sequence[SentenceRecord],
    chinese_sentences: Sequence[SentenceRecord],
    *,
    english_start: int,
    chinese_start: int,
    english_total: int,
    chinese_total: int,
    length_ratio: float,
    alignment_type: AlignmentType,
    config: AlignmentConfig,
) -> float:
    """Calculate a transparent weighted cost for one candidate unit."""
    english_text = " ".join(sentence.text for sentence in english_sentences)
    chinese_text = " ".join(sentence.text for sentence in chinese_sentences)
    english_length = english_word_count(english_text)
    chinese_length = chinese_character_count(chinese_text)
    expected_chinese_length = english_length * length_ratio

    length_cost = abs(chinese_length - expected_chinese_length) / max(
        chinese_length, expected_chinese_length, 1.0
    )
    numeral_cost = _numeral_disagreement(english_text, chinese_text)
    english_midpoint = (
        english_start + (len(english_sentences) / 2)
    ) / english_total
    chinese_midpoint = (
        chinese_start + (len(chinese_sentences) / 2)
    ) / chinese_total
    order_cost = abs(english_midpoint - chinese_midpoint)

    return (
        config.length_weight * length_cost
        + config.numeral_weight * numeral_cost
        + config.order_weight * order_cost
        + _operation_penalty(alignment_type, config)
    )


def _confidence(raw_cost: float, config: AlignmentConfig) -> float:
    """Map cost deterministically to a heuristic 0-1 quality score."""
    return math.exp(-raw_cost / config.confidence_cost_scale)


def _confidence_band(score: float, config: AlignmentConfig) -> ConfidenceBand:
    if score >= config.high_confidence_threshold:
        return "high"
    if score >= config.medium_confidence_threshold:
        return "medium"
    return "low"


def _dynamic_programming_path(
    english_sentences: tuple[SentenceRecord, ...],
    chinese_sentences: tuple[SentenceRecord, ...],
    *,
    length_ratio: float,
    config: AlignmentConfig,
) -> tuple[tuple[int, int, int, int, AlignmentType, float], ...] | None:
    """Find the minimum-cost monotonic path using only v0.1 operations."""
    english_count = len(english_sentences)
    chinese_count = len(chinese_sentences)
    costs = [
        [math.inf for _ in range(chinese_count + 1)]
        for _ in range(english_count + 1)
    ]
    previous: list[list[tuple[int, int, AlignmentType, float] | None]] = [
        [None for _ in range(chinese_count + 1)]
        for _ in range(english_count + 1)
    ]
    costs[0][0] = 0.0

    for english_start in range(english_count + 1):
        for chinese_start in range(chinese_count + 1):
            if math.isinf(costs[english_start][chinese_start]):
                continue
            for english_size, chinese_size, alignment_type in _OPERATIONS:
                english_end = english_start + english_size
                chinese_end = chinese_start + chinese_size
                if english_end > english_count or chinese_end > chinese_count:
                    continue
                unit_cost = _alignment_cost(
                    english_sentences[english_start:english_end],
                    chinese_sentences[chinese_start:chinese_end],
                    english_start=english_start,
                    chinese_start=chinese_start,
                    english_total=english_count,
                    chinese_total=chinese_count,
                    length_ratio=length_ratio,
                    alignment_type=alignment_type,
                    config=config,
                )
                candidate_cost = costs[english_start][chinese_start] + unit_cost
                if candidate_cost < costs[english_end][chinese_end] - 1e-12:
                    costs[english_end][chinese_end] = candidate_cost
                    previous[english_end][chinese_end] = (
                        english_start,
                        chinese_start,
                        alignment_type,
                        unit_cost,
                    )

    if math.isinf(costs[english_count][chinese_count]):
        return None

    reversed_path: list[tuple[int, int, int, int, AlignmentType, float]] = []
    english_end = english_count
    chinese_end = chinese_count
    while english_end or chinese_end:
        step = previous[english_end][chinese_end]
        if step is None:
            return None
        english_start, chinese_start, alignment_type, unit_cost = step
        reversed_path.append(
            (
                english_start,
                english_end,
                chinese_start,
                chinese_end,
                alignment_type,
                unit_cost,
            )
        )
        english_end = english_start
        chinese_end = chinese_start

    return tuple(reversed(reversed_path))


def _ordered_one_to_one_path(
    english_sentences: tuple[SentenceRecord, ...],
    chinese_sentences: tuple[SentenceRecord, ...],
    *,
    length_ratio: float,
    config: AlignmentConfig,
) -> tuple[tuple[int, int, int, int, AlignmentType, float], ...]:
    """Create the required ordered 1:1 path for equal sentence counts."""
    count = len(english_sentences)
    return tuple(
        (
            index,
            index + 1,
            index,
            index + 1,
            "1:1",
            _alignment_cost(
                english_sentences[index : index + 1],
                chinese_sentences[index : index + 1],
                english_start=index,
                chinese_start=index,
                english_total=count,
                chinese_total=count,
                length_ratio=length_ratio,
                alignment_type="1:1",
                config=config,
            ),
        )
        for index in range(count)
    )


def _safe_alignment_key(alignment_key: str) -> str:
    key = _SAFE_ALIGNMENT_KEY.sub("_", alignment_key.strip()).strip("_")
    if not key:
        raise ValueError("alignment_key must contain a letter or number")
    return key


def align_documents(
    english_sentences: Sequence[SentenceRecord],
    chinese_sentences: Sequence[SentenceRecord],
    *,
    alignment_key: str,
    config: AlignmentConfig | None = None,
) -> AlignmentResult:
    """Align sentences hierarchically by official paragraph number."""
    active_config = config or AlignmentConfig()
    safe_key = _safe_alignment_key(alignment_key)
    english_by_paragraph = _group_by_paragraph(
        english_sentences, expected_language="EN"
    )
    chinese_by_paragraph = _group_by_paragraph(
        chinese_sentences, expected_language="ZH"
    )
    length_ratio = _estimated_length_ratio(
        english_by_paragraph,
        chinese_by_paragraph,
        active_config,
    )

    alignments: list[AlignmentUnit] = []
    failures: list[ParagraphAlignmentFailure] = []
    unusual_paragraphs: list[int] = []
    low_confidence_paragraphs: list[int] = []
    paragraph_order = list(english_by_paragraph)
    paragraph_order.extend(
        number for number in chinese_by_paragraph if number not in english_by_paragraph
    )

    for paragraph_no in paragraph_order:
        english_paragraph = english_by_paragraph.get(paragraph_no, ())
        chinese_paragraph = chinese_by_paragraph.get(paragraph_no, ())
        english_count = len(english_paragraph)
        chinese_count = len(chinese_paragraph)

        if (
            max(english_count, chinese_count)
            >= active_config.unusual_sentence_count_threshold
            or abs(english_count - chinese_count)
            >= active_config.unusual_count_difference_threshold
        ):
            unusual_paragraphs.append(paragraph_no)

        if not english_paragraph or not chinese_paragraph:
            missing_language = "English" if not english_paragraph else "Chinese"
            failures.append(
                ParagraphAlignmentFailure(
                    paragraph_no,
                    english_count,
                    chinese_count,
                    f"Paragraph is missing from {missing_language} input.",
                )
            )
            continue

        if english_count == chinese_count:
            path = _ordered_one_to_one_path(
                english_paragraph,
                chinese_paragraph,
                length_ratio=length_ratio,
                config=active_config,
            )
        else:
            path = _dynamic_programming_path(
                english_paragraph,
                chinese_paragraph,
                length_ratio=length_ratio,
                config=active_config,
            )

        if path is None:
            failures.append(
                ParagraphAlignmentFailure(
                    paragraph_no,
                    english_count,
                    chinese_count,
                    "No complete path exists using only 1:1, 1:2, 2:1, and "
                    "2:2 operations; skipping is disabled.",
                )
            )
            continue

        paragraph_has_low_confidence = False
        for unit_number, step in enumerate(path, start=1):
            (
                english_start,
                english_end,
                chinese_start,
                chinese_end,
                alignment_type,
                raw_cost,
            ) = step
            english_unit = english_paragraph[english_start:english_end]
            chinese_unit = chinese_paragraph[chinese_start:chinese_end]
            confidence = _confidence(raw_cost, active_config)
            band = _confidence_band(confidence, active_config)
            paragraph_has_low_confidence |= band == "low"
            alignments.append(
                AlignmentUnit(
                    alignment_id=f"A_{safe_key}_P{paragraph_no:03d}_{unit_number:02d}",
                    english_sentence_ids=tuple(s.sentence_id for s in english_unit),
                    chinese_sentence_ids=tuple(s.sentence_id for s in chinese_unit),
                    english_text=" ".join(s.text for s in english_unit),
                    chinese_text=" ".join(s.text for s in chinese_unit),
                    paragraph_no=paragraph_no,
                    alignment_type=alignment_type,
                    raw_alignment_cost=round(raw_cost, 6),
                    normalized_confidence=round(confidence, 6),
                    confidence_band=band,
                )
            )
        if paragraph_has_low_confidence:
            low_confidence_paragraphs.append(paragraph_no)

    type_counts = Counter(unit.alignment_type for unit in alignments)
    confidence_counts = Counter(unit.confidence_band for unit in alignments)
    diagnostics = AlignmentDiagnostics(
        total_aligned_units=len(alignments),
        count_1_1=type_counts["1:1"],
        count_1_2=type_counts["1:2"],
        count_2_1=type_counts["2:1"],
        count_2_2=type_counts["2:2"],
        high_confidence_count=confidence_counts["high"],
        medium_confidence_count=confidence_counts["medium"],
        low_confidence_count=confidence_counts["low"],
        failed_paragraphs=tuple(failures),
        unusual_sentence_count_paragraphs=tuple(unusual_paragraphs),
        low_confidence_paragraphs=tuple(low_confidence_paragraphs),
    )
    return AlignmentResult(
        alignments=tuple(alignments),
        diagnostics=diagnostics,
        expected_zh_chars_per_en_word=round(length_ratio, 6),
    )


def format_alignment_diagnostics(diagnostics: AlignmentDiagnostics) -> str:
    """Format the required concise diagnostics report."""
    failed_numbers = [failure.paragraph_no for failure in diagnostics.failed_paragraphs]
    return "\n".join(
        (
            "Alignment diagnostics",
            f"Total aligned units: {diagnostics.total_aligned_units}",
            f"1:1 count: {diagnostics.count_1_1}",
            f"1:2 count: {diagnostics.count_1_2}",
            f"2:1 count: {diagnostics.count_2_1}",
            f"2:2 count: {diagnostics.count_2_2}",
            f"High-confidence count: {diagnostics.high_confidence_count}",
            f"Medium-confidence count: {diagnostics.medium_confidence_count}",
            f"Low-confidence count: {diagnostics.low_confidence_count}",
            f"Failed paragraphs: {failed_numbers}",
            "Unusual sentence-count paragraphs: "
            f"{list(diagnostics.unusual_sentence_count_paragraphs)}",
            f"Low-confidence paragraphs: {list(diagnostics.low_confidence_paragraphs)}",
        )
    )


def print_alignment_diagnostics(
    diagnostics: AlignmentDiagnostics,
    *,
    file: TextIO | None = None,
) -> None:
    """Print the concise alignment diagnostics report."""
    print(format_alignment_diagnostics(diagnostics), file=file)
