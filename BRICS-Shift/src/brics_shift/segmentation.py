"""Conservative deterministic English and Chinese sentence segmentation.

Segmentation is applied to one numbered paragraph at a time, so this module
cannot create a sentence that crosses an official paragraph boundary. The
English rules protect a small, documented abbreviation set, initials,
acronyms, decimals, and ellipses. They do not attempt linguistic parsing.
"""

from __future__ import annotations

import re
from typing import Literal


Language = Literal["EN", "ZH"]

SEGMENTATION_IMPLEMENTED = True
SEGMENTATION_VERSION = "brics-shift-segment-v0.1"

_ENGLISH_TERMINATORS = frozenset(".!?")
_CHINESE_TERMINATORS = frozenset("。！？!?")
_ENGLISH_CLOSERS = frozenset('"\'”’»)]}')
_CHINESE_CLOSERS = frozenset('"\'”’》」』】）)]}')

# Deliberately small and inspectable. These are protected only when followed
# by more text; a final remainder is always emitted as a sentence.
_ENGLISH_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "no",
        "nos",
        "art",
        "arts",
        "para",
        "paras",
        "fig",
        "figs",
        "p",
        "pp",
        "vs",
        "approx",
        "dept",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)
_CONTEXTUAL_ABBREVIATIONS = frozenset({"etc", "inc", "ltd"})
_WORD_BEFORE_PERIOD = re.compile(r"([A-Za-z]+)\.$")
_INITIAL_BEFORE_PERIOD = re.compile(r"\b[A-Z]\.$")
_ACRONYM_BEFORE_PERIOD = re.compile(r"(?:\b[A-Za-z]\.){2,}$")


def _next_non_whitespace(text: str, start: int) -> int | None:
    for index in range(start, len(text)):
        if not text[index].isspace():
            return index
    return None


def _english_period_is_boundary(text: str, index: int) -> bool:
    """Return whether one full stop is a conservative sentence boundary."""
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return False
    if following and not following.isspace() and following not in (
        _ENGLISH_TERMINATORS | _ENGLISH_CLOSERS
    ):
        return False
    if following == ".":
        return False

    prefix = text[: index + 1]
    if _ACRONYM_BEFORE_PERIOD.search(prefix):
        return False
    if _INITIAL_BEFORE_PERIOD.search(prefix):
        return False
    word_match = _WORD_BEFORE_PERIOD.search(prefix)
    if word_match and word_match.group(1).lower() in _ENGLISH_ABBREVIATIONS:
        return False
    if word_match and word_match.group(1).lower() in _CONTEXTUAL_ABBREVIATIONS:
        next_content = _next_non_whitespace(text, index + 1)
        if next_content is not None and text[next_content].islower():
            return False

    # e.g. and i.e. are caught by the acronym rule. ``etc.`` may legitimately
    # end a sentence, so it is not globally protected.
    return True


def _normalize_sentence_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _segment_with_rules(
    text: str,
    *,
    language: Language,
) -> tuple[str, ...]:
    terminators = (
        _ENGLISH_TERMINATORS if language == "EN" else _CHINESE_TERMINATORS
    )
    closers = _ENGLISH_CLOSERS if language == "EN" else _CHINESE_CLOSERS
    sentences: list[str] = []
    start = 0
    index = 0

    while index < len(text):
        character = text[index]
        is_boundary = character in terminators
        if language == "EN" and character == ".":
            is_boundary = _english_period_is_boundary(text, index)
        if not is_boundary:
            index += 1
            continue

        end = index + 1
        while end < len(text) and (
            text[end] in terminators or text[end] in closers
        ):
            end += 1
        next_content = _next_non_whitespace(text, end)
        if next_content is None:
            end = len(text)

        sentence = _normalize_sentence_whitespace(text[start:end])
        if sentence:
            sentences.append(sentence)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
        index = start

    remainder = _normalize_sentence_whitespace(text[start:])
    if remainder:
        sentences.append(remainder)
    return tuple(sentences)


def segment_paragraph(language: Language, text: str) -> tuple[str, ...]:
    """Segment one already-cleaned numbered paragraph deterministically.

    Empty input returns an empty tuple. No word, number, punctuation mark, or
    quotation mark is removed; only whitespace between source lines is folded
    to a single space inside a sentence.
    """
    if language not in {"EN", "ZH"}:
        raise ValueError("language must be 'EN' or 'ZH'")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return ()
    return _segment_with_rules(text, language=language)
