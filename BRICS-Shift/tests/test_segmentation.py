"""Tests for conservative deterministic sentence segmentation."""

import pytest

from brics_shift.segmentation import (
    SEGMENTATION_IMPLEMENTED,
    SEGMENTATION_VERSION,
    segment_paragraph,
)


def test_segmentation_version_is_explicit() -> None:
    assert SEGMENTATION_IMPLEMENTED is True
    assert SEGMENTATION_VERSION == "brics-shift-segment-v0.1"


def test_normal_english_sentences() -> None:
    assert segment_paragraph(
        "EN", "We reaffirm cooperation. We welcome progress! Do partners agree?"
    ) == (
        "We reaffirm cooperation.",
        "We welcome progress!",
        "Do partners agree?",
    )


def test_english_abbreviations_do_not_create_false_breaks() -> None:
    text = (
        "We consulted representatives, e.g. from Brazil and India. "
        "Dr. Silva welcomed the U.N. initiative. We reaffirm cooperation."
    )

    assert segment_paragraph("EN", text) == (
        "We consulted representatives, e.g. from Brazil and India.",
        "Dr. Silva welcomed the U.N. initiative.",
        "We reaffirm cooperation.",
    )


def test_english_decimal_and_number_reference_are_preserved() -> None:
    assert segment_paragraph(
        "EN", "Growth reached 3.5 percent. See Art. 17 for details."
    ) == (
        "Growth reached 3.5 percent.",
        "See Art. 17 for details.",
    )


def test_contextual_abbreviation_does_not_split_before_lowercase_continuation() -> None:
    assert segment_paragraph(
        "EN", "We support trade, investment, etc. in all member states. Next item."
    ) == (
        "We support trade, investment, etc. in all member states.",
        "Next item.",
    )


def test_closing_quotation_mark_stays_with_sentence() -> None:
    assert segment_paragraph(
        "EN", 'We reaffirmed: “Cooperation matters.” We welcome progress.'
    ) == (
        'We reaffirmed: “Cooperation matters.”',
        "We welcome progress.",
    )


def test_chinese_sentence_boundaries_and_quotes() -> None:
    assert segment_paragraph(
        "ZH", "我们重申《2030年议程》。我们欢迎进展！是否继续合作？"
    ) == (
        "我们重申《2030年议程》。",
        "我们欢迎进展！",
        "是否继续合作？",
    )


def test_semicolons_do_not_force_a_sentence_break() -> None:
    assert segment_paragraph(
        "ZH", "我们重申承诺；并欢迎进一步合作。下一项议题。"
    ) == (
        "我们重申承诺；并欢迎进一步合作。",
        "下一项议题。",
    )


def test_line_breaks_are_folded_only_within_the_same_paragraph() -> None:
    assert segment_paragraph(
        "EN", "We reaffirm our commitment\nto multilateralism. Next sentence."
    ) == (
        "We reaffirm our commitment to multilateralism.",
        "Next sentence.",
    )


def test_empty_input_and_invalid_language() -> None:
    assert segment_paragraph("EN", " \n ") == ()
    with pytest.raises(ValueError, match="language"):
        segment_paragraph("FR", "Texte.")  # type: ignore[arg-type]


def test_repeatability() -> None:
    text = "We reaffirm cooperation. We welcome the 2030 Agenda."
    assert segment_paragraph("EN", text) == segment_paragraph("EN", text)


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("EN", 'We reaffirm “BRICS+” cooperation. The target is 2030.'),
        ("ZH", "我们重申“金砖+”合作。目标是2030年。"),
    ],
)
def test_segmentation_preserves_all_non_whitespace_characters(
    language: str, text: str
) -> None:
    segmented = " ".join(
        segment_paragraph(language, text)  # type: ignore[arg-type]
    )
    assert "".join(character for character in segmented if not character.isspace()) == (
        "".join(character for character in text if not character.isspace())
    )
