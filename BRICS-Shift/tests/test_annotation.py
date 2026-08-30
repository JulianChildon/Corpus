"""Tests for annotation value-vocabulary validation."""

import pytest

from brics_shift.annotation import (
    ANNOTATOR_CONFIDENCE_VALUES,
    MODALITY_LABELS,
    STANCE_LABELS,
    validate_annotation_values,
)


def test_allowed_annotation_values_match_guideline_template() -> None:
    assert MODALITY_LABELS == (
        "N/A",
        "preserved",
        "strengthened",
        "weakened",
        "added",
        "omitted",
        "uncertain",
    )
    assert STANCE_LABELS == (
        "N/A",
        "preserved",
        "strengthened",
        "weakened",
        "neutralized",
        "uncertain",
    )
    assert ANNOTATOR_CONFIDENCE_VALUES == ("high", "medium", "low")


def test_allowed_annotation_combination_passes_validation() -> None:
    validate_annotation_values("preserved", "neutralized", "medium")


@pytest.mark.parametrize(
    ("modality", "stance", "confidence"),
    [
        ("invented", "preserved", "high"),
        ("preserved", "invented", "high"),
        ("preserved", "preserved", "0.9"),
    ],
)
def test_unsupported_annotation_values_are_rejected(
    modality: str,
    stance: str,
    confidence: str,
) -> None:
    with pytest.raises(ValueError):
        validate_annotation_values(modality, stance, confidence)
