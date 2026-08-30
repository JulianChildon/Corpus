"""Allowed annotation values and non-linguistic input validation.

Substantive definitions belong in the researcher-authored, versioned annotation
guidelines. This module only enforces the permitted v0.1 value vocabulary.
"""

from __future__ import annotations

from typing import Literal


ModalityLabel = Literal[
    "N/A",
    "preserved",
    "strengthened",
    "weakened",
    "added",
    "omitted",
    "uncertain",
]
StanceLabel = Literal[
    "N/A",
    "preserved",
    "strengthened",
    "weakened",
    "neutralized",
    "uncertain",
]
AnnotatorConfidence = Literal["high", "medium", "low"]

MODALITY_LABELS: tuple[ModalityLabel, ...] = (
    "N/A",
    "preserved",
    "strengthened",
    "weakened",
    "added",
    "omitted",
    "uncertain",
)
STANCE_LABELS: tuple[StanceLabel, ...] = (
    "N/A",
    "preserved",
    "strengthened",
    "weakened",
    "neutralized",
    "uncertain",
)
ANNOTATOR_CONFIDENCE_VALUES: tuple[AnnotatorConfidence, ...] = (
    "high",
    "medium",
    "low",
)


def validate_annotation_values(
    modality_label: str,
    stance_label: str,
    annotator_confidence: str,
) -> None:
    """Validate allowed values without interpreting their linguistic meaning."""
    if modality_label not in MODALITY_LABELS:
        raise ValueError(f"Unsupported modality label: {modality_label}")
    if stance_label not in STANCE_LABELS:
        raise ValueError(f"Unsupported stance label: {stance_label}")
    if annotator_confidence not in ANNOTATOR_CONFIDENCE_VALUES:
        raise ValueError(
            f"Unsupported annotator confidence: {annotator_confidence}"
        )
