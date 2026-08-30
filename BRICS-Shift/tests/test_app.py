"""Small tests for state-independent Streamlit UI helpers."""

from app import (
    SELECT_PROMPT,
    _annotation_id,
    _missing_annotation_fields,
    _next_unannotated_id,
    _select_index,
)


def test_annotation_id_is_stable_and_annotator_specific() -> None:
    first = _annotation_id("A_2025_P001_01", "researcher_1")

    assert first == _annotation_id("A_2025_P001_01", "researcher_1")
    assert first != _annotation_id("A_2025_P001_01", "researcher_2")
    assert first.startswith("ANN_A_2025_P001_01_")


def test_next_unannotated_alignment_wraps_without_selecting_current() -> None:
    queue = (
        {"alignment_id": "A1", "annotation_id": None},
        {"alignment_id": "A2", "annotation_id": "ANN_A2"},
        {"alignment_id": "A3", "annotation_id": None},
    )

    assert _next_unannotated_id(queue, 0) == "A3"
    assert _next_unannotated_id(queue, 2) == "A1"


def test_annotation_form_starts_unselected_and_blocks_implicit_labels() -> None:
    options = (SELECT_PROMPT, "N/A", "preserved", "uncertain")

    assert _select_index(options, None) == 0
    assert _missing_annotation_fields(
        SELECT_PROMPT, SELECT_PROMPT, SELECT_PROMPT, ""
    ) == [
        "Modality Shift",
        "Stance Shift",
        "Annotator Confidence",
        "Guideline version",
    ]


def test_na_and_uncertain_are_deliberate_valid_ui_selections() -> None:
    assert _missing_annotation_fields("N/A", "uncertain", "low", "guidelines-v0.1") == []
