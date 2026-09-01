"""Small tests for state-independent Streamlit UI helpers."""

from app import (
    LANGUAGE_EN,
    LANGUAGE_OPTIONS,
    LANGUAGE_ZH,
    SELECT_PROMPT,
    UI_TEXT,
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
        "modality_shift",
        "stance_shift",
        "annotator_confidence",
        "guideline_version",
    ]


def test_na_and_uncertain_are_deliberate_valid_ui_selections() -> None:
    assert _missing_annotation_fields("N/A", "uncertain", "low", "guidelines-v0.1") == []


def test_interface_language_catalog_includes_english_and_simplified_chinese() -> None:
    assert LANGUAGE_OPTIONS == ("简体中文", "English")
    assert UI_TEXT[LANGUAGE_EN]["page.annotation"] == "Annotation"
    assert UI_TEXT[LANGUAGE_ZH]["page.annotation"] == "人工标注"
    assert UI_TEXT[LANGUAGE_EN]["label.preserved"] == "preserved"
    assert UI_TEXT[LANGUAGE_ZH]["label.preserved"] == "保持"
