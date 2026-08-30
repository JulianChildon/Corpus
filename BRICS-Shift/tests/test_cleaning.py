"""Tests for conservative document import and text cleaning."""

from datetime import datetime, timezone
import unicodedata

import pytest

from brics_shift.cleaning import (
    clean_text,
    import_document,
    normalize_line_breaks,
    normalize_unicode,
    read_utf8_text,
)


def test_english_whitespace_normalization() -> None:
    raw = "  BRICS\tpartners   cooperate.  \n\n  Shared prosperity matters. "

    assert clean_text(raw) == (
        "BRICS partners cooperate.\n\nShared prosperity matters."
    )


def test_chinese_whitespace_normalization_preserves_punctuation_and_quotes() -> None:
    raw = "  金砖国家\u3000坚持\t‘开放、包容、合作、共赢’。  "

    assert clean_text(raw) == "金砖国家 坚持 ‘开放、包容、合作、共赢’。"


def test_unicode_normalization_uses_nfc_not_compatibility_normalization() -> None:
    raw = "Cafe\u0301 Ｇ２０"

    assert normalize_unicode(raw) == "Café Ｇ２０"


def test_repeated_blank_lines_become_one_paragraph_boundary() -> None:
    raw = "First paragraph.\n\n  \n\t\nSecond paragraph."

    assert clean_text(raw) == "First paragraph.\n\nSecond paragraph."


def test_paragraph_numbering_is_preserved() -> None:
    raw = "  1. First paragraph.\r\n\r\n  2. Second paragraph.  "

    assert clean_text(raw) == "1. First paragraph.\n\n2. Second paragraph."


def test_policy_numbers_and_alphanumeric_terms_are_preserved() -> None:
    raw = "By 2030, the G20 will review the 17th commitment."

    assert clean_text(raw) == raw


@pytest.mark.parametrize("raw", ["", " ", "\t\r\n\u3000"])
def test_empty_or_whitespace_only_input(raw: str) -> None:
    assert clean_text(raw) == ""


def test_malformed_whitespace_is_normalized_without_joining_paragraphs() -> None:
    raw = "\tAlpha\v\fBeta\r\n \u00a0 \rGamma\t\tDelta "

    assert clean_text(raw) == "Alpha Beta\n\nGamma Delta"


def test_line_break_normalization() -> None:
    raw = "A\r\nB\rC\u2028D\u2029E\u0085F"

    assert normalize_line_breaks(raw) == "A\nB\nC\nD\nE\nF"


def test_cleaning_preserves_every_non_whitespace_character_in_order() -> None:
    raw = (
        "  17. We reaffirm ‘shared but differentiated responsibilities’—"
        "including the 2030 Agenda.\n\n"
        "我们重申《2030年议程》；不推断作者意图。  "
    )

    cleaned = clean_text(raw)

    expected_characters = "".join(
        character
        for character in unicodedata.normalize("NFC", raw)
        if not character.isspace()
    )
    actual_characters = "".join(
        character for character in cleaned if not character.isspace()
    )
    assert actual_characters == expected_characters


def test_read_utf8_text_and_import_document(tmp_path) -> None:
    source = tmp_path / "2025_rio_zh.txt"
    source.write_text("  1. 合作\t共赢。  ", encoding="utf-8")
    imported_at = datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc)

    assert read_utf8_text(source) == "  1. 合作\t共赢。  "

    document = import_document(
        source,
        document_id="2025_rio_zh",
        year=2025,
        title="Rio Declaration",
        language="ZH",
        source_url="https://example.test/official",
        imported_at=imported_at,
    )

    assert document.document_id == "2025_rio_zh"
    assert document.year == 2025
    assert document.title == "Rio Declaration"
    assert document.language == "ZH"
    assert document.source_filename == "2025_rio_zh.txt"
    assert document.source_url == "https://example.test/official"
    assert document.imported_at == imported_at
    assert document.cleaned_text == "1. 合作 共赢。"


def test_import_rejects_unsupported_language(tmp_path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("Text", encoding="utf-8")

    with pytest.raises(ValueError, match="language"):
        import_document(
            source,
            document_id="document",
            year=2025,
            title="Declaration",
            language="FR",  # type: ignore[arg-type]
        )
