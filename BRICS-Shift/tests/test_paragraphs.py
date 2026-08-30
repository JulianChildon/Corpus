"""Tests for deterministic numbered-paragraph detection."""

from brics_shift.paragraphs import (
    compare_paragraph_structures,
    format_structure_validation_report,
    parse_numbered_paragraphs,
    print_structure_validation_report,
)


def warning_codes(result) -> set[str]:
    return {warning.code for warning in result.warnings}


def test_normal_numbered_paragraphs() -> None:
    text = (
        "1. We reaffirm our commitment to multilateralism.\n\n"
        "2. We welcome the progress achieved. We encourage further cooperation."
    )

    result = parse_numbered_paragraphs("2025_rio_en", text)

    assert [paragraph.paragraph_no for paragraph in result.paragraphs] == [1, 2]
    assert result.paragraphs[0].raw_paragraph_text == (
        "We reaffirm our commitment to multilateralism."
    )
    assert result.paragraphs[1].cleaned_paragraph_text == (
        "We welcome the progress achieved. We encourage further cooperation."
    )
    assert [paragraph.original_order for paragraph in result.paragraphs] == [1, 2]
    assert result.warnings == ()


def test_multiline_paragraph_is_not_split_or_lost() -> None:
    text = "1. First line.\nContinuation on a second line.\n\n2. Next paragraph."

    result = parse_numbered_paragraphs("doc_en", text)

    assert len(result.paragraphs) == 2
    assert result.paragraphs[0].raw_paragraph_text == (
        "First line.\nContinuation on a second line."
    )


def test_chinese_text_and_common_number_markers() -> None:
    text = "1．我们重申对多边主义的承诺。\n\n2、我们欢迎取得的进展。"

    result = parse_numbered_paragraphs("doc_zh", text)

    assert [paragraph.paragraph_no for paragraph in result.paragraphs] == [1, 2]
    assert result.paragraphs[0].cleaned_paragraph_text == (
        "我们重申对多边主义的承诺。"
    )
    assert result.paragraphs[1].cleaned_paragraph_text == "我们欢迎取得的进展。"


def test_missing_paragraph_number_produces_warning() -> None:
    result = parse_numbered_paragraphs("doc_en", "1. One.\n\n3. Three.")

    warning = next(
        warning
        for warning in result.warnings
        if warning.code == "missing_paragraph_numbers"
    )
    assert warning.paragraph_numbers == (2,)
    assert [paragraph.paragraph_no for paragraph in result.paragraphs] == [1, 3]


def test_duplicate_numbers_are_preserved_and_warned() -> None:
    result = parse_numbered_paragraphs("doc_en", "1. First.\n\n1. Duplicate.")

    assert [paragraph.paragraph_no for paragraph in result.paragraphs] == [1, 1]
    assert [paragraph.original_order for paragraph in result.paragraphs] == [1, 2]
    assert "duplicate_paragraph_number" in warning_codes(result)


def test_numbering_gap_produces_missing_and_large_gap_warnings() -> None:
    result = parse_numbered_paragraphs("doc_en", "1. One.\n\n10. Ten.")

    assert "missing_paragraph_numbers" in warning_codes(result)
    assert "large_numbering_gap" in warning_codes(result)
    assert [paragraph.paragraph_no for paragraph in result.paragraphs] == [1, 10]


def test_non_monotonic_numbers_are_not_reordered() -> None:
    result = parse_numbered_paragraphs("doc_en", "2. Two.\n\n1. One.")

    assert [paragraph.paragraph_no for paragraph in result.paragraphs] == [2, 1]
    assert "non_monotonic_paragraph_number" in warning_codes(result)


def test_heading_before_paragraph_one_is_kept_separately() -> None:
    text = "XVII BRICS SUMMIT\nRio de Janeiro Declaration\n\n1. First paragraph."

    result = parse_numbered_paragraphs("doc_en", text)

    assert result.heading_text == "XVII BRICS SUMMIT\nRio de Janeiro Declaration"
    assert len(result.paragraphs) == 1
    assert "XVII BRICS SUMMIT" not in result.paragraphs[0].raw_paragraph_text


def test_trailing_unnumbered_text_stays_with_last_paragraph() -> None:
    text = "1. Main paragraph.\nFinal unnumbered diplomatic note."

    result = parse_numbered_paragraphs("doc_en", text)

    assert result.paragraphs[0].raw_paragraph_text == (
        "Main paragraph.\nFinal unnumbered diplomatic note."
    )


def test_parenthesized_and_closing_parenthesis_markers_are_tolerated() -> None:
    result = parse_numbered_paragraphs("doc_en", "(1) One.\n\n2) Two.")

    assert [paragraph.paragraph_no for paragraph in result.paragraphs] == [1, 2]


def test_english_chinese_structure_comparison_and_report(capsys) -> None:
    english = parse_numbered_paragraphs(
        "doc_en", "1. One.\n\n2. Two.\n\n3. Three."
    )
    chinese = parse_numbered_paragraphs(
        "doc_zh", "1. 一。\n\n3. 三。\n\n4. 四。"
    )

    comparison = compare_paragraph_structures(english, chinese)

    assert comparison.english_numbered_paragraphs == 3
    assert comparison.chinese_numbered_paragraphs == 3
    assert comparison.matching_paragraph_numbers == 2
    assert comparison.missing_in_english == (4,)
    assert comparison.missing_in_chinese == (2,)

    report = format_structure_validation_report(comparison)
    assert "English numbered paragraphs: 3" in report
    assert "Chinese numbered paragraphs: 3" in report
    assert "Matching paragraph numbers: 2" in report
    assert "Missing in English: [4]" in report
    assert "Missing in Chinese: [2]" in report

    print_structure_validation_report(comparison)
    assert capsys.readouterr().out == report + "\n"
