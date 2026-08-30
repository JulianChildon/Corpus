"""Synthetic tests for deterministic paragraph-bounded alignment."""

from brics_shift.alignment import (
    ALGORITHM_VERSION,
    AlignmentConfig,
    SentenceRecord,
    align_documents,
    format_alignment_diagnostics,
)


def sentence(language: str, paragraph_no: int, sentence_no: int, text: str) -> SentenceRecord:
    return SentenceRecord(
        sentence_id=f"{language}_2025_P{paragraph_no:03d}_S{sentence_no:02d}",
        document_id=f"2025_rio_{language.lower()}",
        language=language,  # type: ignore[arg-type]
        paragraph_no=paragraph_no,
        sentence_no=sentence_no,
        text=text,
    )


def fixed_ratio_config(**changes) -> AlignmentConfig:
    values = {"length_ratio_override": 1.6}
    values.update(changes)
    return AlignmentConfig(**values)


def test_one_english_to_one_chinese() -> None:
    result = align_documents(
        [sentence("EN", 23, 1, "We reaffirm cooperation.")],
        [sentence("ZH", 23, 1, "我们重申合作。")],
        alignment_key="2025",
        config=fixed_ratio_config(),
    )
    unit = result.alignments[0]
    assert unit.alignment_id == "A_2025_P023_01"
    assert unit.alignment_type == "1:1"
    assert unit.english_sentence_ids == ("EN_2025_P023_S01",)
    assert unit.chinese_sentence_ids == ("ZH_2025_P023_S01",)
    assert unit.algorithm_version == ALGORITHM_VERSION


def test_two_english_to_two_chinese_defaults_to_ordered_one_to_one() -> None:
    result = align_documents(
        [sentence("EN", 1, 1, "First statement."), sentence("EN", 1, 2, "Second statement.")],
        [sentence("ZH", 1, 1, "第一项声明。"), sentence("ZH", 1, 2, "第二项声明。")],
        alignment_key="equal",
        config=fixed_ratio_config(),
    )
    assert [unit.alignment_type for unit in result.alignments] == ["1:1", "1:1"]
    assert result.alignments[0].chinese_sentence_ids == ("ZH_2025_P001_S01",)
    assert result.alignments[1].chinese_sentence_ids == ("ZH_2025_P001_S02",)


def test_two_english_to_one_chinese() -> None:
    result = align_documents(
        [sentence("EN", 1, 1, "We reaffirm cooperation."), sentence("EN", 1, 2, "We welcome progress.")],
        [sentence("ZH", 1, 1, "我们重申合作并欢迎取得进展。")],
        alignment_key="two_to_one",
        config=fixed_ratio_config(),
    )
    unit = result.alignments[0]
    assert unit.alignment_type == "2:1"
    assert len(unit.english_sentence_ids) == 2
    assert len(unit.chinese_sentence_ids) == 1
    assert result.diagnostics.count_2_1 == 1


def test_one_english_to_two_chinese() -> None:
    result = align_documents(
        [sentence("EN", 1, 1, "We reaffirm cooperation and welcome progress.")],
        [sentence("ZH", 1, 1, "我们重申合作。"), sentence("ZH", 1, 2, "我们欢迎取得进展。")],
        alignment_key="one_to_two",
        config=fixed_ratio_config(),
    )
    unit = result.alignments[0]
    assert unit.alignment_type == "1:2"
    assert len(unit.english_sentence_ids) == 1
    assert len(unit.chinese_sentence_ids) == 2
    assert result.diagnostics.count_1_2 == 1


def test_matching_2030_numeral_reduces_cost() -> None:
    english = [sentence("EN", 1, 1, "We reaffirm implementation of the 2030 Agenda.")]
    matching = align_documents(
        english,
        [sentence("ZH", 1, 1, "我们重申落实《2030年议程》。")],
        alignment_key="matching_number",
        config=fixed_ratio_config(),
    )
    mismatching = align_documents(
        english,
        [sentence("ZH", 1, 1, "我们重申落实《2025年议程》。")],
        alignment_key="different_number",
        config=fixed_ratio_config(),
    )
    assert matching.alignments[0].raw_alignment_cost < mismatching.alignments[0].raw_alignment_cost
    assert matching.alignments[0].normalized_confidence > mismatching.alignments[0].normalized_confidence


def test_length_mismatch_reduces_confidence() -> None:
    english = [sentence("EN", 1, 1, "We support cooperation.")]
    balanced = align_documents(
        english,
        [sentence("ZH", 1, 1, "我们支持合作。")],
        alignment_key="balanced",
        config=fixed_ratio_config(),
    )
    mismatched = align_documents(
        english,
        [sentence("ZH", 1, 1, "我们支持并持续全面深入扩大各领域长期务实合作。")],
        alignment_key="mismatched",
        config=fixed_ratio_config(),
    )
    assert balanced.alignments[0].raw_alignment_cost < mismatched.alignments[0].raw_alignment_cost
    assert balanced.alignments[0].normalized_confidence > mismatched.alignments[0].normalized_confidence


def test_repeatability_returns_exactly_equal_results() -> None:
    english = [sentence("EN", 1, 1, "We reaffirm cooperation."), sentence("EN", 1, 2, "We welcome progress.")]
    chinese = [sentence("ZH", 1, 1, "我们重申合作并欢迎取得进展。")]
    config = fixed_ratio_config()
    first = align_documents(english, chinese, alignment_key="repeatable", config=config)
    second = align_documents(english, chinese, alignment_key="repeatable", config=config)
    assert first == second


def test_length_ratio_is_estimated_from_sufficient_ordered_one_to_one_pairs() -> None:
    english = [
        sentence("EN", paragraph_no, 1, word)
        for paragraph_no, word in enumerate(("Cooperate", "Welcome", "Reaffirm"), start=1)
    ]
    chinese = [
        sentence("ZH", paragraph_no, 1, text)
        for paragraph_no, text in enumerate(("合作", "欢迎", "重申"), start=1)
    ]

    result = align_documents(
        english,
        chinese,
        alignment_key="estimated_ratio",
        config=AlignmentConfig(
            fallback_zh_chars_per_en_word=1.6,
            minimum_ratio_samples=3,
        ),
    )

    assert result.expected_zh_chars_per_en_word == 2.0


def test_sentence_alignment_never_crosses_paragraph_boundaries() -> None:
    english = [
        sentence("EN", 1, 1, "Paragraph one mentions 2030."),
        sentence("EN", 2, 1, "Paragraph two mentions 2025."),
    ]
    chinese = [
        sentence("ZH", 1, 1, "第一段提及2025。"),
        sentence("ZH", 2, 1, "第二段提及2030。"),
    ]
    result = align_documents(
        english, chinese, alignment_key="boundaries", config=fixed_ratio_config()
    )
    assert [(unit.paragraph_no, unit.english_sentence_ids, unit.chinese_sentence_ids) for unit in result.alignments] == [
        (1, ("EN_2025_P001_S01",), ("ZH_2025_P001_S01",)),
        (2, ("EN_2025_P002_S01",), ("ZH_2025_P002_S01",)),
    ]


def test_alignment_rejects_mixed_documents_within_one_language() -> None:
    english = [
        sentence("EN", 1, 1, "First declaration."),
        SentenceRecord(
            sentence_id="EN_OTHER_P002_S01",
            document_id="other_declaration_en",
            language="EN",
            paragraph_no=2,
            sentence_no=1,
            text="Second declaration.",
        ),
    ]
    chinese = [
        sentence("ZH", 1, 1, "第一份宣言。"),
        sentence("ZH", 2, 1, "第二份宣言。"),
    ]

    try:
        align_documents(
            english, chinese, alignment_key="mixed", config=fixed_ratio_config()
        )
    except ValueError as error:
        assert "mixes documents" in str(error)
    else:
        raise AssertionError("mixed English documents were silently accepted")


def test_two_to_two_operation_records_exact_sentence_cardinality() -> None:
    english = [
        sentence("EN", 1, index, f"English sentence {index}.")
        for index in range(1, 4)
    ]
    chinese = [
        sentence("ZH", 1, index, f"中文句子{index}。")
        for index in range(1, 5)
    ]
    result = align_documents(
        english,
        chinese,
        alignment_key="two_to_two",
        config=fixed_ratio_config(
            penalty_1_1=10.0,
            penalty_1_2=10.0,
            penalty_2_1=10.0,
            penalty_2_2=0.0,
        ),
    )
    two_to_two = next(
        unit for unit in result.alignments if unit.alignment_type == "2:2"
    )
    assert len(two_to_two.english_sentence_ids) == 2
    assert len(two_to_two.chinese_sentence_ids) == 2
    assert result.diagnostics.count_2_2 == 1


def test_impossible_count_ratio_is_reported_without_skipping() -> None:
    result = align_documents(
        [sentence("EN", 5, 1, "One sentence.")],
        [sentence("ZH", 5, 1, "一。"), sentence("ZH", 5, 2, "二。"), sentence("ZH", 5, 3, "三。")],
        alignment_key="failed",
        config=fixed_ratio_config(),
    )
    assert result.alignments == ()
    assert result.diagnostics.failed_paragraphs[0].paragraph_no == 5
    assert "skipping is disabled" in result.diagnostics.failed_paragraphs[0].reason


def test_diagnostics_report_contains_required_counts() -> None:
    result = align_documents(
        [sentence("EN", 1, 1, "We cooperate.")],
        [sentence("ZH", 1, 1, "我们合作。")],
        alignment_key="report",
        config=fixed_ratio_config(),
    )
    report = format_alignment_diagnostics(result.diagnostics)
    for expected in (
        "Total aligned units: 1",
        "1:1 count: 1",
        "1:2 count: 0",
        "2:1 count: 0",
        "2:2 count: 0",
        "High-confidence count:",
        "Medium-confidence count:",
        "Low-confidence count:",
    ):
        assert expected in report
