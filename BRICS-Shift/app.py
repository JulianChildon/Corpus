"""Minimal Streamlit interface for BRICS-Shift v0.1."""

from __future__ import annotations

import hashlib
from io import BytesIO, StringIO
import os
from pathlib import Path
import re
import sqlite3
import zipfile

import pandas as pd
import streamlit as st

from brics_shift.agreement import (
    DuplicateAgreementStudyError,
    build_agreement_report,
    create_agreement_study,
    get_agreement_progress,
    list_agreement_annotation_queue,
    list_agreement_studies,
)
from brics_shift.annotation import (
    ANNOTATOR_CONFIDENCE_VALUES,
    MODALITY_LABELS,
    STANCE_LABELS,
)
from brics_shift.database import (
    DuplicateAnnotationError,
    get_annotation,
    get_annotation_progress,
    get_available_years,
    get_corpus_overview,
    initialize_database,
    list_alignment_preview,
    list_alignments_for_annotation,
    save_annotation,
    update_annotation,
)
from brics_shift.export import (
    ExportConfig,
    TraceabilityError,
    build_reproducible_export_files,
    export_pilot_annotations_csv,
)
from brics_shift.ingestion import (
    PreparedCorpusPair,
    persist_prepared_pair,
    prepare_uploaded_pair,
)
from brics_shift.statistics import (
    Crosstab,
    build_descriptive_statistics,
    parse_period_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "brics_shift.sqlite3"
SELECT_PROMPT = "– Select –"

# Interface language controls presentation only.  Database values, annotation
# labels, document metadata, and exported research data remain canonical so
# that changing the display language never changes the research record.
LANGUAGE_EN = "en"
LANGUAGE_ZH = "zh-CN"
LANGUAGE_OPTIONS = ("简体中文", "English")
LANGUAGE_DISPLAY_TO_CODE = {
    "简体中文": LANGUAGE_ZH,
    "English": LANGUAGE_EN,
}

UI_TEXT: dict[str, dict[str, str]] = {
    LANGUAGE_EN: {
        "interface_language": "Interface language",
        "language.en": "English",
        "language.zh-CN": "简体中文",
        "navigation": "Navigation",
        "page.import": "Import",
        "page.corpus": "Corpus",
        "page.annotation": "Annotation",
        "page.agreement": "Agreement",
        "page.analysis": "Analysis",
        "page.export": "Export",
        "research_principle": (
            "The software organizes translation evidence. Humans interpret "
            "translation shifts."
        ),
        "select_prompt": "– Select –",
        "all": "All",
        "all_years": "All years",
        "year": "Year",
        "paragraph": "Paragraph",
        "sentence": "Sentence",
        "sentence_id": "Sentence ID",
        "text": "Text",
        "language": "Language",
        "warning_code": "Warning code",
        "paragraph_numbers": "Paragraph numbers",
        "message": "Message",
        "alignment_id": "Alignment ID",
        "alignment_type": "Alignment type",
        "alignment_confidence": "Alignment confidence",
        "confidence_band": "Confidence band",
        "document_title": "Document title",
        "english": "English",
        "chinese": "Chinese",
        "type": "Type",
        "confidence": "Confidence",
        "band": "Band",
        "source_text": "SOURCE TEXT — English",
        "target_text": "TARGET TEXT — Chinese",
        "modality_shift": "Modality Shift",
        "stance_shift": "Stance Shift",
        "annotator_confidence": "Annotator Confidence",
        "annotator_id": "Annotator ID",
        "guideline_version": "Guideline version",
        "notes": "Notes",
        "possible_alignment_error": "Possible alignment error",
        "previous": "Previous",
        "next": "Next",
        "save": "Save",
        "save_next": "Save & Next",
        "high": "high",
        "medium": "medium",
        "low": "low",
        "label.N/A": "N/A",
        "label.preserved": "preserved",
        "label.strengthened": "strengthened",
        "label.weakened": "weakened",
        "label.added": "added",
        "label.omitted": "omitted",
        "label.neutralized": "neutralized",
        "label.uncertain": "uncertain",
    },
    LANGUAGE_ZH: {
        "interface_language": "界面语言",
        "language.en": "English",
        "language.zh-CN": "简体中文",
        "navigation": "导航",
        "page.import": "导入",
        "page.corpus": "语料库",
        "page.annotation": "人工标注",
        "page.agreement": "一致性研究",
        "page.analysis": "描述统计",
        "page.export": "导出",
        "research_principle": "软件整理翻译证据；人类研究者解释翻译偏移。",
        "select_prompt": "– 请选择 –",
        "all": "全部",
        "all_years": "全部年份",
        "year": "年份",
        "paragraph": "段落",
        "sentence": "句子",
        "sentence_id": "句子 ID",
        "text": "文本",
        "language": "语言",
        "warning_code": "警告代码",
        "paragraph_numbers": "段落编号",
        "message": "说明",
        "alignment_id": "对齐 ID",
        "alignment_type": "对齐类型",
        "alignment_confidence": "对齐置信度",
        "confidence_band": "置信度等级",
        "document_title": "文献标题",
        "english": "英文",
        "chinese": "中文",
        "type": "类型",
        "confidence": "置信度",
        "band": "等级",
        "source_text": "源文本 — 英文",
        "target_text": "译文本 — 中文",
        "modality_shift": "情态偏移",
        "stance_shift": "立场偏移",
        "annotator_confidence": "标注者置信度",
        "annotator_id": "标注者 ID",
        "guideline_version": "指南版本",
        "notes": "备注",
        "possible_alignment_error": "可能存在对齐错误",
        "previous": "上一条",
        "next": "下一条",
        "save": "保存",
        "save_next": "保存并下一条",
        "high": "高",
        "medium": "中",
        "low": "低",
        "label.N/A": "不适用（N/A）",
        "label.preserved": "保持",
        "label.strengthened": "强化",
        "label.weakened": "弱化",
        "label.added": "新增",
        "label.omitted": "省略",
        "label.neutralized": "中和",
        "label.uncertain": "不确定",
    },
}


def _ui_language() -> str:
    """Return the current display language without affecting saved data."""
    selected = st.session_state.get("interface_language", "简体中文")
    return LANGUAGE_DISPLAY_TO_CODE.get(selected, selected)


def _t(key: str, **values: object) -> str:
    """Look up a UI string in the selected language."""
    text = UI_TEXT[_ui_language()].get(key, key)
    return text.format(**values)


def _l(english: str, chinese: str) -> str:
    """Select one of two explicitly supplied interface strings."""
    return english if _ui_language() == LANGUAGE_EN else chinese


def _display_annotation_value(value: str) -> str:
    """Display a canonical annotation value without changing the saved value."""
    if value == SELECT_PROMPT:
        return _t("select_prompt")
    if value in ANNOTATOR_CONFIDENCE_VALUES:
        return _t(value)
    return _t(f"label.{value}")


def _database_path() -> Path:
    configured = os.environ.get("BRICS_SHIFT_DB_PATH")
    return Path(configured) if configured else DEFAULT_DATABASE_PATH


def _paragraph_warning_rows(prepared: PreparedCorpusPair) -> list[dict]:
    rows: list[dict] = []
    for language, item in (
        ("EN", prepared.english),
        ("ZH", prepared.chinese),
    ):
        for warning in item.paragraph_result.warnings:
            rows.append(
                {
                    _t("language"): language,
                    _t("warning_code"): warning.code,
                    _t("paragraph_numbers"): ", ".join(
                        str(number) for number in warning.paragraph_numbers
                    ),
                    _t("message"): warning.message,
                }
            )
    return rows


def _sentence_preview_frame(item) -> pd.DataFrame:
    return pd.DataFrame(
        (
            {
                _t("paragraph"): sentence.paragraph_no,
                _t("sentence"): sentence.sentence_no,
                _t("sentence_id"): sentence.sentence_id,
                _t("text"): sentence.text,
            }
            for sentence in item.sentences
        )
    )


def _render_prepared_import(prepared: PreparedCorpusPair) -> None:
    st.subheader(
        "2. Preview paragraphs, sentence segmentation, and alignment"
        if _ui_language() == LANGUAGE_EN
        else "2. 预览段落、切句与对齐"
    )
    preview_prefix = (
        f"Preview: {prepared.pair_id} · {prepared.year} · {prepared.title} · "
        if _ui_language() == LANGUAGE_EN
        else f"预览对象：{prepared.pair_id} · {prepared.year} · {prepared.title} · "
    )
    st.caption(
        preview_prefix
        + f"{prepared.english.document.source_filename} / "
        + f"{prepared.chinese.document.source_filename}"
    )
    comparison = prepared.structure_comparison
    metrics = st.columns(5)
    metrics[0].metric(
        "English numbered paragraphs" if _ui_language() == LANGUAGE_EN else "英文编号段落",
        comparison.english_numbered_paragraphs,
    )
    metrics[1].metric(
        "Chinese numbered paragraphs" if _ui_language() == LANGUAGE_EN else "中文编号段落",
        comparison.chinese_numbered_paragraphs,
    )
    metrics[2].metric(
        "Matching paragraph numbers" if _ui_language() == LANGUAGE_EN else "匹配段落号",
        comparison.matching_paragraph_numbers,
    )
    metrics[3].metric("English sentences" if _ui_language() == LANGUAGE_EN else "英文句子", len(prepared.english.sentences))
    metrics[4].metric("Chinese sentences" if _ui_language() == LANGUAGE_EN else "中文句子", len(prepared.chinese.sentences))

    if comparison.missing_in_english:
        st.warning(
            (f"Missing English paragraph numbers: {list(comparison.missing_in_english)}")
            if _ui_language() == LANGUAGE_EN
            else f"英文缺少段落号：{list(comparison.missing_in_english)}"
        )
    if comparison.missing_in_chinese:
        st.warning(
            (f"Missing Chinese paragraph numbers: {list(comparison.missing_in_chinese)}")
            if _ui_language() == LANGUAGE_EN
            else f"中文缺少段落号：{list(comparison.missing_in_chinese)}"
        )
    warning_rows = _paragraph_warning_rows(prepared)
    if warning_rows:
        st.warning(
            "Paragraph-structure warnings were found. Review each one before import."
            if _ui_language() == LANGUAGE_EN
            else "检测到段落结构警告。请在确认导入前逐项检查。"
        )
        st.dataframe(pd.DataFrame(warning_rows), use_container_width=True, hide_index=True)
    else:
        st.success(
            "No paragraph-numbering anomalies were detected."
            if _ui_language() == LANGUAGE_EN
            else "未检测到段落编号异常。"
        )

    if prepared.english.paragraph_result.heading_text:
        with st.expander("English heading text" if _ui_language() == LANGUAGE_EN else "英文标题区文本"):
            st.text(prepared.english.paragraph_result.heading_text)
    if prepared.chinese.paragraph_result.heading_text:
        with st.expander("Chinese heading text" if _ui_language() == LANGUAGE_EN else "中文标题区文本"):
            st.text(prepared.chinese.paragraph_result.heading_text)

    english_tab, chinese_tab = st.tabs(
        ("English sentences", "Chinese sentences")
        if _ui_language() == LANGUAGE_EN
        else ("英文切句", "中文切句")
    )
    with english_tab:
        st.dataframe(
            _sentence_preview_frame(prepared.english),
            use_container_width=True,
            hide_index=True,
        )
    with chinese_tab:
        st.dataframe(
            _sentence_preview_frame(prepared.chinese),
            use_container_width=True,
            hide_index=True,
        )

    alignment_result = prepared.alignment_result
    st.markdown(
        "**Deterministic alignment preview**"
        if _ui_language() == LANGUAGE_EN
        else "**确定性对齐预览**"
    )
    alignment_metrics = st.columns(5)
    alignment_metrics[0].metric(
        "Aligned units" if _ui_language() == LANGUAGE_EN else "对齐单元",
        len(alignment_result.alignments),
    )
    alignment_metrics[1].metric("1:1", alignment_result.diagnostics.count_1_1)
    alignment_metrics[2].metric("1:2", alignment_result.diagnostics.count_1_2)
    alignment_metrics[3].metric("2:1", alignment_result.diagnostics.count_2_1)
    alignment_metrics[4].metric("2:2", alignment_result.diagnostics.count_2_2)
    st.caption(
        "Alignment confidence is a heuristic quality score, not a probability, "
        "and never generates translation-shift labels."
        if _ui_language() == LANGUAGE_EN
        else "对齐置信度是启发式质量分数，不是概率，也不会生成翻译偏移标签。"
    )
    alignment_frame = pd.DataFrame(
        (
            {
                _t("alignment_id"): unit.alignment_id,
                _t("paragraph"): unit.paragraph_no,
                _t("type"): unit.alignment_type,
                _t("confidence"): unit.normalized_confidence,
                _t("band"): _t(unit.confidence_band),
                _t("english"): unit.english_text,
                _t("chinese"): unit.chinese_text,
            }
            for unit in alignment_result.alignments
        )
    )
    if alignment_frame.empty:
        st.error(
            "Sentence segmentation produced no alignment units that can be saved."
            if _ui_language() == LANGUAGE_EN
            else "当前切句结果没有产生可保存的对齐单元。"
        )
    else:
        st.dataframe(alignment_frame, use_container_width=True, hide_index=True)

    if alignment_result.diagnostics.failed_paragraphs:
        st.error(
            "The following paragraphs cannot be aligned under the v0.1 "
            "1:1, 1:2, 2:1, and 2:2 rules:"
            if _ui_language() == LANGUAGE_EN
            else "以下段落无法在 v0.1 的 1:1、1:2、2:1、2:2 规则下完成对齐："
        )
        st.dataframe(
            pd.DataFrame(
                (
                    {
                        _t("paragraph"): failure.paragraph_no,
                        "EN sentence count" if _ui_language() == LANGUAGE_EN else "英文句子数": failure.english_sentence_count,
                        "ZH sentence count" if _ui_language() == LANGUAGE_EN else "中文句子数": failure.chinese_sentence_count,
                        "Reason" if _ui_language() == LANGUAGE_EN else "原因": failure.reason,
                    }
                    for failure in alignment_result.diagnostics.failed_paragraphs
                )
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_import_page(connection) -> None:
    st.header(_l("Import Raw Text", "上传原始文本"))
    st.caption(
        _l(
            "Upload one official English text and its official Chinese translation. "
            "The application previews cleaning, numbered paragraphs, deterministic "
            "sentence segmentation, and alignment before any database write.",
            "上传一份英文官方文本和对应的中文官方译文。程序将先预览清洗、编号段落、"
            "规则切句和确定性对齐；只有再次确认后才写入数据库。",
        )
    )
    flash = st.session_state.pop("import_flash", None)
    if flash:
        st.success(flash)

    st.subheader(_l("1. Upload files and enter metadata", "1. 上传文件与填写元数据"))
    metadata_columns = st.columns(3)
    pair_id = metadata_columns[0].text_input(
        _l("Corpus-pair ID", "语料对 ID"),
        placeholder=_l("e.g. 2025_rio", "例如：2025_rio"),
        help=_l(
            "Use letters, numbers, underscores, and hyphens only. The system "
            "creates _en and _zh document IDs.",
            "只能使用字母、数字、下划线和连字符；系统会生成 _en 和 _zh 文档 ID。",
        ),
    )
    year = int(
        metadata_columns[1].number_input(
            _t("year"), min_value=1900, max_value=2200, value=2025, step=1
        )
    )
    title = metadata_columns[2].text_input(
        _l("Declaration title", "宣言标题"), placeholder=_l("e.g. Rio Declaration", "例如：Rio Declaration")
    )
    file_columns = st.columns(2)
    english_upload = file_columns[0].file_uploader(
        _l("Official English text (.txt)", "英文官方文本 (.txt)"),
        type=("txt",),
        key="import_en_file",
    )
    chinese_upload = file_columns[1].file_uploader(
        _l("Official Chinese translation (.txt)", "中文官方译文 (.txt)"),
        type=("txt",),
        key="import_zh_file",
    )
    url_columns = st.columns(2)
    english_url = url_columns[0].text_input(
        _l("English source URL (optional)", "英文来源 URL（可选）"), key="import_en_url"
    )
    chinese_url = url_columns[1].text_input(
        _l("Chinese source URL (optional)", "中文来源 URL（可选）"), key="import_zh_url"
    )
    st.info(
        _l(
            "Files must be UTF-8 TXT and use Arabic paragraph numbers. English "
            "sentence segmentation protects common abbreviations, initialisms, and "
            "decimals; researchers must still inspect sentence boundaries in the preview.",
            "文件必须是 UTF-8 TXT，并以阿拉伯数字段落号组织。英文切句保护常见缩写、"
            "首字母缩写和小数；仍须由研究者在预览表中检查句界。",
        )
    )
    prepare_clicked = st.button(
        _l("Generate sentence and alignment preview", "生成切句与对齐预览"),
        type="primary",
        disabled=english_upload is None or chinese_upload is None,
        use_container_width=True,
    )
    if prepare_clicked:
        st.session_state.pop("prepared_corpus_import", None)
        try:
            prepared = prepare_uploaded_pair(
                pair_id=pair_id,
                year=year,
                title=title,
                english_bytes=english_upload.getvalue(),
                english_filename=english_upload.name,
                chinese_bytes=chinese_upload.getvalue(),
                chinese_filename=chinese_upload.name,
                english_source_url=english_url,
                chinese_source_url=chinese_url,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["prepared_corpus_import"] = prepared

    prepared = st.session_state.get("prepared_corpus_import")
    if prepared is None:
        return
    _render_prepared_import(prepared)
    st.subheader(_l("3. Confirm database import", "3. 确认写入数据库"))
    current_english_checksum = (
        hashlib.sha256(english_upload.getvalue()).hexdigest()
        if english_upload is not None
        else None
    )
    current_chinese_checksum = (
        hashlib.sha256(chinese_upload.getvalue()).hexdigest()
        if chinese_upload is not None
        else None
    )
    preview_matches_inputs = (
        pair_id.strip() == prepared.pair_id
        and year == prepared.year
        and title.strip() == prepared.title
        and english_upload is not None
        and english_upload.name == prepared.english.document.source_filename
        and chinese_upload is not None
        and chinese_upload.name == prepared.chinese.document.source_filename
        and current_english_checksum == prepared.english.checksum
        and current_chinese_checksum == prepared.chinese.checksum
        and (english_url.strip() or None) == prepared.english.document.source_url
        and (chinese_url.strip() or None) == prepared.chinese.document.source_url
    )
    if not preview_matches_inputs:
        st.error(
            _l(
                "The files or metadata changed after the preview. Generate a new "
                "preview before importing.",
                "文件或元数据已在预览后发生变化，请重新生成预览后再导入。",
            )
        )
    reviewed = st.checkbox(
        _l(
            "I have reviewed the paragraph warnings, all sentence boundaries, and "
            "unalignable paragraphs, and confirm that this preview may be saved.",
            "我已检查段落警告、全部句界和无法对齐的段落，并确认保存当前预览结果。",
        ),
        key=f"confirm_import_{prepared.pair_id}",
    )
    st.warning(
        _l(
            "Confirmation saves the raw TXT files to data/raw and writes documents, "
            "paragraphs, sentences, and successful alignment units to SQLite. Existing "
            "records and same-named raw files with different content are never overwritten.",
            "确认后会保存原始 TXT 到 data/raw，并把文档、段落、句子和成功的对齐单元"
            "写入 SQLite。已有记录和不同内容的同名原始文件不会被覆盖。",
        )
    )
    if not st.button(
        _l("Confirm import and open Corpus", "确认导入并进入语料库"),
        disabled=(
            not reviewed
            or not preview_matches_inputs
            or not prepared.alignment_result.alignments
        ),
        use_container_width=True,
    ):
        return
    try:
        summary = persist_prepared_pair(
            connection,
            prepared,
            raw_directory=PROJECT_ROOT / "data" / "raw",
        )
    except (ValueError, OSError, sqlite3.IntegrityError) as error:
        st.error(_l(f"Import failed; existing data was not overwritten: {error}", f"导入失败，未覆盖已有数据：{error}"))
        return
    st.session_state.pop("prepared_corpus_import", None)
    st.session_state["import_flash"] = (
        _l(
            f"Import complete: {summary.documents} documents, {summary.paragraphs} paragraphs, "
            f"{summary.sentences} sentences, and {summary.alignments} alignment units. "
            "You can now inspect Corpus and begin human annotation in Annotation.",
            f"导入完成：{summary.documents} 个文档、{summary.paragraphs} 个段落、"
            f"{summary.sentences} 个句子、{summary.alignments} 个对齐单元。"
            "现在可进入语料库检查，并在人工标注页开始标注。",
        )
    )
    st.rerun()


def _render_corpus_page(connection) -> None:
    st.header(_t("page.corpus"))
    st.caption(_l("Basic corpus structure and alignment overview", "语料库结构与对齐概览"))

    years = get_available_years(connection)
    selected_year = st.selectbox(
        _l("Filter by year", "按年份筛选"),
        ("all_years", *years),
        format_func=lambda value: _t(value) if value == "all_years" else str(value),
    )
    year = None if selected_year == "all_years" else int(selected_year)
    overview = get_corpus_overview(connection, year=year)

    metrics = (
        (_l("Documents", "文档数"), overview.document_count),
        (_l("Years represented", "涵盖年份"), ", ".join(map(str, overview.years)) or "—"),
        (_l("English sentences", "英文句子数"), overview.english_sentence_count),
        (_l("Chinese sentences", "中文句子数"), overview.chinese_sentence_count),
        (_l("Aligned units", "对齐单元数"), overview.aligned_unit_count),
        ("1:1", overview.count_1_1),
        ("1:2", overview.count_1_2),
        ("2:1", overview.count_2_1),
        ("2:2", overview.count_2_2),
        (_l("High confidence", "高置信度"), overview.high_confidence_count),
        (_l("Medium confidence", "中等置信度"), overview.medium_confidence_count),
        (_l("Low confidence", "低置信度"), overview.low_confidence_count),
    )
    for start in range(0, len(metrics), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, metrics[start : start + 4]):
            column.metric(label, value)

    st.subheader(_l("Alignment preview", "对齐预览"))
    preview = list_alignment_preview(connection, year=year, limit=100)
    if not preview:
        st.info(_l("No alignments are available for this filter.", "当前筛选条件下没有对齐单元。"))
        return

    frame = pd.DataFrame(preview).rename(
        columns={
            "year": _t("year"),
            "document_title": _t("document_title"),
            "paragraph_no": _t("paragraph"),
            "alignment_id": _t("alignment_id"),
            "alignment_type": _t("type"),
            "alignment_confidence": _t("confidence"),
            "alignment_confidence_band": _t("band"),
            "en_text": _t("english"),
            "zh_text": _t("chinese"),
        }
    )
    frame[_t("confidence")] = frame[_t("confidence")].round(3)
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _annotation_id(alignment_id: str, annotator_id: str) -> str:
    """Create a stable ID without exposing arbitrary annotator input."""
    safe_alignment = re.sub(r"[^A-Za-z0-9_-]+", "_", alignment_id).strip("_")
    annotator_digest = hashlib.sha256(annotator_id.encode("utf-8")).hexdigest()[:12]
    return f"ANN_{safe_alignment}_{annotator_digest}"


def _select_index(options: tuple[str, ...], existing_value: str | None) -> int:
    return options.index(existing_value) if existing_value in options else 0


def _missing_annotation_fields(
    modality_label: str,
    stance_label: str,
    annotator_confidence: str,
    guideline_version: str,
) -> list[str]:
    """Return required fields that were not deliberately selected or entered."""
    missing: list[str] = []
    if modality_label == SELECT_PROMPT:
        missing.append("modality_shift")
    if stance_label == SELECT_PROMPT:
        missing.append("stance_shift")
    if annotator_confidence == SELECT_PROMPT:
        missing.append("annotator_confidence")
    if not guideline_version.strip():
        missing.append("guideline_version")
    return missing


def _next_unannotated_id(
    queue: tuple[dict, ...],
    current_index: int,
) -> str | None:
    """Find the next unannotated row, wrapping once through the queue."""
    later = queue[current_index + 1 :]
    earlier = queue[:current_index]
    for record in (*later, *earlier):
        if record["annotation_id"] is None:
            return record["alignment_id"]
    return None


def _render_annotation_page(connection) -> None:
    st.header(_t("page.annotation"))
    st.caption(_l("Human translation-shift annotation of aligned EN-ZH units", "对齐英汉单元的人工翻译偏移标注"))
    st.caption(
        _l(
            "Alignment confidence is a heuristic quality score, not a calibrated "
            "probability. It never supplies a translation-shift label.",
            "对齐置信度是启发式质量分数，不是经校准的概率，也不会提供翻译偏移标签。",
        )
    )
    annotation_mode = st.radio(
        _l("Mode", "模式"),
        ("annotate", "review"),
        format_func=lambda value: _l("Annotate", "标注") if value == "annotate" else _l("Review annotated", "审阅已标注内容"),
        horizontal=True,
        help=_l("Review mode loads only this annotator's saved pilot examples.", "审阅模式只载入当前标注者已保存的试标示例。"),
    )

    annotator_id = st.sidebar.text_input(
        _t("annotator_id"),
        key="annotator_id",
        placeholder="e.g. researcher_1",
    ).strip()
    guideline_version = st.sidebar.text_input(
        _t("guideline_version"),
        key="guideline_version",
        placeholder="e.g. guidelines-v1.0",
        help=_l("The exact frozen guideline version is stored with every annotation.", "每条标注均会保存所使用的冻结版指南版本。"),
    ).strip()
    minimum_confidence = st.sidebar.slider(
        _l("Annotation confidence threshold", "标注队列置信度阈值"),
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
        help=_l("Alignments below this automatic alignment-confidence score are quarantined.", "低于此自动对齐置信度的单元会被隔离。"),
    )
    show_all = st.sidebar.toggle(
        _l("Show all alignments", "显示全部对齐单元"),
        value=False,
        help=_l("Includes low-confidence and alignment-error-flagged units.", "包括低置信度单元和已标记可能对齐错误的单元。"),
    )

    st.sidebar.subheader(_l("Pilot filters", "试标筛选"))
    years = get_available_years(connection)
    year_choice = st.sidebar.selectbox(
        _t("year"),
        ("all", *years),
        format_func=lambda value: _t("all") if value == "all" else str(value),
        key="annotation_filter_year",
    )
    alignment_type_choice = st.sidebar.selectbox(
        _t("alignment_type"),
        ("all", "1:1", "1:2", "2:1", "2:2"),
        format_func=lambda value: _t("all") if value == "all" else value,
        key="annotation_filter_type",
    )
    band_choice = st.sidebar.selectbox(
        _t("confidence_band"),
        ("all", "high", "medium", "low"),
        format_func=lambda value: _t(value),
        key="annotation_filter_band",
    )
    if annotation_mode == "review":
        annotation_status = "annotated"
        st.sidebar.caption(_l("Annotation status: annotated", "标注状态：已标注"))
    else:
        status_choice = st.sidebar.selectbox(
            _l("Annotation status", "标注状态"),
            ("unannotated", "all", "annotated"),
            format_func=lambda value: {
                "unannotated": _l("Unannotated", "未标注"),
                "all": _t("all"),
                "annotated": _l("Annotated", "已标注"),
            }[value],
            key="annotation_filter_status",
        )
        annotation_status = status_choice
    error_choice = st.sidebar.selectbox(
        _t("possible_alignment_error"),
        ("all", "flagged", "not_flagged"),
        format_func=lambda value: {
            "all": _t("all"),
            "flagged": _l("Flagged", "已标记"),
            "not_flagged": _l("Not flagged", "未标记"),
        }[value],
        key="annotation_filter_error",
    )
    modality_choice = st.sidebar.selectbox(
        _l("Modality label", "情态标签"),
        ("all", *MODALITY_LABELS),
        format_func=lambda value: _t("all") if value == "all" else _display_annotation_value(value),
        key="annotation_filter_modality",
    )
    stance_choice = st.sidebar.selectbox(
        _l("Stance label", "立场标签"),
        ("all", *STANCE_LABELS),
        format_func=lambda value: _t("all") if value == "all" else _display_annotation_value(value),
        key="annotation_filter_stance",
    )
    notes_search = st.sidebar.text_input(
        _l("Search notes", "搜索备注"),
        key="annotation_notes_search",
        placeholder=_l("Review mode: search note text", "审阅模式：搜索备注文本"),
    )

    if not annotator_id:
        st.info(_l("Enter an annotator ID in the sidebar to open the annotation queue.", "请在侧栏输入标注者 ID 以打开标注队列。"))
        return

    queue_threshold = None if show_all else minimum_confidence
    filter_year = None if year_choice == "all" else int(year_choice)
    filter_alignment_type = (
        None if alignment_type_choice == "all" else alignment_type_choice
    )
    filter_band = None if band_choice == "all" else band_choice
    filter_error = {
        "all": None,
        "flagged": True,
        "not_flagged": False,
    }[error_choice]
    filter_modality = None if modality_choice == "all" else modality_choice
    filter_stance = None if stance_choice == "all" else stance_choice
    queue_include_errors = show_all or filter_error is True
    queue = list_alignments_for_annotation(
        connection,
        annotator_id,
        year=filter_year,
        minimum_confidence=queue_threshold,
        include_possible_alignment_errors=queue_include_errors,
        alignment_type=filter_alignment_type,
        confidence_band=filter_band,
        annotation_status=annotation_status,
        possible_alignment_error=filter_error,
        modality_label=filter_modality,
        stance_label=filter_stance,
        notes_search=notes_search,
    )
    progress = get_annotation_progress(
        connection,
        annotator_id,
        year=filter_year,
        minimum_confidence=queue_threshold,
        include_possible_alignment_errors=show_all,
    )
    st.write(
        _l(
            f"Annotated: **{progress.annotated_alignments} / {progress.total_alignments}**",
            f"已标注：**{progress.annotated_alignments} / {progress.total_alignments}**",
        )
    )
    st.progress(
        progress.annotated_alignments / progress.total_alignments
        if progress.total_alignments
        else 0.0,
        text=_l(
            f"Completion: {progress.completion_percentage:.1f}%",
            f"完成度：{progress.completion_percentage:.1f}%",
        ),
    )

    flash_message = st.session_state.pop("annotation_flash", None)
    if flash_message:
        st.success(flash_message)

    if not queue:
        st.info(_l("No alignments are available under the current queue rules.", "当前队列规则下没有可用的对齐单元。"))
        return

    queue_context = (
        annotator_id,
        annotation_mode,
        show_all,
        round(minimum_confidence, 4),
        filter_year,
        filter_alignment_type,
        filter_band,
        annotation_status,
        filter_error,
        filter_modality,
        filter_stance,
        notes_search,
    )
    if st.session_state.get("annotation_queue_context") != queue_context:
        first_unannotated = next(
            (record for record in queue if record["annotation_id"] is None),
            queue[0],
        )
        st.session_state["annotation_current_id"] = first_unannotated["alignment_id"]
        st.session_state["annotation_queue_context"] = queue_context

    queue_ids = [record["alignment_id"] for record in queue]
    current_id = st.session_state.get("annotation_current_id")
    if current_id not in queue_ids:
        current_id = queue[0]["alignment_id"]
        st.session_state["annotation_current_id"] = current_id
    current_index = queue_ids.index(current_id)
    alignment = queue[current_index]

    existing = get_annotation(
        connection,
        alignment_id=alignment["alignment_id"],
        annotator_id=annotator_id,
    )

    st.caption(_l(f"Queue item {current_index + 1} of {len(queue)}", f"队列第 {current_index + 1} 条，共 {len(queue)} 条"))
    metadata_columns = st.columns(4)
    metadata_columns[0].metric(_t("year"), alignment["year"])
    metadata_columns[1].metric(_t("paragraph"), alignment["paragraph_no"])
    metadata_columns[2].metric(_t("alignment_type"), alignment["alignment_type"])
    metadata_columns[3].metric(
        _t("alignment_confidence"), f"{alignment['alignment_confidence']:.3f}"
    )
    st.write(f"**{_t('document_title')}:** {alignment['document_title']}")
    st.write(f"**{_t('alignment_id')}:** `{alignment['alignment_id']}`")
    st.write(f"**{_t('confidence_band')}:** {_t(alignment['alignment_confidence_band'])}")
    if alignment["possible_alignment_error"]:
        st.warning(_l("This alignment has been flagged for possible alignment error.", "该对齐单元已标记为可能存在对齐错误。"))

    source_column, target_column = st.columns(2)
    with source_column:
        with st.container(border=True):
            st.subheader(_t("source_text"))
            st.write(alignment["en_text"])
    with target_column:
        with st.container(border=True):
            st.subheader(_t("target_text"))
            st.write(alignment["zh_text"])

    if existing:
        st.info(
            _l(
                "An existing annotation by this annotator has been loaded. Saving "
                "will perform an explicit update.",
                "已载入该标注者已有的标注记录。保存时将执行明确更新。",
            )
        )

    widget_suffix = f"{annotator_id}_{alignment['alignment_id']}"
    modality_options = (SELECT_PROMPT, *MODALITY_LABELS)
    stance_options = (SELECT_PROMPT, *STANCE_LABELS)
    confidence_options = (SELECT_PROMPT, *ANNOTATOR_CONFIDENCE_VALUES)
    modality_label = st.selectbox(
        _t("modality_shift"),
        modality_options,
        index=_select_index(
            modality_options, existing["modality_label"] if existing else None
        ),
        key=f"modality_{widget_suffix}",
        format_func=_display_annotation_value,
    )
    stance_label = st.selectbox(
        _t("stance_shift"),
        stance_options,
        index=_select_index(
            stance_options, existing["stance_label"] if existing else None
        ),
        key=f"stance_{widget_suffix}",
        format_func=_display_annotation_value,
    )
    annotator_confidence = st.selectbox(
        _t("annotator_confidence"),
        confidence_options,
        index=_select_index(
            confidence_options,
            existing["annotator_confidence"] if existing else None,
        ),
        key=f"annotator_confidence_{widget_suffix}",
        format_func=_display_annotation_value,
    )
    notes = st.text_area(
        _t("notes"),
        value=existing["notes"] if existing else "",
        key=f"notes_{widget_suffix}",
    )
    possible_alignment_error = st.checkbox(
        _t("possible_alignment_error"),
        value=existing["possible_alignment_error"] if existing else False,
        key=f"alignment_error_{widget_suffix}",
        help=(
            _l(
                "The annotator suspects that the English and Chinese texts may not "
                "be correctly aligned. This does not change either shift label.",
                "标注者怀疑英汉文本可能未被正确对齐。这不会改变任一翻译偏移标签。",
            )
        ),
    )

    button_columns = st.columns(4)
    previous_clicked = button_columns[0].button(
        _t("previous"), disabled=current_index == 0, use_container_width=True
    )
    save_clicked = button_columns[1].button(_t("save"), use_container_width=True)
    save_next_clicked = button_columns[2].button(
        _t("save_next"), use_container_width=True
    )
    next_clicked = button_columns[3].button(
        _t("next"), disabled=current_index == len(queue) - 1, use_container_width=True
    )

    if previous_clicked:
        st.session_state["annotation_current_id"] = queue[current_index - 1][
            "alignment_id"
        ]
        st.rerun()
    if next_clicked:
        st.session_state["annotation_current_id"] = queue[current_index + 1][
            "alignment_id"
        ]
        st.rerun()

    if not (save_clicked or save_next_clicked):
        return

    missing_fields = _missing_annotation_fields(
        modality_label,
        stance_label,
        annotator_confidence,
        guideline_version,
    )
    if missing_fields:
        st.error(
            _l("Select or enter: ", "请选择或输入：")
            + ", ".join(_t(field) for field in missing_fields)
            + _l(".", "。")
        )
        return

    if existing:
        update_annotation(
            connection,
            existing["annotation_id"],
            modality_label=modality_label,
            stance_label=stance_label,
            annotator_confidence=annotator_confidence,
            notes=notes,
            possible_alignment_error=possible_alignment_error,
            annotation_guideline_version=guideline_version,
        )
        action = "updated"
    else:
        try:
            save_annotation(
                connection,
                annotation_id=_annotation_id(alignment["alignment_id"], annotator_id),
                alignment_id=alignment["alignment_id"],
                annotator_id=annotator_id,
                modality_label=modality_label,
                stance_label=stance_label,
                annotator_confidence=annotator_confidence,
                notes=notes,
                possible_alignment_error=possible_alignment_error,
                annotation_guideline_version=guideline_version,
            )
        except DuplicateAnnotationError as error:
            st.error(str(error))
            return
        action = "saved"

    message = _l(
        f"Annotation {action} for {alignment['alignment_id']}.",
        f"已{'更新' if action == 'updated' else '保存'}对齐单元 {alignment['alignment_id']} 的标注。",
    )
    if save_next_clicked:
        next_id = _next_unannotated_id(queue, current_index)
        if next_id is not None:
            st.session_state["annotation_current_id"] = next_id
            st.session_state["annotation_flash"] = message
            st.rerun()
        st.success(message + _l(" No further unannotated alignment is available.", " 没有更多未标注的对齐单元。"))
    else:
        st.success(message)


def _agreement_study_selector(studies, *, key: str):
    study_by_id = {study.study_id: study for study in studies}
    selected_id = st.selectbox(
        _l("Agreement study", "一致性研究样本"),
        tuple(study_by_id),
        format_func=lambda value: (
            f"{study_by_id[value].study_name} ({value})"
        ),
        key=key,
    )
    return study_by_id[selected_id]


def _render_agreement_study_creation(connection) -> None:
    st.subheader(_l("Create a reproducible agreement sample", "创建可复现的一致性研究样本"))
    st.caption(
        _l(
            "Sampling is fixed-seed and proportional within the selected strata. "
            "The definition and exact sample membership are stored in SQLite.",
            "抽样使用固定随机种子，并在所选分层内按比例进行。样本定义和精确成员会保存到 SQLite。",
        )
    )
    with st.form("create_agreement_study"):
        study_id = st.text_input(_l("Study ID", "研究 ID"), placeholder="e.g. AGREE_PILOT_01")
        study_name = st.text_input(
            _l("Study name", "研究名称"), placeholder=_l("e.g. Pilot double annotation", "例如：试标双人标注")
        )
        annotator_columns = st.columns(2)
        annotator_a_id = annotator_columns[0].text_input(
            _l("Annotator A ID", "标注者 A ID"), placeholder="e.g. researcher_1"
        )
        annotator_b_id = annotator_columns[1].text_input(
            _l("Annotator B ID", "标注者 B ID"), placeholder="e.g. researcher_2"
        )
        sampling_columns = st.columns(3)
        sample_size = sampling_columns[0].number_input(
            _l("Sample size", "样本量"), min_value=1, value=100, step=1
        )
        random_seed = sampling_columns[1].number_input(
            _l("Random seed", "随机种子"), min_value=0, max_value=2_147_483_647,
            value=2025, step=1
        )
        minimum_confidence = sampling_columns[2].number_input(
            _l("Minimum alignment confidence", "最低对齐置信度"), min_value=0.0, max_value=1.0,
            value=0.70, step=0.05, format="%.2f"
        )
        option_columns = st.columns(3)
        stratify_year = option_columns[0].checkbox(_l("Stratify by year", "按年份分层"))
        stratify_type = option_columns[1].checkbox(_l("Stratify by alignment type", "按对齐类型分层"))
        include_errors = option_columns[2].checkbox(
            _l("Include possible alignment errors", "纳入可能存在对齐错误的单元")
        )
        submitted = st.form_submit_button(
            _l("Create agreement study", "创建一致性研究"), use_container_width=True
        )
    if not submitted:
        return
    try:
        study = create_agreement_study(
            connection,
            study_id=study_id.strip(),
            study_name=study_name.strip(),
            annotator_a_id=annotator_a_id.strip(),
            annotator_b_id=annotator_b_id.strip(),
            sample_size=int(sample_size),
            random_seed=int(random_seed),
            stratify_by_year=stratify_year,
            stratify_by_alignment_type=stratify_type,
            minimum_alignment_confidence=float(minimum_confidence),
            include_possible_alignment_errors=include_errors,
        )
    except (ValueError, DuplicateAgreementStudyError) as error:
        st.error(str(error))
        return
    st.success(
        _l(
            f"Created {study.study_id} with {study.sample_size} sampled alignments.",
            f"已创建 {study.study_id}，包含 {study.sample_size} 个抽样对齐单元。",
        )
    )
    with st.expander(_l("Stored sample definition", "已保存的样本定义"), expanded=True):
        st.json(dict(study.sample_definition))


def _render_agreement_annotation(connection, studies) -> None:
    study = _agreement_study_selector(studies, key="agreement_annotation_study")
    st.info(
        _l(
            "Independent annotation mode: only your own saved labels are loaded. "
            "The other participant's labels remain hidden.",
            "独立标注模式：只载入您自己保存的标签，另一位标注者的标签保持隐藏。",
        )
    )
    input_columns = st.columns(3)
    annotator_id = input_columns[0].text_input(
        _t("annotator_id"),
        value=st.session_state.get("annotator_id", ""),
        key="agreement_annotator_id",
    ).strip()
    guideline_version = input_columns[1].text_input(
        _t("guideline_version"),
        value=st.session_state.get("guideline_version", ""),
        key="agreement_guideline_version",
    ).strip()
    status = input_columns[2].selectbox(
        _l("Queue status", "队列状态"),
        ("unannotated", "all", "annotated"),
        format_func=lambda value: {
            "unannotated": _l("Unannotated", "未标注"),
            "all": _t("all"),
            "annotated": _l("Annotated", "已标注"),
        }[value],
        key="agreement_annotation_status",
    )
    if not annotator_id:
        st.info(_l("Enter your assigned annotator ID to open the sample.", "请输入分配给您的标注者 ID 以打开样本。"))
        return
    try:
        queue = list_agreement_annotation_queue(
            connection, study.study_id, annotator_id,
            annotation_status=status,
        )
    except ValueError as error:
        st.error(str(error))
        return

    progress = get_agreement_progress(connection, study.study_id)
    own_count = (
        progress.annotator_a_count
        if annotator_id == study.annotator_a_id
        else progress.annotator_b_count
    )
    st.write(_l(f"Your progress: **{own_count} / {progress.sample_size}**", f"您的进度：**{own_count} / {progress.sample_size}**"))
    st.progress(
        own_count / progress.sample_size if progress.sample_size else 0.0,
        text=_l(
            f"Completion: {(100 * own_count / progress.sample_size):.1f}%",
            f"完成度：{(100 * own_count / progress.sample_size):.1f}%",
        ),
    )
    flash_message = st.session_state.pop("agreement_annotation_flash", None)
    if flash_message:
        st.success(flash_message)
    if not queue:
        st.info(_l("No sample units are available under this queue status.", "该队列状态下没有可用的样本单元。"))
        return

    context = (study.study_id, annotator_id, status)
    if st.session_state.get("agreement_queue_context") != context:
        first_unannotated = next(
            (record for record in queue if record["annotation_id"] is None),
            queue[0],
        )
        st.session_state["agreement_current_id"] = first_unannotated["alignment_id"]
        st.session_state["agreement_queue_context"] = context
    queue_ids = [record["alignment_id"] for record in queue]
    current_id = st.session_state.get("agreement_current_id")
    if current_id not in queue_ids:
        current_id = queue_ids[0]
        st.session_state["agreement_current_id"] = current_id
    current_index = queue_ids.index(current_id)
    alignment = queue[current_index]
    existing = get_annotation(
        connection,
        alignment_id=alignment["alignment_id"],
        annotator_id=annotator_id,
    )

    st.caption(
        _l(
            f"Sample item {alignment['sample_order']} of {study.sample_size} "
            f"(queue item {current_index + 1} of {len(queue)})",
            f"样本第 {alignment['sample_order']} 条，共 {study.sample_size} 条 "
            f"（队列第 {current_index + 1} 条，共 {len(queue)} 条）",
        )
    )
    metadata_columns = st.columns(4)
    metadata_columns[0].metric(_t("year"), alignment["year"])
    metadata_columns[1].metric(_t("paragraph"), alignment["paragraph_no"])
    metadata_columns[2].metric(_t("alignment_type"), alignment["alignment_type"])
    metadata_columns[3].metric(
        _t("alignment_confidence"), f"{alignment['alignment_confidence']:.3f}"
    )
    st.write(f"**{_t('document_title')}:** {alignment['document_title']}")
    st.write(f"**{_t('alignment_id')}:** `{alignment['alignment_id']}`")
    source_column, target_column = st.columns(2)
    with source_column:
        with st.container(border=True):
            st.subheader(_t("source_text"))
            st.write(alignment["en_text"])
    with target_column:
        with st.container(border=True):
            st.subheader(_t("target_text"))
            st.write(alignment["zh_text"])
    if existing:
        st.info(_l("Your existing record has been loaded for explicit editing.", "已载入您的已有记录，可进行明确编辑。"))

    widget_suffix = f"agreement_{study.study_id}_{annotator_id}_{alignment['alignment_id']}"
    modality_options = (SELECT_PROMPT, *MODALITY_LABELS)
    stance_options = (SELECT_PROMPT, *STANCE_LABELS)
    confidence_options = (SELECT_PROMPT, *ANNOTATOR_CONFIDENCE_VALUES)
    modality_label = st.selectbox(
        _t("modality_shift"), modality_options,
        index=_select_index(
            modality_options, existing["modality_label"] if existing else None
        ),
        key=f"modality_{widget_suffix}", format_func=_display_annotation_value,
    )
    stance_label = st.selectbox(
        _t("stance_shift"), stance_options,
        index=_select_index(
            stance_options, existing["stance_label"] if existing else None
        ),
        key=f"stance_{widget_suffix}", format_func=_display_annotation_value,
    )
    annotator_confidence = st.selectbox(
        _t("annotator_confidence"), confidence_options,
        index=_select_index(
            confidence_options,
            existing["annotator_confidence"] if existing else None,
        ),
        key=f"confidence_{widget_suffix}", format_func=_display_annotation_value,
    )
    notes = st.text_area(
        _t("notes"), value=existing["notes"] if existing else "",
        key=f"notes_{widget_suffix}",
    )
    possible_alignment_error = st.checkbox(
        _t("possible_alignment_error"),
        value=existing["possible_alignment_error"] if existing else False,
        key=f"error_{widget_suffix}",
    )

    buttons = st.columns(4)
    previous_clicked = buttons[0].button(
        _t("previous"), disabled=current_index == 0, use_container_width=True,
        key="agreement_previous"
    )
    save_clicked = buttons[1].button(
        _t("save"), use_container_width=True, key="agreement_save"
    )
    save_next_clicked = buttons[2].button(
        _t("save_next"), use_container_width=True, key="agreement_save_next"
    )
    next_clicked = buttons[3].button(
        _t("next"), disabled=current_index == len(queue) - 1,
        use_container_width=True, key="agreement_next"
    )
    if previous_clicked:
        st.session_state["agreement_current_id"] = queue_ids[current_index - 1]
        st.rerun()
    if next_clicked:
        st.session_state["agreement_current_id"] = queue_ids[current_index + 1]
        st.rerun()
    if not (save_clicked or save_next_clicked):
        return

    missing = _missing_annotation_fields(
        modality_label,
        stance_label,
        annotator_confidence,
        guideline_version,
    )
    if missing:
        st.error(
            _l("Select or enter: ", "请选择或输入：")
            + ", ".join(_t(field) for field in missing)
            + _l(".", "。")
        )
        return
    if existing:
        update_annotation(
            connection, existing["annotation_id"],
            modality_label=modality_label, stance_label=stance_label,
            annotator_confidence=annotator_confidence, notes=notes,
            possible_alignment_error=possible_alignment_error,
            annotation_guideline_version=guideline_version,
        )
        action = "updated"
    else:
        try:
            save_annotation(
                connection,
                annotation_id=_annotation_id(alignment["alignment_id"], annotator_id),
                alignment_id=alignment["alignment_id"],
                annotator_id=annotator_id,
                modality_label=modality_label,
                stance_label=stance_label,
                annotator_confidence=annotator_confidence,
                notes=notes,
                possible_alignment_error=possible_alignment_error,
                annotation_guideline_version=guideline_version,
            )
        except DuplicateAnnotationError as error:
            st.error(str(error))
            return
        action = "saved"
    message = _l(
        f"Independent annotation {action} for {alignment['alignment_id']}.",
        f"已{'更新' if action == 'updated' else '保存'}对齐单元 {alignment['alignment_id']} 的独立标注。",
    )
    if save_next_clicked:
        next_id = _next_unannotated_id(queue, current_index)
        if next_id is not None:
            st.session_state["agreement_current_id"] = next_id
            st.session_state["agreement_annotation_flash"] = message
            st.rerun()
        st.success(message + _l(" No further unannotated sample unit is available.", " 没有更多未标注的样本单元。"))
    else:
        st.success(message)


def _display_agreement_value(value: float | None) -> str:
    return _l("Undefined", "未定义") if value is None else f"{value:.3f}"


def _render_agreement_results(connection, studies) -> None:
    study = _agreement_study_selector(studies, key="agreement_results_study")
    progress = get_agreement_progress(connection, study.study_id)
    progress_columns = st.columns(3)
    progress_columns[0].metric(
        _l(f"{study.annotator_a_id} completed", f"{study.annotator_a_id} 已完成"),
        f"{progress.annotator_a_count}/{progress.sample_size}",
    )
    progress_columns[1].metric(
        _l(f"{study.annotator_b_id} completed", f"{study.annotator_b_id} 已完成"),
        f"{progress.annotator_b_count}/{progress.sample_size}",
    )
    progress_columns[2].metric(
        _l("Doubly annotated", "双人均已标注"), progress.doubly_annotated_count
    )
    if not progress.is_complete:
        st.info(
            _l(
                "Agreement labels and statistics remain hidden until both "
                "annotators complete the full sample.",
                "在两位标注者完成全部样本前，一致性标签和统计结果将保持隐藏。",
            )
        )
        return
    report = build_agreement_report(connection, study.study_id)
    st.subheader(_l("Agreement statistics", "一致性统计"))
    st.caption(
        _l(
            "Cohen's kappa is reported as a descriptive reliability statistic. "
            "BRICS-Shift does not classify values as good, bad, or acceptable.",
            "Cohen's kappa 仅作为描述性信度统计呈现。BRICS-Shift 不会将数值自动判定为好、坏或可接受。",
        )
    )
    statistics_frame = pd.DataFrame(
        (
            {
                _l("Label set", "标签集"): _l(
                    result.dimension.title(),
                    "情态" if result.dimension == "modality" else "立场",
                ),
                _l("Doubly annotated units", "双人标注单元数"): result.doubly_annotated_units,
                _l("Agreement count", "一致数量"): result.agreement_count,
                _l("Raw agreement", "原始一致率"): _display_agreement_value(result.raw_agreement),
                "Cohen's kappa": _display_agreement_value(result.cohen_kappa),
            }
            for result in (report.modality, report.stance)
        )
    )
    st.dataframe(statistics_frame, use_container_width=True, hide_index=True)
    st.subheader(_l("Disagreements", "不一致项"))
    st.caption(
        _l(
            "Disagreements are displayed for researcher-led adjudication and are "
            "not resolved automatically.",
            "不一致项仅供研究者主导裁决，系统不会自动解决。",
        )
    )
    if not report.disagreements:
        st.info(_l("No modality or stance disagreements were found.", "未发现情态或立场标签不一致项。"))
        return
    disagreement_frame = pd.DataFrame(
        (
            {
                _l("Dimension", "维度"): _l(
                    row.dimension.title(),
                    "情态" if row.dimension == "modality" else "立场",
                ),
                _t("alignment_id"): row.alignment_id,
                "EN": row.english_text,
                "ZH": row.chinese_text,
                _l(f"{study.annotator_a_id} label", f"{study.annotator_a_id} 标签"): _display_annotation_value(row.annotator_a_label),
                _l(f"{study.annotator_b_id} label", f"{study.annotator_b_id} 标签"): _display_annotation_value(row.annotator_b_label),
            }
            for row in report.disagreements
        )
    )
    st.dataframe(disagreement_frame, use_container_width=True, hide_index=True)


def _render_agreement_page(connection) -> None:
    st.header(_t("page.agreement"))
    st.caption(_l("Reproducible independent double annotation", "可复现的独立双人标注"))
    st.caption(
        _l(
            "Alignment confidence is a heuristic quality score, not a calibrated probability.",
            "对齐置信度是启发式质量分数，而非经校准的概率。",
        )
    )
    mode = st.radio(
        _l("Mode", "模式"),
        ("create", "annotate", "results"),
        format_func=lambda value: {
            "create": _l("Create sample", "创建样本"),
            "annotate": _l("Independent annotation", "独立标注"),
            "results": _l("Results", "结果"),
        }[value],
        horizontal=True,
    )
    if mode == "create":
        _render_agreement_study_creation(connection)
        return
    studies = list_agreement_studies(connection)
    if not studies:
        st.info(_l("Create an agreement sample before opening this mode.", "请先创建一致性研究样本。"))
        return
    if mode == "annotate":
        _render_agreement_annotation(connection, studies)
    else:
        _render_agreement_results(connection, studies)


def _render_placeholder(page_name: str) -> None:
    st.header(_t(f"page.{page_name}"))
    st.info(
        _l(
            f"The {_t(f'page.{page_name}')} page is reserved for a later BRICS-Shift v0.1 step.",
            f"{_t(f'page.{page_name}')}页面保留给 BRICS-Shift v0.1 的后续步骤。",
        )
    )


def _distribution_frame(distribution) -> pd.DataFrame:
    return pd.DataFrame(
        (
            {
                _l("Category", "类别"): _display_annotation_value(row.category)
                if row.category in (*MODALITY_LABELS, *STANCE_LABELS)
                else _t(row.category),
                _l("Count", "数量"): row.count,
                _l("Percentage (%)", "百分比（%）"): (
                    "—" if row.percentage is None else f"{row.percentage:.2f}"
                ),
            }
            for row in distribution
        )
    )


def _crosstab_frame(crosstab: Crosstab) -> pd.DataFrame:
    records = []
    for group, counts in crosstab.rows:
        record = {crosstab.row_dimension: group}
        record.update(dict(zip(crosstab.columns, counts)))
        records.append(record)
    return pd.DataFrame(records, columns=(crosstab.row_dimension, *crosstab.columns))


def _render_crosstab(title: str, crosstab: Crosstab) -> None:
    st.markdown(f"**{title}**")
    frame = _crosstab_frame(crosstab)
    if frame.empty:
        st.info(_l("No included human annotations are available for this table.", "该表没有纳入的人类标注数据。"))
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_analysis_page(connection) -> None:
    st.header(_t("page.analysis"))
    st.caption(_l("Descriptive distributions from human annotations only", "仅基于人类标注的描述性分布"))

    annotator_id = st.text_input(
        _t("annotator_id"),
        value=st.session_state.get("annotator_id", ""),
        key="analysis_annotator_id",
        placeholder="e.g. researcher_1",
    ).strip()
    if not annotator_id:
        st.info(_l("Enter an annotator ID to calculate descriptive statistics.", "请输入标注者 ID 以计算描述统计。"))
        return

    control_columns = st.columns(3)
    threshold = control_columns[0].slider(
        _l("Eligibility confidence threshold", "可纳入对齐单元的置信度阈值"),
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
    )
    denominator_choice = control_columns[1].selectbox(
        _l("Percentage denominator", "百分比的分母"),
        (
            "all_annotated",
            "applicable",
        ),
        format_func=lambda value: {
            "all_annotated": _l("All eligible annotated alignments", "全部符合条件的已标注对齐单元"),
            "applicable": _l("Applicable alignments excluding N/A", "排除不适用（N/A）的适用对齐单元"),
        }[value],
    )
    grouping_choice = control_columns[2].selectbox(
        _l("Grouping", "分组方式"),
        ("year", "declaration", "period"),
        format_func=lambda value: {
            "year": _t("year"),
            "declaration": _l("Declaration", "宣言"),
            "period": _l("Historical period", "历史时期"),
        }[value],
    )
    denominator_mode = (
        denominator_choice
    )

    historical_periods = ()
    if grouping_choice == "period":
        period_text = st.text_area(
            _l("Historical period configuration", "历史时期配置"),
            placeholder=(
                _l(
                    "One inclusive range per line:\nperiod_name,start_year,end_year",
                    "每行一个闭区间：\n时期名称,起始年份,结束年份",
                )
            ),
            help=(
                _l(
                    "Names and year boundaries are supplied by the researcher. "
                    "Years outside configured ranges are reported as Unassigned.",
                    "名称和年份边界由研究者指定。配置区间外的年份会显示为“未分配”。",
                )
            ),
        )
        if period_text.strip():
            try:
                historical_periods = parse_period_configuration(period_text)
            except ValueError as error:
                st.error(str(error))
                return

    report = build_descriptive_statistics(
        connection,
        annotator_id=annotator_id,
        minimum_alignment_confidence=threshold,
        denominator_mode=denominator_mode,
        historical_periods=historical_periods,
    )
    overview = report.overview

    st.info(
        _l(
            "Alignment distributions use eligible alignments. Translation-shift "
            "distributions use only this annotator's human annotations on eligible, "
            "non-error alignments. Uncertain remains visible as its own category. "
            "Quarantined and error-flagged records are reported separately below.",
            "对齐分布使用符合条件的对齐单元。翻译偏移分布仅使用该标注者在符合条件、"
            "未标记错误的单元上的人类标注。不确定会作为独立类别保留；隔离和错误标记记录另行报告。",
        )
    )
    overview_metrics = (
        (_l("Eligible alignments", "符合条件的对齐单元"), overview.total_eligible_alignments),
        (_l("Annotated alignments (all)", "已标注对齐单元（全部）"), overview.total_annotated_alignments),
        (_l("Eligible annotated", "符合条件且已标注"), overview.eligible_annotated_alignments),
        (_l("Completion", "完成度"), f"{overview.annotation_completion_rate:.1f}%"),
        (_l("Possible alignment errors", "可能存在对齐错误"), overview.possible_alignment_error_count),
        (_l("Uncertain annotations", "不确定标注"), overview.uncertain_annotation_count),
        (_l("Quarantined alignments", "已隔离对齐单元"), overview.quarantined_alignment_count),
        (_l("Annotated but quarantined", "已标注但被隔离"), overview.annotated_quarantined_count),
    )
    for start in range(0, len(overview_metrics), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(
            columns, overview_metrics[start : start + 4]
        ):
            column.metric(label, value)
    st.caption(
        _l(
            "Completion denominator: eligible alignments. Numerator: eligible "
            "alignments annotated by the selected annotator.",
            "完成度分母：符合条件的对齐单元。分子：由所选标注者完成标注的符合条件单元。",
        )
    )
    st.caption(
        _l(
            "The overview's uncertain-annotation count covers all saved annotations "
            "for this annotator, including records separately reported as quarantined.",
            "概览中的不确定标注数包括该标注者全部已保存标注，其中也包括另行报告为隔离的记录。",
        )
    )

    st.subheader(_l("Alignment statistics", "对齐统计"))
    alignment_column, confidence_column = st.columns(2)
    with alignment_column:
        st.markdown(_l("**Alignment-type distribution**", "**对齐类型分布**"))
        st.caption(_l("Denominator: all eligible alignments.", "分母：全部符合条件的对齐单元。"))
        st.dataframe(
            _distribution_frame(report.alignment_type_distribution),
            use_container_width=True,
            hide_index=True,
        )
    with confidence_column:
        st.markdown(_l("**Alignment confidence-band distribution**", "**对齐置信度等级分布**"))
        st.caption(_l("Denominator: all eligible alignments.", "分母：全部符合条件的对齐单元。"))
        st.dataframe(
            _distribution_frame(report.confidence_band_distribution),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(_l("Translation-shift statistics", "翻译偏移统计"))
    modality_column, stance_column = st.columns(2)
    with modality_column:
        st.markdown(f"**{_t('modality_shift')}**")
        st.caption(
            _l(
                f"Denominator: {report.modality_denominator}. {report.modality_denominator_description}",
                f"分母：{report.modality_denominator}。{report.modality_denominator_description}",
            )
        )
        st.dataframe(
            _distribution_frame(report.modality_distribution),
            use_container_width=True,
            hide_index=True,
        )
    with stance_column:
        st.markdown(f"**{_t('stance_shift')}**")
        st.caption(
            _l(
                f"Denominator: {report.stance_denominator}. {report.stance_denominator_description}",
                f"分母：{report.stance_denominator}。{report.stance_denominator_description}",
            )
        )
        st.dataframe(
            _distribution_frame(report.stance_distribution),
            use_container_width=True,
            hide_index=True,
        )

    grouping_label = {
        "year": _t("year"),
        "declaration": _l("Declaration", "宣言"),
        "period": _l("Historical period", "历史时期"),
    }[grouping_choice]
    st.subheader(_l(f"Grouped by {grouping_label.lower()}", f"按{grouping_label}分组"))
    if grouping_choice == "year":
        grouped_modality = report.year_modality
        grouped_stance = report.year_stance
    elif grouping_choice == "declaration":
        grouped_modality = report.declaration_modality
        grouped_stance = report.declaration_stance
    elif not historical_periods:
        st.info(_l("Enter a historical period configuration to display this grouping.", "请输入历史时期配置以显示此分组。"))
        grouped_modality = None
        grouped_stance = None
    else:
        grouped_modality = report.period_modality
        grouped_stance = report.period_stance
    if grouped_modality is not None and grouped_stance is not None:
        _render_crosstab(_l("Grouped modality labels", "分组后的情态标签"), grouped_modality)
        _render_crosstab(_l("Grouped stance labels", "分组后的立场标签"), grouped_stance)

    st.subheader(_l("Required crosstabs", "交叉表"))
    with st.expander(_l("Year × labels", "年份 × 标签"), expanded=True):
        _render_crosstab(_l("Year × modality label", "年份 × 情态标签"), report.year_modality)
        _render_crosstab(_l("Year × stance label", "年份 × 立场标签"), report.year_stance)
    with st.expander(_l("Alignment type × labels", "对齐类型 × 标签"), expanded=True):
        _render_crosstab(
            _l("Alignment type × modality label", "对齐类型 × 情态标签"), report.alignment_type_modality
        )
        _render_crosstab(
            _l("Alignment type × stance label", "对齐类型 × 立场标签"), report.alignment_type_stance
        )

    st.warning(
        _l(
            "These are descriptive empirical distributions only. No significance "
            "tests or substantive linguistic interpretations are performed.",
            "此处仅报告描述性经验分布，不进行显著性检验或实质性语言学解释。",
        )
    )


def _render_export_page(connection) -> None:
    st.header(_t("page.export"))
    st.caption(_l("Pilot review and reproducible research exports", "试标审阅与可复现研究导出"))

    annotator_id = st.text_input(
        _t("annotator_id"),
        value=st.session_state.get("annotator_id", ""),
        key="export_annotator_id",
        placeholder="e.g. researcher_1",
    ).strip()
    if not annotator_id:
        st.info(_l("Enter an annotator ID to prepare a pilot export.", "请输入标注者 ID 以准备试标导出。"))
        return

    years = get_available_years(connection)
    first_row = st.columns(3)
    year_choice = first_row[0].selectbox(
        _t("year"), ("all", *years),
        format_func=lambda value: _t("all") if value == "all" else str(value),
        key="export_year",
    )
    alignment_type_choice = first_row[1].selectbox(
        _t("alignment_type"),
        ("all", "1:1", "1:2", "2:1", "2:2"),
        format_func=lambda value: _t("all") if value == "all" else value,
        key="export_type",
    )
    band_choice = first_row[2].selectbox(
        _t("confidence_band"),
        ("all", "high", "medium", "low"),
        format_func=lambda value: _t(value),
        key="export_band",
    )
    second_row = st.columns(3)
    error_choice = second_row[0].selectbox(
        _t("possible_alignment_error"),
        ("all", "flagged", "not_flagged"),
        format_func=lambda value: {
            "all": _t("all"),
            "flagged": _l("Flagged", "已标记"),
            "not_flagged": _l("Not flagged", "未标记"),
        }[value],
        key="export_error",
    )
    modality_choice = second_row[1].selectbox(
        _l("Modality label", "情态标签"),
        ("all", *MODALITY_LABELS),
        format_func=lambda value: _t("all") if value == "all" else _display_annotation_value(value),
        key="export_modality",
    )
    stance_choice = second_row[2].selectbox(
        _l("Stance label", "立场标签"),
        ("all", *STANCE_LABELS),
        format_func=lambda value: _t("all") if value == "all" else _display_annotation_value(value),
        key="export_stance",
    )
    notes_search = st.text_input(
        _l("Search notes", "搜索备注"),
        key="export_notes_search",
        placeholder=_l("Optional substring", "可选文本片段"),
    )

    csv_text = export_pilot_annotations_csv(
        connection,
        annotator_id,
        year=None if year_choice == "all" else int(year_choice),
        alignment_type=(
            None if alignment_type_choice == "all" else alignment_type_choice
        ),
        confidence_band=None if band_choice == "all" else band_choice,
        possible_alignment_error={
            "all": None,
            "flagged": True,
            "not_flagged": False,
        }[error_choice],
        modality_label=None if modality_choice == "all" else modality_choice,
        stance_label=None if stance_choice == "all" else stance_choice,
        notes_search=notes_search,
    )
    export_frame = pd.read_csv(StringIO(csv_text), keep_default_na=False)
    st.write(_l(f"Saved annotations selected: **{len(export_frame)}**", f"已选中的已保存标注：**{len(export_frame)}**"))
    st.download_button(
        _l("Download pilot CSV", "下载试标 CSV"),
        data=b"\xef\xbb\xbf" + csv_text.encode("utf-8"),
        file_name=f"brics_shift_pilot_{annotator_id}.csv",
        mime="text/csv",
        disabled=export_frame.empty,
        use_container_width=True,
    )
    if not export_frame.empty:
        st.dataframe(export_frame.head(100), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(_l("Reproducible research bundle", "可复现研究数据包"))
    st.caption(
        _l(
            "The bundle keeps all derived alignments and saved annotations in "
            "separate files. The joined research dataset applies the declared "
            "confidence and alignment-error policy. Raw source documents are not included.",
            "数据包将派生对齐数据与已保存标注分开存放。合并研究数据集应用所声明的"
            "置信度和对齐错误规则，但不包含原始全文文档。",
        )
    )
    confidence_threshold = st.number_input(
        _l("Research dataset confidence threshold", "研究数据集置信度阈值"),
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        format="%.2f",
        key="reproducible_export_threshold",
    )
    include_errors = st.checkbox(
        _l("Include possible alignment errors in joined research dataset", "在合并研究数据集中包含可能存在对齐错误的单元"),
        value=False,
        key="reproducible_export_include_errors",
    )
    config = ExportConfig(
        alignment_confidence_threshold=float(confidence_threshold),
        annotator_id=annotator_id,
        include_possible_alignment_errors=include_errors,
    )
    try:
        files = build_reproducible_export_files(connection, config)
    except TraceabilityError as error:
        st.error(_l(f"Export stopped because source traceability is incomplete: {error}", f"因来源可追溯性不完整，导出已停止：{error}"))
        return

    archive = BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for filename, content in files.items():
            bundle.writestr(filename, content.encode("utf-8"))
    st.download_button(
        _l("Download reproducible export bundle", "下载可复现导出数据包"),
        data=archive.getvalue(),
        file_name=f"brics_shift_v0.1_{annotator_id}.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.caption(
        _l(
            "Contains aligned_corpus.csv, annotations.csv, research_dataset.csv, "
            "research_dataset.jsonl, and export_metadata.json.",
            "包含 aligned_corpus.csv、annotations.csv、research_dataset.csv、"
            "research_dataset.jsonl 和 export_metadata.json。",
        )
    )


def main() -> None:
    st.set_page_config(page_title="BRICS-Shift", page_icon="🔎", layout="wide")
    st.sidebar.title("BRICS-Shift")
    st.sidebar.caption("Version 0.1")
    st.sidebar.selectbox(
        _t("interface_language"),
        LANGUAGE_OPTIONS,
        key="interface_language",
    )
    page = st.sidebar.radio(
        _t("navigation"),
        ("import", "corpus", "annotation", "agreement", "analysis", "export"),
        format_func=lambda value: _t(f"page.{value}"),
    )
    st.sidebar.divider()
    st.sidebar.caption(_t("research_principle"))

    connection = initialize_database(_database_path())
    try:
        if page == "import":
            _render_import_page(connection)
        elif page == "corpus":
            _render_corpus_page(connection)
        elif page == "annotation":
            _render_annotation_page(connection)
        elif page == "agreement":
            _render_agreement_page(connection)
        elif page == "analysis":
            _render_analysis_page(connection)
        elif page == "export":
            _render_export_page(connection)
        else:
            _render_placeholder(page)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
