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
                    "Language": language,
                    "Warning code": warning.code,
                    "Paragraph numbers": ", ".join(
                        str(number) for number in warning.paragraph_numbers
                    ),
                    "Message": warning.message,
                }
            )
    return rows


def _sentence_preview_frame(item) -> pd.DataFrame:
    return pd.DataFrame(
        (
            {
                "Paragraph": sentence.paragraph_no,
                "Sentence": sentence.sentence_no,
                "Sentence ID": sentence.sentence_id,
                "Text": sentence.text,
            }
            for sentence in item.sentences
        )
    )


def _render_prepared_import(prepared: PreparedCorpusPair) -> None:
    st.subheader("2. 预览段落、切句与对齐")
    st.caption(
        f"预览对象：{prepared.pair_id} · {prepared.year} · {prepared.title} · "
        f"{prepared.english.document.source_filename} / "
        f"{prepared.chinese.document.source_filename}"
    )
    comparison = prepared.structure_comparison
    metrics = st.columns(5)
    metrics[0].metric(
        "英文编号段落", comparison.english_numbered_paragraphs
    )
    metrics[1].metric(
        "中文编号段落", comparison.chinese_numbered_paragraphs
    )
    metrics[2].metric("匹配段落号", comparison.matching_paragraph_numbers)
    metrics[3].metric("英文句子", len(prepared.english.sentences))
    metrics[4].metric("中文句子", len(prepared.chinese.sentences))

    if comparison.missing_in_english:
        st.warning(f"英文缺少段落号：{list(comparison.missing_in_english)}")
    if comparison.missing_in_chinese:
        st.warning(f"中文缺少段落号：{list(comparison.missing_in_chinese)}")
    warning_rows = _paragraph_warning_rows(prepared)
    if warning_rows:
        st.warning("检测到段落结构警告。请在确认导入前逐项检查。")
        st.dataframe(pd.DataFrame(warning_rows), use_container_width=True, hide_index=True)
    else:
        st.success("未检测到段落编号异常。")

    if prepared.english.paragraph_result.heading_text:
        with st.expander("英文标题区文本"):
            st.text(prepared.english.paragraph_result.heading_text)
    if prepared.chinese.paragraph_result.heading_text:
        with st.expander("中文标题区文本"):
            st.text(prepared.chinese.paragraph_result.heading_text)

    english_tab, chinese_tab = st.tabs(("英文切句", "中文切句"))
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
    st.markdown("**确定性对齐预览**")
    alignment_metrics = st.columns(5)
    alignment_metrics[0].metric("对齐单元", len(alignment_result.alignments))
    alignment_metrics[1].metric("1:1", alignment_result.diagnostics.count_1_1)
    alignment_metrics[2].metric("1:2", alignment_result.diagnostics.count_1_2)
    alignment_metrics[3].metric("2:1", alignment_result.diagnostics.count_2_1)
    alignment_metrics[4].metric("2:2", alignment_result.diagnostics.count_2_2)
    st.caption(
        "Alignment confidence 是启发式质量分数，不是概率，也不会生成翻译偏移标签。"
    )
    alignment_frame = pd.DataFrame(
        (
            {
                "Alignment ID": unit.alignment_id,
                "Paragraph": unit.paragraph_no,
                "Type": unit.alignment_type,
                "Confidence": unit.normalized_confidence,
                "Band": unit.confidence_band,
                "English": unit.english_text,
                "Chinese": unit.chinese_text,
            }
            for unit in alignment_result.alignments
        )
    )
    if alignment_frame.empty:
        st.error("当前切句结果没有产生可保存的对齐单元。")
    else:
        st.dataframe(alignment_frame, use_container_width=True, hide_index=True)

    if alignment_result.diagnostics.failed_paragraphs:
        st.error("以下段落无法在 v0.1 的 1:1、1:2、2:1、2:2 规则下完成对齐：")
        st.dataframe(
            pd.DataFrame(
                (
                    {
                        "Paragraph": failure.paragraph_no,
                        "EN sentence count": failure.english_sentence_count,
                        "ZH sentence count": failure.chinese_sentence_count,
                        "Reason": failure.reason,
                    }
                    for failure in alignment_result.diagnostics.failed_paragraphs
                )
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_import_page(connection) -> None:
    st.header("Import Raw Text / 上传原始文本")
    st.caption(
        "上传一份英文官方文本和对应的中文官方译文。程序将先预览清洗、编号段落、"
        "规则切句和确定性对齐；只有再次确认后才写入数据库。"
    )
    flash = st.session_state.pop("import_flash", None)
    if flash:
        st.success(flash)

    st.subheader("1. 上传文件与填写元数据")
    metadata_columns = st.columns(3)
    pair_id = metadata_columns[0].text_input(
        "语料对 ID",
        placeholder="例如：2025_rio",
        help="只能使用字母、数字、下划线和连字符；系统会生成 _en 和 _zh 文档 ID。",
    )
    year = int(
        metadata_columns[1].number_input(
            "年份", min_value=1900, max_value=2200, value=2025, step=1
        )
    )
    title = metadata_columns[2].text_input(
        "宣言标题", placeholder="例如：Rio Declaration"
    )
    file_columns = st.columns(2)
    english_upload = file_columns[0].file_uploader(
        "English official text (.txt)", type=("txt",), key="import_en_file"
    )
    chinese_upload = file_columns[1].file_uploader(
        "中文官方译文 (.txt)", type=("txt",), key="import_zh_file"
    )
    url_columns = st.columns(2)
    english_url = url_columns[0].text_input(
        "英文来源 URL（可选）", key="import_en_url"
    )
    chinese_url = url_columns[1].text_input(
        "中文来源 URL（可选）", key="import_zh_url"
    )
    st.info(
        "文件必须是 UTF-8 TXT，并以阿拉伯数字段落号组织。英文切句保护常见缩写、"
        "首字母缩写和小数；仍须由研究者在预览表中检查句界。"
    )
    prepare_clicked = st.button(
        "生成切句与对齐预览",
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
    st.subheader("3. 确认写入数据库")
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
        st.error("文件或元数据已在预览后发生变化，请重新生成预览后再导入。")
    reviewed = st.checkbox(
        "我已检查段落警告、全部句界和无法对齐的段落，并确认保存当前预览结果。",
        key=f"confirm_import_{prepared.pair_id}",
    )
    st.warning(
        "确认后会保存原始 TXT 到 data/raw，并把文档、段落、句子和成功的对齐单元"
        "写入 SQLite。已有记录和不同内容的同名原始文件不会被覆盖。"
    )
    if not st.button(
        "确认导入并进入语料库",
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
        st.error(f"导入失败，未覆盖已有数据：{error}")
        return
    st.session_state.pop("prepared_corpus_import", None)
    st.session_state["import_flash"] = (
        f"导入完成：{summary.documents} 个文档、{summary.paragraphs} 个段落、"
        f"{summary.sentences} 个句子、{summary.alignments} 个对齐单元。"
        "现在可进入 Corpus 检查，并在 Annotation 开始人工标注。"
    )
    st.rerun()


def _render_corpus_page(connection) -> None:
    st.header("Corpus")
    st.caption("Basic corpus structure and alignment overview")

    years = get_available_years(connection)
    selected_year = st.selectbox("Filter by year", ("All years", *years))
    year = None if selected_year == "All years" else int(selected_year)
    overview = get_corpus_overview(connection, year=year)

    metrics = (
        ("Documents", overview.document_count),
        ("Years represented", ", ".join(map(str, overview.years)) or "—"),
        ("English sentences", overview.english_sentence_count),
        ("Chinese sentences", overview.chinese_sentence_count),
        ("Aligned units", overview.aligned_unit_count),
        ("1:1", overview.count_1_1),
        ("1:2", overview.count_1_2),
        ("2:1", overview.count_2_1),
        ("2:2", overview.count_2_2),
        ("High confidence", overview.high_confidence_count),
        ("Medium confidence", overview.medium_confidence_count),
        ("Low confidence", overview.low_confidence_count),
    )
    for start in range(0, len(metrics), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, metrics[start : start + 4]):
            column.metric(label, value)

    st.subheader("Alignment preview")
    preview = list_alignment_preview(connection, year=year, limit=100)
    if not preview:
        st.info("No alignments are available for this filter.")
        return

    frame = pd.DataFrame(preview).rename(
        columns={
            "year": "Year",
            "document_title": "Document title",
            "paragraph_no": "Paragraph",
            "alignment_id": "Alignment ID",
            "alignment_type": "Type",
            "alignment_confidence": "Confidence",
            "alignment_confidence_band": "Band",
            "en_text": "English",
            "zh_text": "Chinese",
        }
    )
    frame["Confidence"] = frame["Confidence"].round(3)
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
        missing.append("Modality Shift")
    if stance_label == SELECT_PROMPT:
        missing.append("Stance Shift")
    if annotator_confidence == SELECT_PROMPT:
        missing.append("Annotator Confidence")
    if not guideline_version.strip():
        missing.append("Guideline version")
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
    st.header("Annotation")
    st.caption("Human translation-shift annotation of aligned EN-ZH units")
    st.caption(
        "Alignment confidence is a heuristic quality score, not a calibrated "
        "probability. It never supplies a translation-shift label."
    )
    annotation_mode = st.radio(
        "Mode",
        ("Annotate", "Review annotated"),
        horizontal=True,
        help="Review mode loads only this annotator's saved pilot examples.",
    )

    annotator_id = st.sidebar.text_input(
        "Annotator ID",
        key="annotator_id",
        placeholder="e.g. researcher_1",
    ).strip()
    guideline_version = st.sidebar.text_input(
        "Guideline version",
        key="guideline_version",
        placeholder="e.g. guidelines-v1.0",
        help="The exact frozen guideline version is stored with every annotation.",
    ).strip()
    minimum_confidence = st.sidebar.slider(
        "Annotation confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
        help="Alignments below this automatic alignment-confidence score are quarantined.",
    )
    show_all = st.sidebar.toggle(
        "Show all alignments",
        value=False,
        help="Includes low-confidence and alignment-error-flagged units.",
    )

    st.sidebar.subheader("Pilot filters")
    years = get_available_years(connection)
    year_choice = st.sidebar.selectbox(
        "Year",
        ("All", *years),
        key="annotation_filter_year",
    )
    alignment_type_choice = st.sidebar.selectbox(
        "Alignment type",
        ("All", "1:1", "1:2", "2:1", "2:2"),
        key="annotation_filter_type",
    )
    band_choice = st.sidebar.selectbox(
        "Alignment confidence band",
        ("All", "high", "medium", "low"),
        key="annotation_filter_band",
    )
    if annotation_mode == "Review annotated":
        annotation_status = "annotated"
        st.sidebar.caption("Annotation status: annotated")
    else:
        status_choice = st.sidebar.selectbox(
            "Annotation status",
            ("Unannotated", "All", "Annotated"),
            key="annotation_filter_status",
        )
        annotation_status = status_choice.lower()
    error_choice = st.sidebar.selectbox(
        "Possible alignment error",
        ("All", "Flagged", "Not flagged"),
        key="annotation_filter_error",
    )
    modality_choice = st.sidebar.selectbox(
        "Modality label",
        ("All", *MODALITY_LABELS),
        key="annotation_filter_modality",
    )
    stance_choice = st.sidebar.selectbox(
        "Stance label",
        ("All", *STANCE_LABELS),
        key="annotation_filter_stance",
    )
    notes_search = st.sidebar.text_input(
        "Search notes",
        key="annotation_notes_search",
        placeholder="Review mode: search note text",
    )

    if not annotator_id:
        st.info("Enter an annotator ID in the sidebar to open the annotation queue.")
        return

    queue_threshold = None if show_all else minimum_confidence
    filter_year = None if year_choice == "All" else int(year_choice)
    filter_alignment_type = (
        None if alignment_type_choice == "All" else alignment_type_choice
    )
    filter_band = None if band_choice == "All" else band_choice
    filter_error = {
        "All": None,
        "Flagged": True,
        "Not flagged": False,
    }[error_choice]
    filter_modality = None if modality_choice == "All" else modality_choice
    filter_stance = None if stance_choice == "All" else stance_choice
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
        f"Annotated: **{progress.annotated_alignments} / "
        f"{progress.total_alignments}**"
    )
    st.progress(
        progress.annotated_alignments / progress.total_alignments
        if progress.total_alignments
        else 0.0,
        text=f"Completion: {progress.completion_percentage:.1f}%",
    )

    flash_message = st.session_state.pop("annotation_flash", None)
    if flash_message:
        st.success(flash_message)

    if not queue:
        st.info("No alignments are available under the current queue rules.")
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

    st.caption(f"Queue item {current_index + 1} of {len(queue)}")
    metadata_columns = st.columns(4)
    metadata_columns[0].metric("Year", alignment["year"])
    metadata_columns[1].metric("Paragraph", alignment["paragraph_no"])
    metadata_columns[2].metric("Alignment type", alignment["alignment_type"])
    metadata_columns[3].metric(
        "Alignment confidence", f"{alignment['alignment_confidence']:.3f}"
    )
    st.write(f"**Document title:** {alignment['document_title']}")
    st.write(f"**Alignment ID:** `{alignment['alignment_id']}`")
    st.write(f"**Confidence band:** {alignment['alignment_confidence_band']}")
    if alignment["possible_alignment_error"]:
        st.warning("This alignment has been flagged for possible alignment error.")

    source_column, target_column = st.columns(2)
    with source_column:
        with st.container(border=True):
            st.subheader("SOURCE TEXT — English")
            st.write(alignment["en_text"])
    with target_column:
        with st.container(border=True):
            st.subheader("TARGET TEXT — Chinese")
            st.write(alignment["zh_text"])

    if existing:
        st.info(
            "An existing annotation by this annotator has been loaded. "
            "Saving will perform an explicit update."
        )

    widget_suffix = f"{annotator_id}_{alignment['alignment_id']}"
    modality_options = (SELECT_PROMPT, *MODALITY_LABELS)
    stance_options = (SELECT_PROMPT, *STANCE_LABELS)
    confidence_options = (SELECT_PROMPT, *ANNOTATOR_CONFIDENCE_VALUES)
    modality_label = st.selectbox(
        "Modality Shift",
        modality_options,
        index=_select_index(
            modality_options, existing["modality_label"] if existing else None
        ),
        key=f"modality_{widget_suffix}",
    )
    stance_label = st.selectbox(
        "Stance Shift",
        stance_options,
        index=_select_index(
            stance_options, existing["stance_label"] if existing else None
        ),
        key=f"stance_{widget_suffix}",
    )
    annotator_confidence = st.selectbox(
        "Annotator Confidence",
        confidence_options,
        index=_select_index(
            confidence_options,
            existing["annotator_confidence"] if existing else None,
        ),
        key=f"annotator_confidence_{widget_suffix}",
    )
    notes = st.text_area(
        "Notes",
        value=existing["notes"] if existing else "",
        key=f"notes_{widget_suffix}",
    )
    possible_alignment_error = st.checkbox(
        "Possible alignment error",
        value=existing["possible_alignment_error"] if existing else False,
        key=f"alignment_error_{widget_suffix}",
        help=(
            "The annotator suspects that the English and Chinese texts may not "
            "be correctly aligned. This does not change either shift label."
        ),
    )

    button_columns = st.columns(4)
    previous_clicked = button_columns[0].button(
        "Previous", disabled=current_index == 0, use_container_width=True
    )
    save_clicked = button_columns[1].button("Save", use_container_width=True)
    save_next_clicked = button_columns[2].button(
        "Save & Next", use_container_width=True
    )
    next_clicked = button_columns[3].button(
        "Next", disabled=current_index == len(queue) - 1, use_container_width=True
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
        st.error("Select or enter: " + ", ".join(missing_fields) + ".")
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

    message = f"Annotation {action} for {alignment['alignment_id']}."
    if save_next_clicked:
        next_id = _next_unannotated_id(queue, current_index)
        if next_id is not None:
            st.session_state["annotation_current_id"] = next_id
            st.session_state["annotation_flash"] = message
            st.rerun()
        st.success(message + " No further unannotated alignment is available.")
    else:
        st.success(message)


def _agreement_study_selector(studies, *, key: str):
    study_by_id = {study.study_id: study for study in studies}
    selected_id = st.selectbox(
        "Agreement study",
        tuple(study_by_id),
        format_func=lambda value: (
            f"{study_by_id[value].study_name} ({value})"
        ),
        key=key,
    )
    return study_by_id[selected_id]


def _render_agreement_study_creation(connection) -> None:
    st.subheader("Create a reproducible agreement sample")
    st.caption(
        "Sampling is fixed-seed and proportional within the selected strata. "
        "The definition and exact sample membership are stored in SQLite."
    )
    with st.form("create_agreement_study"):
        study_id = st.text_input("Study ID", placeholder="e.g. AGREE_PILOT_01")
        study_name = st.text_input(
            "Study name", placeholder="e.g. Pilot double annotation"
        )
        annotator_columns = st.columns(2)
        annotator_a_id = annotator_columns[0].text_input(
            "Annotator A ID", placeholder="e.g. researcher_1"
        )
        annotator_b_id = annotator_columns[1].text_input(
            "Annotator B ID", placeholder="e.g. researcher_2"
        )
        sampling_columns = st.columns(3)
        sample_size = sampling_columns[0].number_input(
            "Sample size", min_value=1, value=100, step=1
        )
        random_seed = sampling_columns[1].number_input(
            "Random seed", min_value=0, max_value=2_147_483_647,
            value=2025, step=1
        )
        minimum_confidence = sampling_columns[2].number_input(
            "Minimum alignment confidence", min_value=0.0, max_value=1.0,
            value=0.70, step=0.05, format="%.2f"
        )
        option_columns = st.columns(3)
        stratify_year = option_columns[0].checkbox("Stratify by year")
        stratify_type = option_columns[1].checkbox("Stratify by alignment type")
        include_errors = option_columns[2].checkbox(
            "Include possible alignment errors"
        )
        submitted = st.form_submit_button(
            "Create agreement study", use_container_width=True
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
        f"Created {study.study_id} with {study.sample_size} sampled alignments."
    )
    with st.expander("Stored sample definition", expanded=True):
        st.json(dict(study.sample_definition))


def _render_agreement_annotation(connection, studies) -> None:
    study = _agreement_study_selector(studies, key="agreement_annotation_study")
    st.info(
        "Independent annotation mode: only your own saved labels are loaded. "
        "The other participant's labels remain hidden."
    )
    input_columns = st.columns(3)
    annotator_id = input_columns[0].text_input(
        "Annotator ID",
        value=st.session_state.get("annotator_id", ""),
        key="agreement_annotator_id",
    ).strip()
    guideline_version = input_columns[1].text_input(
        "Guideline version",
        value=st.session_state.get("guideline_version", ""),
        key="agreement_guideline_version",
    ).strip()
    status = input_columns[2].selectbox(
        "Queue status", ("Unannotated", "All", "Annotated"),
        key="agreement_annotation_status"
    ).lower()
    if not annotator_id:
        st.info("Enter your assigned annotator ID to open the sample.")
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
    st.write(f"Your progress: **{own_count} / {progress.sample_size}**")
    st.progress(
        own_count / progress.sample_size if progress.sample_size else 0.0,
        text=f"Completion: {(100 * own_count / progress.sample_size):.1f}%",
    )
    flash_message = st.session_state.pop("agreement_annotation_flash", None)
    if flash_message:
        st.success(flash_message)
    if not queue:
        st.info("No sample units are available under this queue status.")
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
        f"Sample item {alignment['sample_order']} of {study.sample_size} "
        f"(queue item {current_index + 1} of {len(queue)})"
    )
    metadata_columns = st.columns(4)
    metadata_columns[0].metric("Year", alignment["year"])
    metadata_columns[1].metric("Paragraph", alignment["paragraph_no"])
    metadata_columns[2].metric("Alignment type", alignment["alignment_type"])
    metadata_columns[3].metric(
        "Alignment confidence", f"{alignment['alignment_confidence']:.3f}"
    )
    st.write(f"**Document title:** {alignment['document_title']}")
    st.write(f"**Alignment ID:** `{alignment['alignment_id']}`")
    source_column, target_column = st.columns(2)
    with source_column:
        with st.container(border=True):
            st.subheader("SOURCE TEXT — English")
            st.write(alignment["en_text"])
    with target_column:
        with st.container(border=True):
            st.subheader("TARGET TEXT — Chinese")
            st.write(alignment["zh_text"])
    if existing:
        st.info("Your existing record has been loaded for explicit editing.")

    widget_suffix = f"agreement_{study.study_id}_{annotator_id}_{alignment['alignment_id']}"
    modality_options = (SELECT_PROMPT, *MODALITY_LABELS)
    stance_options = (SELECT_PROMPT, *STANCE_LABELS)
    confidence_options = (SELECT_PROMPT, *ANNOTATOR_CONFIDENCE_VALUES)
    modality_label = st.selectbox(
        "Modality Shift", modality_options,
        index=_select_index(
            modality_options, existing["modality_label"] if existing else None
        ),
        key=f"modality_{widget_suffix}",
    )
    stance_label = st.selectbox(
        "Stance Shift", stance_options,
        index=_select_index(
            stance_options, existing["stance_label"] if existing else None
        ),
        key=f"stance_{widget_suffix}",
    )
    annotator_confidence = st.selectbox(
        "Annotator Confidence", confidence_options,
        index=_select_index(
            confidence_options,
            existing["annotator_confidence"] if existing else None,
        ),
        key=f"confidence_{widget_suffix}",
    )
    notes = st.text_area(
        "Notes", value=existing["notes"] if existing else "",
        key=f"notes_{widget_suffix}",
    )
    possible_alignment_error = st.checkbox(
        "Possible alignment error",
        value=existing["possible_alignment_error"] if existing else False,
        key=f"error_{widget_suffix}",
    )

    buttons = st.columns(4)
    previous_clicked = buttons[0].button(
        "Previous", disabled=current_index == 0, use_container_width=True,
        key="agreement_previous"
    )
    save_clicked = buttons[1].button(
        "Save", use_container_width=True, key="agreement_save"
    )
    save_next_clicked = buttons[2].button(
        "Save & Next", use_container_width=True, key="agreement_save_next"
    )
    next_clicked = buttons[3].button(
        "Next", disabled=current_index == len(queue) - 1,
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
        st.error("Select or enter: " + ", ".join(missing) + ".")
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
    message = f"Independent annotation {action} for {alignment['alignment_id']}."
    if save_next_clicked:
        next_id = _next_unannotated_id(queue, current_index)
        if next_id is not None:
            st.session_state["agreement_current_id"] = next_id
            st.session_state["agreement_annotation_flash"] = message
            st.rerun()
        st.success(message + " No further unannotated sample unit is available.")
    else:
        st.success(message)


def _display_agreement_value(value: float | None) -> str:
    return "Undefined" if value is None else f"{value:.3f}"


def _render_agreement_results(connection, studies) -> None:
    study = _agreement_study_selector(studies, key="agreement_results_study")
    progress = get_agreement_progress(connection, study.study_id)
    progress_columns = st.columns(3)
    progress_columns[0].metric(
        f"{study.annotator_a_id} completed", f"{progress.annotator_a_count}/{progress.sample_size}"
    )
    progress_columns[1].metric(
        f"{study.annotator_b_id} completed", f"{progress.annotator_b_count}/{progress.sample_size}"
    )
    progress_columns[2].metric(
        "Doubly annotated", progress.doubly_annotated_count
    )
    if not progress.is_complete:
        st.info(
            "Agreement labels and statistics remain hidden until both "
            "annotators complete the full sample."
        )
        return
    report = build_agreement_report(connection, study.study_id)
    st.subheader("Agreement statistics")
    st.caption(
        "Cohen's kappa is reported as a descriptive reliability statistic. "
        "BRICS-Shift does not classify values as good, bad, or acceptable."
    )
    statistics_frame = pd.DataFrame(
        (
            {
                "Label set": result.dimension.title(),
                "Doubly annotated units": result.doubly_annotated_units,
                "Agreement count": result.agreement_count,
                "Raw agreement": _display_agreement_value(result.raw_agreement),
                "Cohen's kappa": _display_agreement_value(result.cohen_kappa),
            }
            for result in (report.modality, report.stance)
        )
    )
    st.dataframe(statistics_frame, use_container_width=True, hide_index=True)
    st.subheader("Disagreements")
    st.caption(
        "Disagreements are displayed for researcher-led adjudication and are "
        "not resolved automatically."
    )
    if not report.disagreements:
        st.info("No modality or stance disagreements were found.")
        return
    disagreement_frame = pd.DataFrame(
        (
            {
                "Dimension": row.dimension.title(),
                "Alignment ID": row.alignment_id,
                "EN": row.english_text,
                "ZH": row.chinese_text,
                f"{study.annotator_a_id} label": row.annotator_a_label,
                f"{study.annotator_b_id} label": row.annotator_b_label,
            }
            for row in report.disagreements
        )
    )
    st.dataframe(disagreement_frame, use_container_width=True, hide_index=True)


def _render_agreement_page(connection) -> None:
    st.header("Agreement Study")
    st.caption("Reproducible independent double annotation")
    st.caption(
        "Alignment confidence is a heuristic quality score, not a calibrated "
        "probability."
    )
    mode = st.radio(
        "Mode", ("Create sample", "Independent annotation", "Results"),
        horizontal=True,
    )
    if mode == "Create sample":
        _render_agreement_study_creation(connection)
        return
    studies = list_agreement_studies(connection)
    if not studies:
        st.info("Create an agreement sample before opening this mode.")
        return
    if mode == "Independent annotation":
        _render_agreement_annotation(connection, studies)
    else:
        _render_agreement_results(connection, studies)


def _render_placeholder(page_name: str) -> None:
    st.header(page_name)
    st.info(f"The {page_name} page is reserved for a later BRICS-Shift v0.1 step.")


def _distribution_frame(distribution) -> pd.DataFrame:
    return pd.DataFrame(
        (
            {
                "Category": row.category,
                "Count": row.count,
                "Percentage (%)": (
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
        st.info("No included human annotations are available for this table.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_analysis_page(connection) -> None:
    st.header("Analysis")
    st.caption("Descriptive distributions from human annotations only")

    annotator_id = st.text_input(
        "Annotator ID",
        value=st.session_state.get("annotator_id", ""),
        key="analysis_annotator_id",
        placeholder="e.g. researcher_1",
    ).strip()
    if not annotator_id:
        st.info("Enter an annotator ID to calculate descriptive statistics.")
        return

    control_columns = st.columns(3)
    threshold = control_columns[0].slider(
        "Eligibility confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
    )
    denominator_choice = control_columns[1].selectbox(
        "Percentage denominator",
        (
            "All eligible annotated alignments",
            "Applicable alignments excluding N/A",
        ),
    )
    grouping_choice = control_columns[2].selectbox(
        "Grouping",
        ("Year", "Declaration", "Historical period"),
    )
    denominator_mode = (
        "all_annotated"
        if denominator_choice == "All eligible annotated alignments"
        else "applicable"
    )

    historical_periods = ()
    if grouping_choice == "Historical period":
        period_text = st.text_area(
            "Historical period configuration",
            placeholder=(
                "One inclusive range per line:\n"
                "period_name,start_year,end_year"
            ),
            help=(
                "Names and year boundaries are supplied by the researcher. "
                "Years outside configured ranges are reported as Unassigned."
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
        "Alignment distributions use eligible alignments. Translation-shift "
        "distributions use only this annotator's human annotations on eligible, "
        "non-error alignments. Uncertain remains visible as its own category. "
        "Quarantined and error-flagged records are reported separately below."
    )
    overview_metrics = (
        ("Eligible alignments", overview.total_eligible_alignments),
        ("Annotated alignments (all)", overview.total_annotated_alignments),
        ("Eligible annotated", overview.eligible_annotated_alignments),
        ("Completion", f"{overview.annotation_completion_rate:.1f}%"),
        ("Possible alignment errors", overview.possible_alignment_error_count),
        ("Uncertain annotations", overview.uncertain_annotation_count),
        ("Quarantined alignments", overview.quarantined_alignment_count),
        ("Annotated but quarantined", overview.annotated_quarantined_count),
    )
    for start in range(0, len(overview_metrics), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(
            columns, overview_metrics[start : start + 4]
        ):
            column.metric(label, value)
    st.caption(
        "Completion denominator: eligible alignments. Numerator: eligible "
        "alignments annotated by the selected annotator."
    )
    st.caption(
        "The overview's uncertain-annotation count covers all saved annotations "
        "for this annotator, including records separately reported as quarantined."
    )

    st.subheader("Alignment statistics")
    alignment_column, confidence_column = st.columns(2)
    with alignment_column:
        st.markdown("**Alignment-type distribution**")
        st.caption("Denominator: all eligible alignments.")
        st.dataframe(
            _distribution_frame(report.alignment_type_distribution),
            use_container_width=True,
            hide_index=True,
        )
    with confidence_column:
        st.markdown("**Alignment confidence-band distribution**")
        st.caption("Denominator: all eligible alignments.")
        st.dataframe(
            _distribution_frame(report.confidence_band_distribution),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Translation-shift statistics")
    modality_column, stance_column = st.columns(2)
    with modality_column:
        st.markdown("**Modality Shift**")
        st.caption(
            f"Denominator: {report.modality_denominator}. "
            f"{report.modality_denominator_description}"
        )
        st.dataframe(
            _distribution_frame(report.modality_distribution),
            use_container_width=True,
            hide_index=True,
        )
    with stance_column:
        st.markdown("**Stance Shift**")
        st.caption(
            f"Denominator: {report.stance_denominator}. "
            f"{report.stance_denominator_description}"
        )
        st.dataframe(
            _distribution_frame(report.stance_distribution),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(f"Grouped by {grouping_choice.lower()}")
    if grouping_choice == "Year":
        grouped_modality = report.year_modality
        grouped_stance = report.year_stance
    elif grouping_choice == "Declaration":
        grouped_modality = report.declaration_modality
        grouped_stance = report.declaration_stance
    elif not historical_periods:
        st.info("Enter a historical period configuration to display this grouping.")
        grouped_modality = None
        grouped_stance = None
    else:
        grouped_modality = report.period_modality
        grouped_stance = report.period_stance
    if grouped_modality is not None and grouped_stance is not None:
        _render_crosstab("Grouped modality labels", grouped_modality)
        _render_crosstab("Grouped stance labels", grouped_stance)

    st.subheader("Required crosstabs")
    with st.expander("Year × labels", expanded=True):
        _render_crosstab("Year × modality label", report.year_modality)
        _render_crosstab("Year × stance label", report.year_stance)
    with st.expander("Alignment type × labels", expanded=True):
        _render_crosstab(
            "Alignment type × modality label", report.alignment_type_modality
        )
        _render_crosstab(
            "Alignment type × stance label", report.alignment_type_stance
        )

    st.warning(
        "These are descriptive empirical distributions only. No significance "
        "tests or substantive linguistic interpretations are performed."
    )


def _render_export_page(connection) -> None:
    st.header("Export")
    st.caption("Pilot review and reproducible research exports")

    annotator_id = st.text_input(
        "Annotator ID",
        value=st.session_state.get("annotator_id", ""),
        key="export_annotator_id",
        placeholder="e.g. researcher_1",
    ).strip()
    if not annotator_id:
        st.info("Enter an annotator ID to prepare a pilot export.")
        return

    years = get_available_years(connection)
    first_row = st.columns(3)
    year_choice = first_row[0].selectbox(
        "Year", ("All", *years), key="export_year"
    )
    alignment_type_choice = first_row[1].selectbox(
        "Alignment type",
        ("All", "1:1", "1:2", "2:1", "2:2"),
        key="export_type",
    )
    band_choice = first_row[2].selectbox(
        "Alignment confidence band",
        ("All", "high", "medium", "low"),
        key="export_band",
    )
    second_row = st.columns(3)
    error_choice = second_row[0].selectbox(
        "Possible alignment error",
        ("All", "Flagged", "Not flagged"),
        key="export_error",
    )
    modality_choice = second_row[1].selectbox(
        "Modality label",
        ("All", *MODALITY_LABELS),
        key="export_modality",
    )
    stance_choice = second_row[2].selectbox(
        "Stance label",
        ("All", *STANCE_LABELS),
        key="export_stance",
    )
    notes_search = st.text_input(
        "Search notes",
        key="export_notes_search",
        placeholder="Optional substring",
    )

    csv_text = export_pilot_annotations_csv(
        connection,
        annotator_id,
        year=None if year_choice == "All" else int(year_choice),
        alignment_type=(
            None if alignment_type_choice == "All" else alignment_type_choice
        ),
        confidence_band=None if band_choice == "All" else band_choice,
        possible_alignment_error={
            "All": None,
            "Flagged": True,
            "Not flagged": False,
        }[error_choice],
        modality_label=None if modality_choice == "All" else modality_choice,
        stance_label=None if stance_choice == "All" else stance_choice,
        notes_search=notes_search,
    )
    export_frame = pd.read_csv(StringIO(csv_text), keep_default_na=False)
    st.write(f"Saved annotations selected: **{len(export_frame)}**")
    st.download_button(
        "Download pilot CSV",
        data=b"\xef\xbb\xbf" + csv_text.encode("utf-8"),
        file_name=f"brics_shift_pilot_{annotator_id}.csv",
        mime="text/csv",
        disabled=export_frame.empty,
        use_container_width=True,
    )
    if not export_frame.empty:
        st.dataframe(export_frame.head(100), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Reproducible research bundle")
    st.caption(
        "The bundle keeps all derived alignments and saved annotations in "
        "separate files. The joined research dataset applies the declared "
        "confidence and alignment-error policy. Raw source documents are not included."
    )
    confidence_threshold = st.number_input(
        "Research dataset confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        format="%.2f",
        key="reproducible_export_threshold",
    )
    include_errors = st.checkbox(
        "Include possible alignment errors in joined research dataset",
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
        st.error(f"Export stopped because source traceability is incomplete: {error}")
        return

    archive = BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for filename, content in files.items():
            bundle.writestr(filename, content.encode("utf-8"))
    st.download_button(
        "Download reproducible export bundle",
        data=archive.getvalue(),
        file_name=f"brics_shift_v0.1_{annotator_id}.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.caption(
        "Contains aligned_corpus.csv, annotations.csv, research_dataset.csv, "
        "research_dataset.jsonl, and export_metadata.json."
    )


def main() -> None:
    st.set_page_config(page_title="BRICS-Shift", page_icon="🔎", layout="wide")
    st.sidebar.title("BRICS-Shift")
    st.sidebar.caption("Version 0.1")
    page = st.sidebar.radio(
        "Navigation",
        ("Import", "Corpus", "Annotation", "Agreement", "Analysis", "Export"),
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "The software organizes translation evidence. Humans interpret "
        "translation shifts."
    )

    connection = initialize_database(_database_path())
    try:
        if page == "Import":
            _render_import_page(connection)
        elif page == "Corpus":
            _render_corpus_page(connection)
        elif page == "Annotation":
            _render_annotation_page(connection)
        elif page == "Agreement":
            _render_agreement_page(connection)
        elif page == "Analysis":
            _render_analysis_page(connection)
        elif page == "Export":
            _render_export_page(connection)
        else:
            _render_placeholder(page)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
