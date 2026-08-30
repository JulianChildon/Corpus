"""Tests for SQLite persistence and research-data traceability."""

from datetime import datetime, timezone
import csv
from io import StringIO
import json
import sqlite3

import pytest

from brics_shift.alignment import (
    AlignmentConfig,
    AlignmentUnit,
    SentenceRecord,
    align_documents,
)
from brics_shift.cleaning import ImportedDocument
from brics_shift.database import (
    DuplicateAnnotationError,
    get_alignment,
    get_annotation,
    get_annotation_progress,
    get_available_years,
    get_corpus_overview,
    get_next_unannotated_alignment,
    initialize_database,
    insert_alignments,
    insert_document,
    insert_paragraphs,
    insert_sentences,
    list_alignment_preview,
    list_alignments_for_annotation,
    save_annotation,
    update_annotation,
)
from brics_shift.paragraphs import ParsedParagraph
from brics_shift.export import (
    ALIGNED_CORPUS_COLUMNS,
    ANNOTATION_EXPORT_COLUMNS,
    ExportConfig,
    PILOT_EXPORT_COLUMNS,
    TraceabilityError,
    build_reproducible_export_files,
    export_aligned_corpus_csv,
    export_annotations_csv,
    export_pilot_annotations_csv,
    export_research_dataset_csv,
    export_research_dataset_jsonl,
)


FIXED_TIME = datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def database():
    connection = initialize_database(":memory:")
    yield connection
    connection.close()


def imported_document(language: str) -> ImportedDocument:
    return ImportedDocument(
        document_id=f"2025_rio_{language.lower()}",
        year=2025,
        title="Rio Declaration",
        language=language,  # type: ignore[arg-type]
        source_filename=f"2025_rio_{language.lower()}.txt",
        source_url="https://example.test/official",
        imported_at=FIXED_TIME,
        cleaned_text=(
            "1. We reaffirm implementation of the 2030 Agenda."
            if language == "EN"
            else "1. 我们重申落实《2030年议程》。"
        ),
    )


def insert_documents_and_paragraphs(connection) -> None:
    for language in ("EN", "ZH"):
        document = imported_document(language)
        insert_document(
            connection,
            document,
            checksum=("a" if language == "EN" else "b") * 64,
            preprocessing_version="clean-v0.1",
        )
        paragraph_text = (
            "We reaffirm implementation of the 2030 Agenda."
            if language == "EN"
            else "我们重申落实《2030年议程》。"
        )
        insert_paragraphs(
            connection,
            [
                ParsedParagraph(
                    document_id=document.document_id,
                    paragraph_no=1,
                    raw_paragraph_text=paragraph_text,
                    cleaned_paragraph_text=paragraph_text,
                    original_order=1,
                )
            ],
        )


def sentence_pair() -> tuple[SentenceRecord, SentenceRecord]:
    return (
        SentenceRecord(
            sentence_id="EN_2025_P001_S01",
            document_id="2025_rio_en",
            language="EN",
            paragraph_no=1,
            sentence_no=1,
            text="We reaffirm implementation of the 2030 Agenda.",
        ),
        SentenceRecord(
            sentence_id="ZH_2025_P001_S01",
            document_id="2025_rio_zh",
            language="ZH",
            paragraph_no=1,
            sentence_no=1,
            text="我们重申落实《2030年议程》。",
        ),
    )


def populate_alignment(connection) -> str:
    insert_documents_and_paragraphs(connection)
    english, chinese = sentence_pair()
    insert_sentences(
        connection,
        [english, chinese],
        segmentation_version="segment-v0.1",
    )
    result = align_documents(
        [english],
        [chinese],
        alignment_key="2025",
        config=AlignmentConfig(length_ratio_override=1.6),
    )
    insert_alignments(
        connection,
        result.alignments,
        year=2025,
        created_at=FIXED_TIME,
    )
    return result.alignments[0].alignment_id


def save_sample_annotation(
    connection,
    alignment_id: str,
    *,
    annotation_id: str = "ANN_001",
    annotator_id: str = "researcher_1",
) -> None:
    save_annotation(
        connection,
        annotation_id=annotation_id,
        alignment_id=alignment_id,
        annotator_id=annotator_id,
        modality_label="preserved",
        stance_label="preserved",
        annotator_confidence="high",
        notes="Synthetic test annotation.",
        possible_alignment_error=False,
        annotation_guideline_version="guidelines-v0.1",
        created_at=FIXED_TIME,
    )


def test_database_creation_and_foreign_keys(database) -> None:
    table_names = {
        row[0]
        for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

    assert {
        "documents",
        "paragraphs",
        "sentences",
        "alignments",
        "alignment_sentences",
        "annotations",
    } <= table_names
    assert database.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_insert_document(database) -> None:
    document = imported_document("EN")

    insert_document(
        database,
        document,
        checksum="a" * 64,
        preprocessing_version="clean-v0.1",
    )

    row = database.execute(
        "SELECT * FROM documents WHERE document_id = ?", (document.document_id,)
    ).fetchone()
    assert row["source_filename"] == "2025_rio_en.txt"
    assert row["checksum"] == "a" * 64
    assert row["preprocessing_version"] == "clean-v0.1"


def test_insert_sentence_and_trace_to_original_document(database) -> None:
    insert_documents_and_paragraphs(database)
    english, _ = sentence_pair()

    insert_sentences(
        database,
        [english],
        segmentation_version="segment-v0.1",
    )

    row = database.execute(
        """
        SELECT s.sentence_id, p.paragraph_no, d.source_filename
        FROM sentences AS s
        JOIN paragraphs AS p ON p.paragraph_id = s.paragraph_id
        JOIN documents AS d ON d.document_id = p.document_id
        WHERE s.sentence_id = ?
        """,
        (english.sentence_id,),
    ).fetchone()
    assert dict(row) == {
        "sentence_id": "EN_2025_P001_S01",
        "paragraph_no": 1,
        "source_filename": "2025_rio_en.txt",
    }


def test_insert_and_get_alignment(database) -> None:
    alignment_id = populate_alignment(database)

    alignment = get_alignment(database, alignment_id)

    assert alignment is not None
    assert alignment["en_sentence_ids"] == ["EN_2025_P001_S01"]
    assert alignment["zh_sentence_ids"] == ["ZH_2025_P001_S01"]
    assert alignment["alignment_type"] == "1:1"
    linked_count = database.execute(
        "SELECT COUNT(*) FROM alignment_sentences WHERE alignment_id = ?",
        (alignment_id,),
    ).fetchone()[0]
    assert linked_count == 2


def test_corpus_overview_preview_and_annotation_queue(database) -> None:
    alignment_id = populate_alignment(database)

    overview = get_corpus_overview(database)
    assert overview.document_count == 2
    assert overview.years == (2025,)
    assert overview.english_sentence_count == 1
    assert overview.chinese_sentence_count == 1
    assert overview.aligned_unit_count == 1
    assert overview.count_1_1 == 1
    assert overview.high_confidence_count == 1
    assert get_available_years(database) == (2025,)

    preview = list_alignment_preview(database)
    assert preview[0]["alignment_id"] == alignment_id
    assert preview[0]["document_title"] == "Rio Declaration"

    queue = list_alignments_for_annotation(database, "researcher_1")
    assert queue[0]["alignment_id"] == alignment_id
    assert queue[0]["annotation_id"] is None
    assert queue[0]["document_title"] == "Rio Declaration"


def test_save_and_retrieve_annotation(database) -> None:
    alignment_id = populate_alignment(database)

    save_sample_annotation(database, alignment_id)
    annotation = get_annotation(
        database,
        alignment_id=alignment_id,
        annotator_id="researcher_1",
    )

    assert annotation is not None
    assert annotation["annotation_id"] == "ANN_001"
    assert annotation["modality_label"] == "preserved"
    assert annotation["possible_alignment_error"] is False

    filtered = list_alignments_for_annotation(
        database,
        "researcher_1",
        year=2025,
        alignment_type="1:1",
        confidence_band="high",
        annotation_status="annotated",
        possible_alignment_error=False,
        modality_label="preserved",
        stance_label="preserved",
        notes_search="Synthetic test",
    )
    assert [record["alignment_id"] for record in filtered] == [alignment_id]
    assert list_alignments_for_annotation(
        database,
        "researcher_1",
        annotation_status="unannotated",
    ) == ()


def test_pilot_csv_export_has_exact_columns_and_preserves_text(database) -> None:
    alignment_id = populate_alignment(database)
    save_sample_annotation(database, alignment_id)

    csv_text = export_pilot_annotations_csv(database, "researcher_1")
    reader = csv.DictReader(StringIO(csv_text))
    rows = list(reader)

    assert tuple(reader.fieldnames) == PILOT_EXPORT_COLUMNS
    assert len(rows) == 1
    assert rows[0]["alignment_id"] == alignment_id
    assert rows[0]["english_text"] == (
        "We reaffirm implementation of the 2030 Agenda."
    )
    assert rows[0]["chinese_text"] == "我们重申落实《2030年议程》。"
    assert rows[0]["modality_label"] == "preserved"
    assert rows[0]["stance_label"] == "preserved"
    assert rows[0]["annotator_confidence"] == "high"
    assert rows[0]["possible_alignment_error"] == "false"
    assert rows[0]["guideline_version"] == "guidelines-v0.1"


def test_duplicate_annotation_is_prevented_but_second_annotator_is_allowed(database) -> None:
    alignment_id = populate_alignment(database)
    save_sample_annotation(database, alignment_id)

    with pytest.raises(DuplicateAnnotationError, match="update_annotation"):
        save_sample_annotation(
            database,
            alignment_id,
            annotation_id="ANN_DUPLICATE",
        )

    save_sample_annotation(
        database,
        alignment_id,
        annotation_id="ANN_002",
        annotator_id="researcher_2",
    )
    assert database.execute(
        "SELECT COUNT(*) FROM annotations WHERE alignment_id = ?",
        (alignment_id,),
    ).fetchone()[0] == 2


def test_explicit_annotation_update_preserves_created_at(database) -> None:
    alignment_id = populate_alignment(database)
    save_sample_annotation(database, alignment_id)
    updated_time = datetime(2026, 9, 1, 9, 30, tzinfo=timezone.utc)

    update_annotation(
        database,
        "ANN_001",
        modality_label="strengthened",
        stance_label="neutralized",
        annotator_confidence="medium",
        notes="Reviewed explicitly.",
        possible_alignment_error=True,
        annotation_guideline_version="guidelines-v0.2",
        updated_at=updated_time,
    )
    annotation = get_annotation(
        database,
        alignment_id=alignment_id,
        annotator_id="researcher_1",
    )

    assert annotation["created_at"] == FIXED_TIME.isoformat()
    assert annotation["updated_at"] == updated_time.isoformat()
    assert annotation["modality_label"] == "strengthened"
    assert annotation["possible_alignment_error"] is True


def test_annotation_progress_and_next_unannotated_alignment(database) -> None:
    alignment_id = populate_alignment(database)

    assert get_next_unannotated_alignment(
        database, "researcher_1"
    )["alignment_id"] == alignment_id
    progress = get_annotation_progress(database, "researcher_1")
    assert (progress.total_alignments, progress.annotated_alignments) == (1, 0)

    save_sample_annotation(database, alignment_id)

    assert get_next_unannotated_alignment(database, "researcher_1") is None
    progress = get_annotation_progress(database, "researcher_1")
    assert progress.remaining_alignments == 0
    assert progress.completion_percentage == 100.0


def test_annotation_traces_to_sentences_paragraphs_and_source_files(database) -> None:
    alignment_id = populate_alignment(database)
    save_sample_annotation(database, alignment_id)

    rows = database.execute(
        """
        SELECT an.annotation_id, als.language, s.sentence_id,
               p.paragraph_no, d.source_filename
        FROM annotations AS an
        JOIN alignment_sentences AS als
          ON als.alignment_id = an.alignment_id
        JOIN sentences AS s ON s.sentence_id = als.sentence_id
        JOIN paragraphs AS p ON p.paragraph_id = s.paragraph_id
        JOIN documents AS d ON d.document_id = p.document_id
        WHERE an.annotation_id = ?
        ORDER BY als.language, als.sentence_position
        """,
        ("ANN_001",),
    ).fetchall()

    assert [(row["language"], row["source_filename"]) for row in rows] == [
        ("EN", "2025_rio_en.txt"),
        ("ZH", "2025_rio_zh.txt"),
    ]
    assert all(row["paragraph_no"] == 1 for row in rows)


def test_foreign_key_integrity_rejects_orphan_paragraph(database) -> None:
    orphan = ParsedParagraph(
        document_id="missing_document",
        paragraph_no=1,
        raw_paragraph_text="Orphan paragraph.",
        cleaned_paragraph_text="Orphan paragraph.",
        original_order=1,
    )

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        insert_paragraphs(database, [orphan])


def test_duplicate_paragraph_numbers_fail_atomically(database) -> None:
    document = imported_document("EN")
    insert_document(
        database,
        document,
        checksum="a" * 64,
        preprocessing_version="clean-v0.1",
    )
    duplicates = (
        ParsedParagraph(document.document_id, 1, "First.", "First.", 1),
        ParsedParagraph(document.document_id, 1, "Duplicate.", "Duplicate.", 2),
    )

    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        insert_paragraphs(database, duplicates)

    assert database.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0] == 0


def test_alignment_insert_rejects_type_that_disagrees_with_sentence_counts(
    database,
) -> None:
    insert_documents_and_paragraphs(database)
    english, chinese = sentence_pair()
    insert_sentences(database, [english, chinese], segmentation_version="segment-v0.1")
    inconsistent = AlignmentUnit(
        alignment_id="A_INCONSISTENT",
        english_sentence_ids=(english.sentence_id,),
        chinese_sentence_ids=(chinese.sentence_id,),
        english_text=english.text,
        chinese_text=chinese.text,
        paragraph_no=1,
        alignment_type="1:2",  # type: ignore[arg-type]
        raw_alignment_cost=0.1,
        normalized_confidence=0.9,
        confidence_band="high",
    )

    with pytest.raises(ValueError, match="contains 1:1 sentences"):
        insert_alignments(database, [inconsistent], year=2025, created_at=FIXED_TIME)
    assert database.execute("SELECT COUNT(*) FROM alignments").fetchone()[0] == 0

    wrong_text = AlignmentUnit(
        alignment_id="A_WRONG_TEXT",
        english_sentence_ids=(english.sentence_id,),
        chinese_sentence_ids=(chinese.sentence_id,),
        english_text="Text not present in the linked sentence.",
        chinese_text=chinese.text,
        paragraph_no=1,
        alignment_type="1:1",
        raw_alignment_cost=0.1,
        normalized_confidence=0.9,
        confidence_band="high",
    )
    with pytest.raises(ValueError, match="English text does not match"):
        insert_alignments(database, [wrong_text], year=2025, created_at=FIXED_TIME)


def test_one_annotators_error_flag_does_not_hide_unit_from_another(database) -> None:
    alignment_id = populate_alignment(database)
    save_annotation(
        database,
        annotation_id="ANN_FLAG_A",
        alignment_id=alignment_id,
        annotator_id="researcher_a",
        modality_label="uncertain",
        stance_label="uncertain",
        annotator_confidence="low",
        possible_alignment_error=True,
        annotation_guideline_version="guidelines-v0.1",
        created_at=FIXED_TIME,
    )

    queue_b = list_alignments_for_annotation(
        database, "researcher_b", annotation_status="unannotated"
    )

    assert [record["alignment_id"] for record in queue_b] == [alignment_id]
    assert database.execute(
        "SELECT possible_alignment_error FROM alignments WHERE alignment_id = ?",
        (alignment_id,),
    ).fetchone()[0] == 0


def test_annotation_save_rejects_blank_trace_identifiers(database) -> None:
    alignment_id = populate_alignment(database)

    with pytest.raises(ValueError, match="annotation_id"):
        save_annotation(
            database,
            annotation_id=" ",
            alignment_id=alignment_id,
            annotator_id="researcher",
            modality_label="N/A",
            stance_label="N/A",
            annotator_confidence="high",
            annotation_guideline_version="guidelines-v0.1",
        )


def test_complete_alignment_and_annotation_exports(database) -> None:
    alignment_id = populate_alignment(database)
    save_sample_annotation(database, alignment_id)

    aligned_reader = csv.DictReader(StringIO(export_aligned_corpus_csv(database)))
    aligned_rows = list(aligned_reader)
    annotation_reader = csv.DictReader(StringIO(export_annotations_csv(database)))
    annotation_rows = list(annotation_reader)

    assert tuple(aligned_reader.fieldnames or ()) == ALIGNED_CORPUS_COLUMNS
    assert aligned_rows[0]["alignment_id"] == alignment_id
    assert json.loads(aligned_rows[0]["en_sentence_ids"]) == [
        "EN_2025_P001_S01"
    ]
    assert tuple(annotation_reader.fieldnames or ()) == ANNOTATION_EXPORT_COLUMNS
    assert annotation_rows[0]["annotator_id"] == "researcher_1"
    assert annotation_rows[0]["possible_alignment_error"] == "false"


def test_research_csv_and_jsonl_preserve_source_traceability(database) -> None:
    alignment_id = populate_alignment(database)
    save_sample_annotation(database, alignment_id)
    config = ExportConfig(alignment_confidence_threshold=0.0)

    csv_rows = list(
        csv.DictReader(StringIO(export_research_dataset_csv(database, config)))
    )
    jsonl_rows = [
        json.loads(line)
        for line in export_research_dataset_jsonl(database, config).splitlines()
    ]

    assert csv_rows[0]["annotation_id"] == "ANN_001"
    assert json.loads(csv_rows[0]["en_document_ids"]) == ["2025_rio_en"]
    assert json.loads(csv_rows[0]["zh_source_filenames"]) == [
        "2025_rio_zh.txt"
    ]
    assert jsonl_rows[0]["alignment"]["alignment_id"] == alignment_id
    assert jsonl_rows[0]["annotation"]["annotation_id"] == "ANN_001"
    assert jsonl_rows[0]["traceability"]["english_sentences"][0][
        "source_filename"
    ] == "2025_rio_en.txt"
    assert jsonl_rows[0]["traceability"]["chinese_sentences"][0][
        "sentence_id"
    ] == "ZH_2025_P001_S01"


def test_research_export_rejects_alignment_text_that_disagrees_with_sources(
    database,
) -> None:
    alignment_id = populate_alignment(database)
    save_sample_annotation(database, alignment_id)
    database.execute(
        "UPDATE alignments SET en_text = 'Tampered text.' WHERE alignment_id = ?",
        (alignment_id,),
    )
    database.commit()

    with pytest.raises(TraceabilityError, match="text disagrees"):
        export_research_dataset_jsonl(
            database, ExportConfig(alignment_confidence_threshold=0.0)
        )


def test_reproducible_bundle_has_declared_metadata_and_no_raw_documents(database) -> None:
    alignment_id = populate_alignment(database)
    save_sample_annotation(database, alignment_id)
    config = ExportConfig(
        alignment_confidence_threshold=0.0,
        annotator_id="researcher_1",
    )

    first = build_reproducible_export_files(
        database, config, export_timestamp=FIXED_TIME
    )
    second = build_reproducible_export_files(
        database, config, export_timestamp=FIXED_TIME
    )
    metadata = json.loads(first["export_metadata.json"])

    assert first == second
    assert tuple(first) == (
        "aligned_corpus.csv",
        "annotations.csv",
        "research_dataset.csv",
        "research_dataset.jsonl",
        "export_metadata.json",
    )
    assert metadata["project_version"] == "0.1.0"
    assert metadata["preprocessing_version"] == ["clean-v0.1"]
    assert metadata["segmentation_version"] == ["segment-v0.1"]
    assert metadata["annotation_guideline_version"] == ["guidelines-v0.1"]
    assert metadata["alignment_confidence_threshold"] == 0.0
    assert metadata["export_timestamp"] == FIXED_TIME.isoformat()
    assert metadata["raw_source_files_included"] is False
    assert "1. We reaffirm" not in "".join(first.values())
