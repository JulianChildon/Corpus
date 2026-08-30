"""Tests for reproducible independent double annotation and agreement."""

from collections import Counter
from datetime import datetime, timezone
import json

import pytest

from brics_shift.agreement import (
    AgreementStudyIncompleteError,
    build_agreement_report,
    calculate_cohens_kappa,
    create_agreement_study,
    get_agreement_progress,
    get_agreement_sample_ids,
    list_agreement_annotation_queue,
)
from brics_shift.database import initialize_database, save_annotation


FIXED_TIME = datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def database():
    connection = initialize_database(":memory:")
    yield connection
    connection.close()


def insert_alignment(
    connection,
    *,
    alignment_id: str,
    year: int,
    paragraph_no: int,
    alignment_type: str,
) -> None:
    connection.execute(
        """
        INSERT INTO alignments (
            alignment_id, year, paragraph_no, en_sentence_ids,
            zh_sentence_ids, en_text, zh_text, alignment_type,
            alignment_cost, alignment_confidence,
            alignment_confidence_band, alignment_algorithm_version,
            created_at
        ) VALUES (?, ?, ?, '[]', '[]', ?, ?, ?, 0.1, 0.9, 'high', ?, ?)
        """,
        (
            alignment_id,
            year,
            paragraph_no,
            f"English {alignment_id}",
            f"Chinese {alignment_id}",
            alignment_type,
            "brics-shift-align-v0.1",
            FIXED_TIME.isoformat(),
        ),
    )
    connection.commit()


def populate_alignments(connection) -> None:
    index = 1
    for year in (2024, 2025):
        for alignment_type in ("1:1", "1:2"):
            for _ in range(2):
                insert_alignment(
                    connection,
                    alignment_id=f"A_{year}_{index:02d}",
                    year=year,
                    paragraph_no=index,
                    alignment_type=alignment_type,
                )
                index += 1


def save_label(
    connection,
    alignment_id: str,
    annotator_id: str,
    modality_label: str,
    stance_label: str,
) -> None:
    save_annotation(
        connection,
        annotation_id=f"ANN_{alignment_id}_{annotator_id}",
        alignment_id=alignment_id,
        annotator_id=annotator_id,
        modality_label=modality_label,
        stance_label=stance_label,
        annotator_confidence="high",
        annotation_guideline_version="guidelines-v0.1",
        created_at=FIXED_TIME,
    )


def create_study(connection, study_id: str = "STUDY_01", **overrides):
    values = {
        "study_id": study_id,
        "study_name": "Pilot agreement study",
        "annotator_a_id": "annotator_a",
        "annotator_b_id": "annotator_b",
        "sample_size": 4,
        "random_seed": 2025,
        "minimum_alignment_confidence": 0.70,
        "created_at": FIXED_TIME,
    }
    values.update(overrides)
    return create_agreement_study(connection, **values)


def test_fixed_seed_sample_and_definition_are_reproducible(database) -> None:
    populate_alignments(database)

    first = create_study(database, "STUDY_A")
    second = create_study(database, "STUDY_B")

    assert get_agreement_sample_ids(database, first.study_id) == (
        get_agreement_sample_ids(database, second.study_id)
    )
    assert first.sample_definition == second.sample_definition
    assert first.sample_definition["random_seed"] == 2025
    assert "eligible_population_alignment_ids_sha256" in first.sample_definition


def test_sampling_can_stratify_by_year_and_alignment_type(database) -> None:
    populate_alignments(database)

    study = create_study(
        database,
        stratify_by_year=True,
        stratify_by_alignment_type=True,
    )
    rows = database.execute(
        """
        SELECT stratum_key FROM agreement_sample_items
        WHERE study_id = ? ORDER BY sample_order
        """,
        (study.study_id,),
    ).fetchall()
    strata = Counter(
        (json.loads(row["stratum_key"])["year"],
         json.loads(row["stratum_key"])["alignment_type"])
        for row in rows
    )

    assert strata == Counter(
        {(2024, "1:1"): 1, (2024, "1:2"): 1,
         (2025, "1:1"): 1, (2025, "1:2"): 1}
    )


def test_agreement_queue_does_not_reveal_other_annotator_labels(database) -> None:
    populate_alignments(database)
    study = create_study(database, sample_size=1)
    alignment_id = get_agreement_sample_ids(database, study.study_id)[0]
    save_label(database, alignment_id, "annotator_a", "strengthened", "weakened")

    queue_b = list_agreement_annotation_queue(
        database, study.study_id, "annotator_b"
    )

    assert queue_b[0]["annotation_id"] is None
    assert queue_b[0]["modality_label"] is None
    assert queue_b[0]["stance_label"] is None
    assert all("other" not in key and "annotator_a" not in key for key in queue_b[0])


def test_nonparticipant_cannot_open_agreement_queue(database) -> None:
    populate_alignments(database)
    study = create_study(database)

    with pytest.raises(ValueError, match="not an annotator"):
        list_agreement_annotation_queue(database, study.study_id, "outsider")


def test_cohens_kappa_formula() -> None:
    agreement_count, raw_agreement, kappa = calculate_cohens_kappa(
        ["preserved", "preserved", "strengthened", "strengthened"],
        ["preserved", "strengthened", "strengthened", "strengthened"],
    )

    assert agreement_count == 3
    assert raw_agreement == pytest.approx(0.75)
    assert kappa == pytest.approx(0.5)


def test_kappa_is_explicitly_undefined_when_expected_agreement_is_one() -> None:
    agreement_count, raw_agreement, kappa = calculate_cohens_kappa(
        ["preserved", "preserved"],
        ["preserved", "preserved"],
    )

    assert agreement_count == 2
    assert raw_agreement == 1.0
    assert kappa is None


def test_report_is_separate_for_modality_and_stance(database) -> None:
    populate_alignments(database)
    study = create_study(database)
    alignment_ids = get_agreement_sample_ids(database, study.study_id)
    modality_a = ("preserved", "preserved", "strengthened", "strengthened")
    modality_b = ("preserved", "strengthened", "strengthened", "strengthened")
    stance = ("N/A", "preserved", "weakened", "uncertain")
    for index, alignment_id in enumerate(alignment_ids):
        save_label(database, alignment_id, "annotator_a", modality_a[index], stance[index])
        save_label(database, alignment_id, "annotator_b", modality_b[index], stance[index])

    report = build_agreement_report(database, study.study_id)

    assert report.progress.doubly_annotated_count == 4
    assert report.modality.agreement_count == 3
    assert report.modality.raw_agreement == pytest.approx(0.75)
    assert report.modality.cohen_kappa == pytest.approx(0.5)
    assert report.stance.agreement_count == 4
    assert report.stance.raw_agreement == pytest.approx(1.0)
    assert report.stance.cohen_kappa == pytest.approx(1.0)
    assert len(report.disagreements) == 1
    assert report.disagreements[0].dimension == "modality"
    assert report.disagreements[0].annotator_a_label == "preserved"
    assert report.disagreements[0].annotator_b_label == "strengthened"


def test_results_remain_hidden_until_both_annotators_finish(database) -> None:
    populate_alignments(database)
    study = create_study(database, sample_size=1)
    alignment_id = get_agreement_sample_ids(database, study.study_id)[0]
    save_label(database, alignment_id, "annotator_a", "preserved", "preserved")

    progress = get_agreement_progress(database, study.study_id)
    assert progress.annotator_a_count == 1
    assert progress.annotator_b_count == 0
    assert progress.is_complete is False
    with pytest.raises(AgreementStudyIncompleteError, match="remain hidden"):
        build_agreement_report(database, study.study_id)
