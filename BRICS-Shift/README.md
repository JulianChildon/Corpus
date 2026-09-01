# BRICS-Shift

BRICS-Shift is a lightweight computational translation studies tool for
constructing and manually annotating an English-Chinese parallel corpus of
BRICS diplomatic declarations.

This repository contains the initial **v0.1** project scaffold. The intended
workflow is:

1. Import an English declaration and its official Chinese translation.
2. Preprocess the source texts.
3. Detect numbered paragraphs.
4. Segment paragraphs into sentences.
5. Align English and Chinese sentences deterministically.
6. Construct a parallel corpus.
7. Record human annotations.
8. Calculate descriptive statistics.
9. Export the corpus as CSV.

> **Research principle:** The software organizes translation evidence. Humans
> interpret translation shifts.

The human annotator is the only source of gold-standard translation-shift
labels. BRICS-Shift does not use machine learning, language-model APIs, or
automatic translation-shift classification.

## Current status

The v0.1 project currently provides conservative UTF-8 document import,
document metadata, deterministic Unicode and whitespace normalization,
numbered-paragraph detection, EN-ZH paragraph-structure comparison, and a
deterministic paragraph-bounded rule-based segmenter and sentence aligner.
It also includes SQLite persistence for documents, paragraphs, sentences,
alignments, and independent annotator records, plus a minimal Streamlit research
interface for corpus review, human annotation, descriptive statistics, and
reproducible export.

Sentence segmentation is deterministic and applied separately inside each
numbered paragraph. English rules protect a small documented set of common
abbreviations, initials, acronyms, decimals, and ellipses; Chinese rules use
explicit sentence-final punctuation. These transparent rules are not a general
linguistic parser, so the Import page always presents sentence boundaries for
researcher review before persistence.

The cleaning layer uses NFC Unicode normalization and preserves line and
paragraph order, Arabic paragraph numbers, punctuation, quotation marks,
terminology, and other non-whitespace text. Invalid UTF-8 is reported rather
than silently replaced.

The paragraph layer recognizes a small explicit set of Arabic-number markers,
keeps headings separate, preserves source order and multiline paragraph text,
and reports missing, duplicate, non-monotonic, and unusually large numbering
gaps without changing them.

The alignment layer first matches official paragraph numbers. Equal sentence
counts use ordered 1:1 alignment; unequal counts use dynamic programming with
only 1:1, 1:2, 2:1, and 2:2 operations. Its configurable cost uses length,
Arabic-numeral agreement, sentence order, and structural penalties. Alignment
confidence is a heuristic quality score, not a calibrated probability.

When at least three ordered 1:1 sentence pairs are available, the default
configuration estimates the median number of Chinese alphanumeric characters
per English word token. Otherwise it uses the documented, configurable v0.1
fallback of `1.6`. A researcher can also supply an explicit ratio override.

The persistence layer uses Python's standard `sqlite3` module, enables foreign
keys on every initialized connection, and preserves a queryable trace from an
annotation through its alignment and sentences to paragraphs, documents, and
original source filenames. Creating an annotation never overwrites an existing
record for the same alignment and annotator; changes require the explicit
update function.

Alignment quality-control tools provide filterable review records, derived
eligible/quarantined status with explicit reasons, an alignment-level
`possible_alignment_error` flag, and fixed-seed random samples. Sampling may be
unstratified or use configurable alignment-type proportions. Quarantine is a
query rule only: it never deletes or rewrites an alignment.

The annotation architecture defines only the allowed v0.1 value vocabulary and
storage validation. All linguistic definitions, criteria, and examples remain
researcher-authored placeholders in the versioned guideline template. Each
saved annotation stores its guideline version, and annotator confidence uses
the categorical values `high`, `medium`, or `low`.

Agreement-study mode supports two independent annotators on the same persisted
sample. Samples use a fixed random seed and may be proportionally stratified by
year, alignment type, or their combination. The database stores the sampling
algorithm version, population hash, eligibility settings, stratum allocations,
and exact sample membership/order. During independent annotation, each queue
loads only that annotator's own record. Labels and agreement results remain
hidden until both annotators complete the full sample.

Agreement reporting calculates raw agreement and Cohen's kappa separately for
modality and stance, using paired human annotations only. Disagreements retain
the alignment text and both labels for later researcher-led adjudication. The
application reports the statistics without assigning qualitative descriptions
such as “good” or “bad,” and it never resolves disagreements automatically.

The minimal Streamlit interface now provides Import, Corpus, Annotation,
Agreement, descriptive Analysis, and Export pages. Import accepts paired UTF-8
TXT uploads, previews cleaning, paragraph warnings, sentence boundaries, and
alignments, then persists the reviewed pipeline in one transaction. Original
files are saved under `data/raw/`; existing database records and different
content under the same filename are never overwritten. The annotation form
never preselects a shift label, loads an annotator's existing record for
explicit editing, filters the default queue by configurable alignment
confidence, and supports Previous, Next, Save, and Save & Next navigation.

Pilot controls filter annotation units by year, alignment type, confidence
band, annotation status, alignment-error flag, modality label, stance label,
and note text. Review mode restricts the queue to an annotator's saved examples.
The pilot CSV export applies the same research filters and includes aligned
texts, labels, annotator confidence, notes, flags, and guideline versions.

The Analysis page reports descriptive distributions from human annotations
only. Eligible, non-error annotations form the shift-statistics population;
uncertain labels remain visible, while quarantined and alignment-error records
are counted separately. Percentage denominators are displayed explicitly, and
researchers may group by year, declaration, or their own named year ranges. No
significance tests or substantive linguistic claims are generated.

## Exported data and source materials

The Export page can download a reproducible ZIP bundle containing five UTF-8
files:

- `aligned_corpus.csv` contains all derived alignment units, including units
  below the configured research threshold.
- `annotations.csv` contains saved human labels separately from derived data.
- `research_dataset.csv` joins eligible alignments with human annotations for
  tabular analysis.
- `research_dataset.jsonl` represents the same joined records with nested
  document, paragraph, sentence, alignment, and annotation traceability.
- `export_metadata.json` records software and processing versions, the export
  timestamp, confidence threshold, alignment-error policy, and row counts.

The confidence threshold and possible-alignment-error rule apply only to the
joined research dataset. They never delete records from the complete alignment
or annotation exports. This makes every exclusion explicit and recoverable.

Original UTF-8 declarations belong in `data/raw/`. They are research source
files and are **not exported by default**, because redistribution rights may
differ from the right to analyse them. `data/processed/` stores deterministic
intermediate data, while `data/exports/` is reserved for research metadata,
human annotations, and derived aligned data. The JSONL provenance uses source
filenames and checksums to trace records without copying complete declarations.

## Requirements

- Python 3.11 or newer

## Local installation

From the `BRICS-Shift` directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the application

```bash
streamlit run app.py
```

The sidebar includes an **Interface language / 界面语言** selector. Choose
**English** or **简体中文** at any time. This changes only the Streamlit display;
stored document data, canonical annotation-label values, and exported research
files are not translated or rewritten.

On the **Import** page:

1. Enter a stable pair ID such as `2025_rio`, year, and declaration title.
2. Upload the English official `.txt` and Chinese official `.txt` files.
3. Select **Generate segmentation and alignment preview**.
4. Review paragraph warnings, every sentence boundary, alignment failures, and
   the EN-ZH alignment table.
5. Check the explicit review confirmation and import the preview.
6. Open **Corpus** to verify counts, then **Annotation** to begin human coding.

Both uploads must be valid UTF-8 and contain recognizable Arabic-numbered
paragraphs. A preview never writes to the database; only the final confirmation
does so.

## Run the tests

```bash
pytest
```

## Project layout

- `app.py`: Streamlit application entry point.
- `src/brics_shift/`: Python package for the research pipeline, including
  uploaded-pair ingestion in `ingestion.py`, deterministic segmentation in
  `segmentation.py`, and agreement sampling/statistics in `agreement.py`.
- `annotation/guidelines.md`: evolving human-annotation protocol.
- `data/raw/`: original declarations and official translations.
- `data/processed/`: deterministic intermediate data.
- `data/exports/`: exported corpus files.
- `tests/`: automated tests for deterministic processing modules.
