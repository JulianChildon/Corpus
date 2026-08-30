# BRICS-Shift Annotation Guidelines

Version: `[Researcher to assign]`  
Date: `[YYYY-MM-DD]`

> Template status: `[Draft / Frozen]`

This document is a template. All substantive linguistic definitions, decision
rules, and examples must be supplied and approved by the researcher before a
guideline version is frozen for annotation.

## Unit of annotation

The annotation unit is one aligned English-Chinese translation unit. The
annotator judges only the displayed English source text (ST) and Chinese target
text (TT) belonging to that alignment unit.

Additional researcher instructions about unit boundaries: `[To be completed]`

## Modality Shift

Allowed labels:

- `N/A`
- `preserved`
- `strengthened`
- `weakened`
- `added`
- `omitted`
- `uncertain`

### N/A

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### preserved

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### strengthened

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### weakened

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### added

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### omitted

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### uncertain

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

## Stance Shift

Allowed labels:

- `N/A`
- `preserved`
- `strengthened`
- `weakened`
- `neutralized`
- `uncertain`

### N/A

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### preserved

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### strengthened

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### weakened

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### neutralized

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

### uncertain

- Definition: `[Researcher to define]`
- Inclusion criteria: `[Researcher to define]`
- Exclusion criteria: `[Researcher to define]`
- Positive examples: `[Researcher to supply]`
- Negative examples: `[Researcher to supply]`
- Borderline cases: `[Researcher to supply]`

## Annotator Confidence

Allowed values:

- `high`
- `medium`
- `low`

Researcher instructions for assigning confidence: `[To be completed]`

Annotator confidence records the annotator's self-assessment under the frozen
guidelines. It is separate from the automatic alignment-confidence score.

## Alignment Problems

Annotators may set:

```text
possible_alignment_error = true
```

This means:

> The annotator suspects the English and Chinese texts may not be correctly
> aligned.

It does not itself change either translation-shift label. Annotators are not
required to manually realign the text in v0.1.

## Notes

Free-text notes are optional.

Researcher instructions for notes: `[To be completed, if needed]`

## General principles

- Judge only the displayed ST-TT pair.
- Follow the frozen guideline version assigned to the annotation task.
- Use `uncertain` rather than guessing.
- Use `N/A` if the phenomenon is genuinely absent under the researcher's
  definition.
- Do not infer author intention.
- Do not use external LLMs to determine gold-standard labels.
- Do not treat automatic alignment confidence as a translation-shift label.
- Use the alignment-error flag without changing the displayed source text.

## Guideline version control

Every annotation record must store the exact annotation guideline version used
to create or revise it. Once annotation begins under a frozen version, edits to
the guidelines must receive a new version identifier and date. Previous
annotation records must not be silently relabeled or assigned the new version.
