# Annotation task: 300 social-network decisions

Thank you for helping. You will judge 300 short events. Each row is one decision
made by a hidden language model in a social-network simulation.

## What you need

- `annotation_sheet.csv` (300 rows)
- `codebook.md` (read this first)

Do **not** ask for or use the file `annotation_key.csv`; it is for the research
team only and would bias the annotation.

## Your job

For each row, fill exactly these columns:

1. `is_action_reasonable` — one of `reasonable`, `unreasonable`, `cannot_judge`.
2. `preferred_action` — one of `keep`, `mute`, `unfollow`, `block`.
3. `confidence_1_5` — 1 (very unsure) to 5 (very sure).
4. `comments` — optional, especially for `cannot_judge`.

Use only the exact lowercase tokens. Do not leave Q1 or Q2 blank. Do not edit
any other column.

## Independence

- Work alone; do not discuss individual rows with the other annotator.
- Do not try to identify the model or the paper's hypothesis.
- Rate "reasonable", not "what I would choose": a different but defensible
  action is still reasonable.
- A 20-row pilot is recommended; then take breaks every 45–60 minutes.

## Returning the sheet

Save as `annotation_sheet_annotatorA.csv` or `annotation_sheet_annotatorB.csv`
and return it to the research team. Do not rename or reorder rows.
