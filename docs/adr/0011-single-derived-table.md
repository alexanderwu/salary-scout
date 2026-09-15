# 0011. One derived table is the only input to modelling

Date: 2026-09-15
Status: Accepted

## Context

The EDA notebook loaded DuckDB, filtered, scrubbed and split in its own cells. Repeating
that in every later notebook and app costs about 30 seconds per run and, more
importantly, creates places where raw text could reach a featuriser by accident.
ADR 0005 depends on the scrubbing happening exactly once and everywhere.

## Decision

- `salary_scout.dataset.build_derived` produces one table: plausible USD rows only,
  `*_clean` text columns, both targets, the derived location columns, and the split
  assignment from ADR 0006 (`split`, `cv_fold`, `company_fold`).
- The raw text columns are dropped from the derived table. Downstream code cannot
  featurise unscrubbed text because it is not there.
- The table is persisted to `data/derived.parquet` (git-ignored, about 80 MB) and read
  with `load_derived()`, which rebuilds it when missing. Notebooks and training scripts
  start from it and never open DuckDB.
- `prepare_inputs` is the single function that turns a raw posting into the columns the
  feature blocks expect. The apps call the same function on user input.

## Consequences

- Changing a cleaning rule or the split protocol means rebuilding the table
  (`uv run python -m salary_scout.dataset`) and re-running anything that consumed it.
  The rebuild is cheap; the discipline is the point.
- Split membership is fixed in the table, so every model in the benchmark sees the
  same folds without re-deriving them.
- Anyone reproducing the project needs the DuckDB file once, then only the Parquet.
