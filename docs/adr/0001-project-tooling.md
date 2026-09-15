# 0001. Python project layout and tooling

Date: 2026-09-15
Status: Accepted

## Context

The project needs to serve three audiences at once: exploratory notebooks, a reusable
modelling library, and later two deployed applications. The raw data is a 1 GB DuckDB
file that must never be committed.

## Decision

- Python 3.12+ managed with `uv`; dependencies and lock file in `pyproject.toml` / `uv.lock`.
- `src/` layout with a single package, `salary_scout`, installed in editable mode.
  Notebooks import from the package; they do not define reusable logic inline.
- Cleaning rules and constants live in the package (`salary_scout.cleaning`) so the EDA
  notebook, the feature pipeline, and the deployed apps apply identical logic.
- `data/` is git-ignored. The loader opens `jobs.duckdb` read-only and dedupes on
  `requisition_id`, the only unique key.
- Executed notebooks are committed with outputs so the repository reads as a report
  without a kernel.

## Consequences

- Anyone can reproduce the environment with `uv sync --extra dev`.
- Committed notebook outputs make diffs noisy; we accept that for a portfolio project
  where the rendered result is the point.
- The data file has to be obtained separately; the data dictionary documents its shape.
