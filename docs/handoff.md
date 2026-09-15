# Handoff: Salary Scout

Last updated: 2026-09-15, after commit `d663fb0` on `main`.

Read this first in a new session. It says what exists, what was decided, and what to
build next. Decisions are recorded in `docs/adr/`; do not re-litigate them without
writing a superseding record.

## Project goal

An explainable model that predicts the advertised salary range of a job posting from its
description and metadata, working even when input groups are masked. Deliverables, in
order: EDA notebook, feature pipeline, split and benchmark notebook, technical report,
general-audience blog post, a browser-only demo app, and a scalable service deployment.
It doubles as a portfolio piece for recruiters, so code and prose should be legible.

## Current state

Done and committed:

| Item | Where |
|---|---|
| Project scaffold (uv, src layout, tests) | `pyproject.toml`, `src/salary_scout/` |
| DuckDB loader with dedupe and JSON flattening | `src/salary_scout/data.py` |
| Cleaning: plausibility filter and salary scrubber | `src/salary_scout/cleaning.py` |
| Executed EDA notebook, 13 figures, decisions section | `notebooks/01_eda.ipynb` |
| Data dictionary | `docs/jobs_data_dictionary.md` |
| Eight ADRs (six accepted, two proposed) | `docs/adr/` |

Not started: feature pipeline, benchmarks, report, blog post, both apps.

## Commands

```sh
uv sync --extra dev                       # create .venv and install everything
uv run pytest -q                          # 5 tests, ~4 s (needs data/jobs.duckdb)
uv run jupyter lab                        # interactive
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

`data/jobs.duckdb` (1 GB) is git-ignored and must be present locally. The loader opens
it read-only; the full table loads in about two seconds.

## Key facts about the data

- 27,098 postings, unique on `requisition_id`. Individual-contributor roles only,
  heavily data/analytics and software, 95% published April to September 2026.
- 24,191 have a parsed salary; 23,824 survive the plausibility filter (USD, min ≥ 20k,
  max ≤ 1M, max/min ≤ 3).
- The exact salary figure appears in 67% of raw descriptions. `strip_salary_mentions`
  brings that to zero on the cleaned set. Never featurise unscrubbed text.
- `collapse_key` groups sibling postings (same role, different location or level).
  Nearly 90% of postings have siblings. All splits must be grouped by it.
- Salary disclosure varies by state from 77% to 99%, so the model learns the pricing
  behaviour of disclosing employers.
- Metadata-only ridge on six categorical fields, grouped 5-fold: R² 0.48, log MAE 0.22
  (about 25% typical error). This is the floor to beat.
- `job_information_json` holds third-party user IDs and is excluded everywhere (ADR 0002).

## Decisions in force

- Target: `log_mid = log((min+max)/2)` and `log_spread = log(max/min)`; reconstruct the
  range from both (ADR 0003).
- Metrics: MAE and MAPE of the midpoint in dollars, plus share of test rows whose true
  range overlaps the predicted range.
- Splits: newest 15% by publish date as final test, whole collapse groups on one side;
  grouped 5-fold CV for selection; company-held-out folds reported separately (ADR 0006).
- Masking: five feature blocks (`title`, `description`, `role_meta`, `location`,
  `company`), one model trained with random block dropout, SHAP summed per block
  (ADR 0007, proposed).
- Deployment: browser demo on ONNX Runtime Web with a hashing text featuriser, plus a
  FastAPI service in Docker for the full model (ADR 0008, proposed). This means the text
  featuriser must be hash-based from the first benchmark.

## Next steps

### Step 3: feature pipeline (`src/salary_scout/features.py`)

- A `FeatureBlocks` transformer built on sklearn `ColumnTransformer`, one sub-pipeline
  per block, each exposing its output column range so attributions can be summed per block.
- Text blocks: `HashingVectorizer` on scrubbed text (word 1-2 grams for title, word
  unigrams for description with a cap on document length), plus a length feature.
- `role_meta`: one-hot with `min_frequency`, ordinal for seniority, numeric for YOE,
  multi-hot for `technical_tools` (top N by frequency, hashed beyond that).
- `location`: one-hot primary state and workplace type, lat/lon raw, remote flag.
- `company`: target-encoded or frequency-encoded `company_name` (fit on train folds
  only), log headcount, company age, one-hot organisation type and source.
- Masking contract: a function `mask_blocks(X, blocks)` that nulls a block and sets its
  presence indicator to zero. Block dropout is a training-time augmentation that calls it.
- A leakage check that fails if the exact salary figure survives in more than 0.1% of
  scrubbed text.
- Persist the derived table (cleaned rows, targets, split assignment) to a Parquet file
  under `data/` so notebooks do not repeat the load and scrub.

### Step 4: benchmark notebook (`notebooks/02_benchmark.ipynb`)

Models, in order: median by category (baseline), ridge on metadata, ridge on all blocks,
LightGBM on all blocks without dropout, LightGBM with block dropout. Report per mask
pattern (full, description only, metadata only, title only) on the grouped CV, then once
on the time holdout, then company-held-out. Record what dropout costs on full inputs;
that number decides whether ADR 0007 stands. SHAP per block on a handful of examples.

### Later

Report (`docs/report.md`), blog post (`docs/blog.md`), browser app, service app.
Confirm or supersede ADRs 0007 and 0008 after the benchmark.

## Working notes for the next session

- Notebooks are generated by a builder script with `nbformat` and executed with
  `nbconvert`, then committed with outputs. Keep that pattern; write the builder with
  the Write tool, not a shell heredoc.
- Heredocs in the Bash tool break on apostrophes inside the content. Use the Write
  tool for any file containing prose.
- Set `PYTHONIOENCODING=utf-8` when printing notebook output on Windows, or the
  `≥` symbols in cell text raise an encoding error.
- `uv run` prints a harmless warning about a mismatched `VIRTUAL_ENV` from a conda
  environment. Ignore it.
- Chart style in the notebook: single blue `#2a78d6` for magnitude, recessive grey
  grid, left-aligned bold titles, `orientation="horizontal"` for boxplots
  (`vert=` is deprecated in matplotlib 3.11).
- Commit after each step with the attribution line
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
