# Handoff: Salary Scout

Last updated: 2026-09-15, after step 3 (feature pipeline) on `main`.

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
| Derived table: scrub, targets, grouped time split, CV folds, Parquet | `src/salary_scout/dataset.py` |
| Feature blocks, masking, block dropout, leakage check | `src/salary_scout/features.py` |
| Executed EDA notebook, 13 figures, decisions section | `notebooks/01_eda.ipynb` |
| Data dictionary | `docs/jobs_data_dictionary.md` |
| Eleven ADRs (nine accepted, two proposed) | `docs/adr/` |

Not started: benchmark notebook, report, blog post, both apps.

## Commands

```sh
uv sync --extra dev                       # create .venv and install everything
uv run pytest -q                          # 19 tests, ~3 s (needs data/jobs.duckdb)
uv run ruff check src tests
uv run python -m salary_scout.dataset     # rebuild data/derived.parquet (~30 s)
uv run jupyter lab                        # interactive
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

`data/jobs.duckdb` (1 GB) is git-ignored and must be present locally. The loader opens
it read-only; the full table loads in about two seconds. `data/derived.parquet` (80 MB)
is also git-ignored; `load_derived()` builds it on first use.

## Key facts about the data

- 27,098 postings, unique on `requisition_id`. Individual-contributor roles only,
  heavily data/analytics and software, 95% published April to September 2026.
- 24,191 have a parsed salary; 23,824 survive the plausibility filter (USD, min ≥ 20k,
  max ≤ 1M, max/min ≤ 3).
- The exact salary figure appears in 67% of raw descriptions. After scrubbing, a row's
  own min or max figure survives in 0.02% of rows (5 of 23,824); the pipeline check
  fails above 0.1% (ADR 0009). Never featurise unscrubbed text: the derived table does not carry
  the raw text columns at all.
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
- Employer identity: smoothed target encoding of `company_name`, cross-fitted by
  collapse group; the feature transformer is fitted inside every fold (ADR 0010).
- Single input: `data/derived.parquet` from `salary_scout.dataset` is the only
  thing notebooks and apps read; raw text is not in it (ADR 0011).
- Deployment: browser demo on ONNX Runtime Web with a hashing text featuriser, plus a
  FastAPI service in Docker for the full model (ADR 0008, proposed). This means the text
  featuriser must be hash-based from the first benchmark.

## How the feature pipeline works (step 3, done)

`salary_scout.dataset`:

- `prepare_inputs(df)` adds `*_clean` text columns (HTML stripped, salary scrubbed),
  `primary_state`, `n_states`, `is_remote`. Works on one posting at inference time.
- `add_targets`, `assign_splits`: `split` (train/test), `cv_fold` (0-4 by
  `collapse_key`, -1 on test), `company_fold` (0-4 by `company_name`, -1 on test).
  Test = 3,567 rows, train = 20,257.
- `load_derived()` / `write_derived()` persist `data/derived.parquet`.

`salary_scout.features`:

- `BLOCKS` maps each block to its source columns. `FeatureBlocks` is a
  `ColumnTransformer` wrapper; sub-transformers are named `<block>:<family>`, and after
  fitting `block_slices_` gives one contiguous column range per block and `layout_` a
  table of every sub-transformer. `sum_by_block(values)` collapses per-feature
  attributions to per-block.
- Default width is 463,742 sparse columns (title 2×2^16 hashed 1-2 grams, description
  2^18 + 2^16 hashed unigrams, capped at 30k chars). Fit on the train split takes 7 s.
  Shrink `title_features` and `description_features` for the browser model.
- Company name uses `GroupTargetEncoder`: smoothed mean of `log_mid`, cross-fitted by
  `collapse_key` inside `fit_transform` so siblings do not leak into each other, full
  encoding in `transform`. Consequence: `fit_transform(train)` and `transform(train)`
  differ in the company block by design. Compare masked and unmasked outputs with
  `transform`, never against the `fit_transform` result.
- Masking: `mask_blocks(X, blocks, rows=None)` sets a block's columns to NaN and
  `present_<block>` to 0. `dropout_blocks(X, p, rng)` is the training augmentation and
  always leaves at least one block per row. Missing presence columns default to 1, and
  absent source columns (e.g. no `collapse_key` at inference) are filled with NaN.
- `check_salary_leakage(derived)` returns the residual leak rate and raises above 0.1%.

## Next steps

### Step 4: benchmark notebook (`notebooks/02_benchmark.ipynb`)

Start from `load_derived()`; do not reload DuckDB. Models, in order: median by
category (baseline), ridge on metadata, ridge on all blocks, LightGBM on all blocks
without dropout, LightGBM with block dropout. Report per mask pattern (full, description
only, metadata only, title only) on the grouped CV (`cv_fold`), then once on the time
holdout (`split == "test"`), then company-held-out (`company_fold`). Record what dropout
costs on full inputs; that number decides whether ADR 0007 stands. SHAP per block on a
handful of examples via `FeatureBlocks.sum_by_block`.

Practical notes for the benchmark:

- Fit `FeatureBlocks` inside each fold (the company encoder uses the target). Pass
  `y=log_mid`; the second target is modelled on the same matrix.
- LightGBM accepts the CSR matrix directly. Ridge on 460k sparse columns is fine with
  the default solver. Consider `description_features=2**16` for speed while iterating.
- The dropout model needs `dropout_blocks` applied to the training frame before
  `fit_transform`, and the presence columns are then part of the matrix.

### Later

Report (`docs/report.md`), blog post (`docs/blog.md`), browser app, service app.
Confirm or supersede ADRs 0007 and 0008 after the benchmark.

## Working notes for the next session

- Notebooks are generated by a builder script with `nbformat` and executed with
  `nbconvert`, then committed with outputs. Keep that pattern; write the builder with
  the Write tool, not a shell heredoc.
- Heredocs in the Bash tool mangle backslashes and apostrophes in the content (a regex
  `\b` became a literal backspace). For any file edit involving prose or regex, write a
  small Python patch script with the Write tool and run it, or use the Write tool on
  the whole file.
- scikit-learn forbids `__` in `ColumnTransformer` names; the block prefix uses `:`.
- Set `PYTHONIOENCODING=utf-8` when printing notebook output on Windows, or the
  `≥` symbols in cell text raise an encoding error.
- `uv run` prints a harmless warning about a mismatched `VIRTUAL_ENV` from a conda
  environment. Ignore it.
- Chart style in the notebook: single blue `#2a78d6` for magnitude, recessive grey
  grid, left-aligned bold titles, `orientation="horizontal"` for boxplots
  (`vert=` is deprecated in matplotlib 3.11).
- Commit after each step with the attribution line
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
