# Handoff: Salary Scout

Last updated: 2026-09-15, after step 7 (browser demo) on `main`.

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
| Executed benchmark notebook, findings section | `notebooks/02_benchmark.ipynb` |
| Technical report | `docs/report.md` |
| Blog post draft | `docs/blog.md` |
| Figures exported from both notebooks (17 PNGs) | `docs/figures/` |
| Benchmark result tables (per fold, model, pattern) | `docs/results/benchmark_*.csv` |
| Data dictionary | `docs/jobs_data_dictionary.md` |
| Benchmark models as library code, deployable `SalaryModel`, `train_final()` | `src/salary_scout/models.py` |
| Browser export: featuriser spec, flattened trees, metrics, fixtures | `src/salary_scout/export.py` |
| Browser demo (static HTML, dependency-free JS) and its model files | `web/`, `web/model/` |
| Node replay of the JS port against Python fixtures | `web/test/verify.mjs`, `tests/test_web.py` |
| GitHub Pages deploy of `web/`, live at https://alexanderwu.github.io/salary-scout/ | `.github/workflows/pages.yml` |
| Twelve ADRs, all accepted | `docs/adr/` |

Not started: the service deployment (step 8). The benchmark notebook still defines
its models inline; `models.py` holds the same code, and the notebook was not rebuilt
to import it (its outputs are the committed record).

## Commands

```sh
uv sync --extra dev                       # create .venv and install everything
uv run pytest -q                          # 34 tests, ~4 s (needs data/jobs.duckdb and node)
uv run ruff check src tests
uv run python -m salary_scout.dataset     # rebuild data/derived.parquet (~30 s)
uv run jupyter lab                        # interactive
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_benchmark.ipynb   # ~40 min
uv run python -m salary_scout.models      # train the full model -> models/salary_model.joblib (git-ignored, ~2 min)
uv run python -m salary_scout.export      # refit the browser model, write web/model/ (~3 min)
node web/test/verify.mjs                  # JS port vs Python fixtures (also run by pytest when node exists)
python -m http.server -d web 8000         # serve the demo locally
```

`data/jobs.duckdb` (1 GB) is git-ignored and must be present locally. The loader opens
it read-only; the full table loads in about two seconds. `data/derived.parquet` (80 MB)
is also git-ignored; `load_derived()` builds it on first use.

## Key facts about the data

- 27,098 postings, unique on `requisition_id`. Individual-contributor roles only,
  heavily data/analytics and software, 95% published April to September 2026.
- 24,191 have a parsed salary; 23,824 survive the plausibility filter (USD, min ≥ 20k,
  max ≤ 1M, max/min ≤ 3). Train 20,257, time-holdout test 3,567.
- The exact salary figure appears in 67% of raw descriptions. After scrubbing, a row's
  own min or max figure survives in 0.02% of rows (5 of 23,824); the pipeline check
  fails above 0.1% (ADR 0009). Never featurise unscrubbed text: the derived table does not carry
  the raw text columns at all.
- `collapse_key` groups sibling postings (same role, different location or level).
  Nearly 90% of postings have siblings. All splits must be grouped by it.
- Salary disclosure varies by state from 77% to 99%, so the model learns the pricing
  behaviour of disclosing employers.
- `job_information_json` holds third-party user IDs and is excluded everywhere (ADR 0002).

## Benchmark results (step 4, done)

Grouped 5-fold CV on the training split, log MAE on `log_mid` (lower is better):

| model | full | description only | metadata only | title only |
|---|---|---|---|---|
| median_by_category | 0.256 | 0.317 | 0.256 | 0.317 |
| ridge_metadata | 0.212 | 0.325 | 0.212 | 0.325 |
| ridge_all_blocks | 0.174 | 0.246 | 0.414 | 0.270 |
| lgbm_all_blocks | 0.147 | 0.247 | 0.291 | 0.380 |
| lgbm_block_dropout | 0.147 | 0.190 | 0.185 | 0.224 |

- Time holdout, dropout model, full inputs: log MAE 0.144, MAPE 13.7%, R² 0.76,
  range overlap 93%. CV: MAPE 15.0%, R² 0.74, overlap 84%.
- Dropout cost on full inputs: −0.0002 ± 0.0009 log MAE. ADR 0007 confirmed and accepted.
- Company-held-out folds: within 0.002 of the grouped CV for every model and pattern.
  Recognising the employer adds almost nothing beyond company metadata and text.
- `log_spread` MAE 0.115 (LightGBM) vs 0.157 (constant median).
- Timing test: shrinking text hashes to 2^16 (description) and 2^14 (title) columns
  changes log MAE by under 0.002. Use that for the browser model.
- Per-fold tables: `docs/results/benchmark_grouped_cv.csv`, `benchmark_time_holdout.csv`,
  `benchmark_company_heldout.csv`. Columns: protocol, fold, model, pattern, then metrics.

Model settings used: LightGBM 800 trees, learning rate 0.05, 63 leaves, colsample 0.3,
subsample 0.8, min_child_samples 20; ridge alpha 1.0; dropout p = 0.3 with one masked
copy of each training row appended (its own `FeatureBlocks` fitted on the augmented frame).

## Decisions in force

- Target: `log_mid = log((min+max)/2)` and `log_spread = log(max/min)`; reconstruct the
  range from both (ADR 0003).
- Metrics: MAE and MAPE of the midpoint in dollars, plus share of test rows whose true
  range overlaps the predicted range.
- Splits: newest 15% by publish date as final test, whole collapse groups on one side;
  grouped 5-fold CV for selection; company-held-out folds reported separately (ADR 0006).
- Masking: five feature blocks (`title`, `description`, `role_meta`, `location`,
  `company`), one model trained with random block dropout, SHAP summed per block
  (ADR 0007, accepted after the benchmark).
- Employer identity: smoothed target encoding of `company_name`, cross-fitted by
  collapse group; the feature transformer is fitted inside every fold (ADR 0010).
- Single input: `data/derived.parquet` from `salary_scout.dataset` is the only
  thing notebooks and apps read; raw text is not in it (ADR 0011).
- Deployment: two targets (ADR 0008). The browser demo is a plain JavaScript port of
  the featuriser and trees, verified against Python fixtures, with exact block-coalition
  Shapley explanations (ADR 0012); the FastAPI service in Docker runs the full model.
  The text featuriser is hash-based so the browser needs no vocabulary.

## How the feature pipeline works (step 3, done)

`salary_scout.dataset`:

- `prepare_inputs(df)` adds `*_clean` text columns (HTML stripped, salary scrubbed),
  `primary_state`, `n_states`, `is_remote`. Works on one posting at inference time.
- `add_targets`, `assign_splits`: `split` (train/test), `cv_fold` (0-4 by
  `collapse_key`, -1 on test), `company_fold` (0-4 by `company_name`, -1 on test).
- `load_derived()` / `write_derived()` persist `data/derived.parquet`.

`salary_scout.features`:

- `BLOCKS` maps each block to its source columns. `FeatureBlocks` is a
  `ColumnTransformer` wrapper; sub-transformers are named `<block>:<family>`, and after
  fitting `block_slices_` gives one contiguous column range per block and `layout_` a
  table of every sub-transformer. `sum_by_block(values)` collapses per-feature
  attributions to per-block.
- Default width is 463,742 sparse columns (title 2×2^16 hashed 1-2 grams, description
  2^18 + 2^16 hashed unigrams, capped at 30k chars). Fit on the train split takes 7 s.
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
- Per-block attributions: `booster.predict(X, pred_contrib=True)` returns a sparse
  matrix with one extra column (the expected value); pass the rest to `sum_by_block`.

## How the model and export work (step 7, done)

`salary_scout.models`:

- `fit_models`, `evaluate`, `summarise`, `MASKS`, `LGBM_PARAMS` are the notebook code.
- `SalaryModel(fb_kwargs, lgbm_params)`: `fit(df)` appends one dropout copy and fits
  `FeatureBlocks` plus a `PairModel` of two LightGBMs. `predict(df, masked=[...])`
  returns a frame of `low, high, mid, log_mid, log_spread`. `explain(df, method=
  "tree_shap"|"coalition")` returns five block columns plus `baseline`; they sum to
  `log_mid`. `frame_from_postings(list_of_dicts)` runs `prepare_inputs` for raw input.
  `save`/`load` via joblib (the LightGBM factory is a class, not a lambda, for this).
- `train_final()` fits on the train split (so the report's holdout numbers describe it)
  and saves `models/salary_model.joblib` (6.8 MB, git-ignored). Time holdout of the
  browser-sized model: log MAE 0.145, MAPE 13.7%, overlap 92.5%.

`salary_scout.export` (ADR 0012):

- `featuriser_spec(fb)` walks `fb.ct_.transformers_` and emits one entry per
  sub-transformer: kind (`presence`, `text_hash`, `log_len`, `onehot`, `numeric`,
  `multihot`, `target_enc`), output `start`/`width`, and fitted state. The one-hot entry
  maps category -> offset with sklearn's ordering (frequent categories in
  `categories_` order, then one infrequent column; unknown -> zeros; NaN is a category
  if it was frequent at fit). The spec also carries the cleaning regexes, stop words,
  seniority map and reference year so the JS has no constants of its own.
- `trees_spec(booster)` flattens `dump_model()`; `predict_from_spec` is the NumPy
  reference evaluator and `export_browser_model` refuses to write if it disagrees with
  LightGBM. Children: index >= 0 is a node, `-(k+1)` is leaf `k`.
- `holdout_metrics` scores all 31 block combinations on the test split; the UI quotes
  the one in use. `build_fixtures` pulls 60 raw test postings from DuckDB (joined by
  `requisition_id`) and records predictions per mask pattern, per-block nnz/sum of the
  feature vector, and coalition explanations for eight rows.

`web/`: `js/hash.js` (MurmurHash3 x86_32, signed, `abs(h) % n`), `js/featurizer.js`
(`prepare` + per-block sparse featurisation, present or masked), `js/trees.js`,
`js/model.js` (`predict`, `explain` = exact Shapley over 32 coalitions with the
training-mean baseline, `metricsFor`), `js/app.js` (form <-> raw posting, SVG bars,
example loader). `verify.mjs` passes 829/829 checks with zero error.

## Resume here (state at the end of the 2026-09-15 session)

Step 7 is done and committed: model code in `src/`, the browser demo in `web/`, ADR
0012 accepted. GitHub Pages is enabled and the demo is live at https://alexanderwu.github.io/salary-scout/;
the blog and README link to it. Next is step 8, the FastAPI service (see below).

## Next steps

### Step 5 and 6: report and blog (done)

`docs/report.md` and `docs/blog.md`. Figures are exported from the executed notebooks by a
small script (session scratchpad) that maps notebook code-cell index to a slug; the map is:
EDA cells 9, 19, 21, 27, 30, 32, 34, 35, 36, 37, 41, 42, 43 and benchmark cells 12, 19, 22.
If a notebook is rebuilt, re-export and check the cell indices still line up.

### Step 7: browser demo (done)

See "How the model and export work" above and ADR 0012. If `features.py` changes,
re-run the export and `node web/test/verify.mjs`; the fixtures are the contract.

### Step 8: service deployment

FastAPI app in `src/salary_scout/service.py` (or `service/`) loading
`models/salary_model.joblib` via `SalaryModel.load()`. Endpoints: `POST /predict`
taking a list of raw postings (the `RAW_INPUT_COLUMNS` fields from `export.py`) plus an
optional `masked` list, returning range, targets, per-block attributions (tree SHAP by
default, coalition on request) and the holdout error for the pattern; `GET /health`
and `GET /schema`. Batch scoring is a loop over `frame_from_postings` in chunks.
Dockerfile on `python:3.12-slim` with `uv sync --no-dev`; the model file is built in
the image by running `python -m salary_scout.models` at build time (needs
`data/derived.parquet` as a build context input) or copied in. Add `models/` to
`.dockerignore` decisions accordingly. Write ADR 0013 for the API contract.

## Working notes for the next session

- Notebooks are generated by a builder script with `nbformat` and executed with
  `nbconvert`, then committed with outputs. The builder for `02_benchmark.ipynb` lived in
  the session scratchpad; the notebook itself is the source of truth. Keep that pattern;
  write the builder with the Write tool, not a shell heredoc.
- Smoke-test a notebook before the long run: exec its code cells on a 1,500-row sample
  with tiny hash widths and 15 trees. The full benchmark run is about 40 minutes and
  nbconvert only writes outputs at the end.
- Matplotlib treats `$` in labels as math mode; write `\\$` in f-strings.
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
- `mask_blocks` casts int and bool columns to float before writing NaN: raw postings
  from JSON arrive with `nb_employees` as int, and pandas refuses NaN in int columns.
- Headless screenshots work with Chrome:
  `chrome.exe --headless=new --screenshot=out.png --window-size=1200,900 --virtual-time-budget=15000 http://127.0.0.1:8000/`
  after `python -m http.server -d web 8000`. Desktop Chrome will not go narrower than
  about 500 px, so phone layouts cannot be checked this way.
- Known Python/JS boundary: JS `\b` and `\d` are ASCII in the salary regex while
  Python's are Unicode; no fixture row hits it. Node 22 is installed locally.
- The web `fetch` of `model/` needs HTTP; `file://` fails with a clear message in the UI.
