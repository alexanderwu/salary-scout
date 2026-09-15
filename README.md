# Salary Scout

Explainable machine learning for predicting the advertised salary range of a job posting
from its description and metadata. Predictions still work when input groups are masked
(description only, metadata only, and so on), and every prediction comes with an
explanation of what drove it.

Data: ~27k job postings scraped from [hiring.cafe](https://hiring.cafe), stored in
`data/jobs.duckdb` (not tracked in git). See
[docs/jobs_data_dictionary.md](docs/jobs_data_dictionary.md).

## Setup

```sh
uv sync --extra dev
uv run jupyter lab
```

## Layout

- `src/salary_scout/` — library code (data loading, features, models, browser export)
- `web/` — browser-only demo: static HTML and dependency-free JavaScript, model files in `web/model/`
- `notebooks/` — exploratory analysis and benchmarks
- `docs/` — data dictionary, report, blog post
- `tests/` — unit tests (including a Node replay of the JavaScript port against Python fixtures)

## Roadmap

1. Exploratory data analysis and target definition
2. Feature engineering pipeline with grouped, maskable feature blocks
3. Train/test split and model benchmarking
4. Technical report and general-audience blog post
5. Browser-only demo app (done, see below) and a scalable service deployment

## Browser demo

Live at [https://alexanderwu.github.io/salary-scout/](https://alexanderwu.github.io/salary-scout/), published from `web/` by GitHub Actions on every push to `main` that touches it.

```sh
uv run python -m salary_scout.export   # refit the browser model and write web/model/ (~3 min)
node web/test/verify.mjs               # JavaScript port must match Python on 60 fixture rows
python -m http.server -d web 8000      # then open http://localhost:8000
```

The demo loads about 7 MB of JSON (two LightGBM boosters and the featuriser state) and
runs everything in the page. See ADR 0012 for why it is plain JavaScript rather than ONNX.
