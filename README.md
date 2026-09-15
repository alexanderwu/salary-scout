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

- `src/salary_scout/` — library code (data loading, features, models)
- `notebooks/` — exploratory analysis and benchmarks
- `docs/` — data dictionary, report, blog post
- `tests/` — unit tests

## Roadmap

1. Exploratory data analysis and target definition
2. Feature engineering pipeline with grouped, maskable feature blocks
3. Train/test split and model benchmarking
4. Technical report and general-audience blog post
5. Browser-only demo app and a scalable service deployment
