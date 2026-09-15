"""Build and persist the derived modelling table.

The derived table is the single input for every notebook and model from here on.
It holds cleaned rows, scrubbed text, both targets, the derived columns the
feature blocks expect, and the split assignment from ADR 0006. It is written to
``data/derived.parquet`` so later steps never repeat the DuckDB load and scrub.

Split columns:

- ``split``: ``"test"`` for the newest ~15% of postings (whole collapse groups
  kept together), ``"train"`` otherwise.
- ``cv_fold``: 0..4 for train rows, grouped by ``collapse_key``; -1 on test rows.
- ``company_fold``: 0..4 for train rows, grouped by ``company_name``; -1 on test rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from salary_scout.cleaning import filter_plausible_compensation, strip_html, strip_salary_mentions
from salary_scout.data import DEFAULT_DB_PATH, REPO_ROOT, load_jobs

DEFAULT_DERIVED_PATH = REPO_ROOT / "data" / "derived.parquet"

# Text columns that get HTML-stripped and salary-scrubbed. Every text feature is
# built from the ``*_clean`` versions; the raw columns are not carried forward.
TEXT_COLUMNS = ["title", "core_job_title", "description", "requirements_summary"]

TEST_FRACTION = 0.15
N_FOLDS = 5
SEED = 0


def parse_json_list(value: object) -> list[str]:
    """Turn a JSON-encoded list column value into a list of stripped strings."""
    if not isinstance(value, str) or not value.startswith("["):
        return []
    try:
        items = json.loads(value)
    except ValueError:
        return []
    return [str(x).strip() for x in items if isinstance(x, (str, int, float)) and str(x).strip()]


def clean_text(text: object) -> str:
    """HTML removal followed by salary scrubbing. Non-strings become empty text."""
    if not isinstance(text, str):
        return ""
    return strip_salary_mentions(strip_html(text))


def prepare_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived input columns the feature blocks consume.

    Safe to call on a single posting at inference time; it only reads the row.
    Adds ``*_clean`` text columns, ``primary_state``, ``n_states`` and ``is_remote``.
    """
    out = df.copy()
    for col in TEXT_COLUMNS:
        if col in out:
            out[f"{col}_clean"] = out[col].map(clean_text)
    states = out["workplace_states"].map(parse_json_list) if "workplace_states" in out else None
    if states is not None:
        out["primary_state"] = states.map(
            lambda s: s[0].replace(", US", "") if s else np.nan
        )
        out["n_states"] = states.map(len).astype(float)
    if "workplace_type" in out:
        out["is_remote"] = (out["workplace_type"] == "Remote").astype(float)
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``log_mid`` and ``log_spread`` (ADR 0003). Requires plausible USD rows."""
    out = df.copy()
    lo, hi = out["yearly_min_compensation"], out["yearly_max_compensation"]
    out["mid"] = (lo + hi) / 2
    out["log_mid"] = np.log(out["mid"])
    out["log_spread"] = np.log(hi / lo)
    return out


def assign_splits(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Time holdout by collapse group, then grouped CV folds on the remainder (ADR 0006).

    A group's date is the newest publish date among its postings, so a repost that
    straddles the boundary lands in the test set with its siblings.
    """
    out = df.copy()
    group_date = out.groupby("collapse_key")["publish_ts"].max()
    group_size = out.groupby("collapse_key").size()
    order = group_date.sort_values(ascending=False).index
    cum = group_size.reindex(order).cumsum()
    n_test = round(test_fraction * len(out))
    test_groups = set(cum[cum <= n_test].index)
    if not test_groups:  # tiny frames: take at least the newest group
        test_groups = {order[0]}
    is_test = out["collapse_key"].isin(test_groups)
    out["split"] = np.where(is_test, "test", "train")

    out["cv_fold"] = -1
    out["company_fold"] = -1
    train_idx = out.index[~is_test]
    train = out.loc[train_idx]
    for col, groups in (("cv_fold", train["collapse_key"]), ("company_fold", train["company_name"])):
        k = min(n_folds, groups.nunique())
        if k < 2:
            continue
        gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = np.empty(len(train), dtype=int)
        for fold, (_, idx) in enumerate(gkf.split(train, groups=groups.fillna("(unknown)"))):
            folds[idx] = fold
        out.loc[train_idx, col] = folds
    return out


def build_derived(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Raw loader output -> cleaned, scrubbed, targeted, split modelling table."""
    df = filter_plausible_compensation(df_raw)
    df = prepare_inputs(df)
    df = add_targets(df)
    df = assign_splits(df)
    keep = [c for c in df.columns if c not in TEXT_COLUMNS]  # drop unscrubbed text
    return df[keep].sort_values("publish_ts").reset_index(drop=True)


def write_derived(
    path: Path | str = DEFAULT_DERIVED_PATH,
    db_path: Path | str = DEFAULT_DB_PATH,
    limit: int | None = None,
) -> pd.DataFrame:
    """Build the derived table from DuckDB and write it to Parquet."""
    derived = build_derived(load_jobs(db_path, limit=limit))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    derived.to_parquet(path, index=False)
    return derived


def load_derived(path: Path | str = DEFAULT_DERIVED_PATH, rebuild: bool = False) -> pd.DataFrame:
    """Read the derived table, building it first if missing or ``rebuild`` is set."""
    path = Path(path)
    if rebuild or not path.exists():
        return write_derived(path)
    return pd.read_parquet(path)


if __name__ == "__main__":
    derived = write_derived()
    print(f"wrote {len(derived):,} rows to {DEFAULT_DERIVED_PATH}")
    print(derived["split"].value_counts().to_string())
