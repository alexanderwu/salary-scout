"""Export a :class:`SalaryModel` for the browser demo (ADR 0012).

The browser re-implements the feature pipeline and the tree evaluation in JavaScript.
This module writes everything it needs to ``web/model/``:

- ``featuriser.json``: one entry per sub-transformer of the fitted
  :class:`FeatureBlocks`, in output order, with the fitted state (one-hot categories,
  imputer medians, top-K vocabularies, the company table) and the hashing settings.
  Hashed text needs no state beyond the width, which is the point of ADR 0008.
- ``trees_mid.json`` and ``trees_spread.json``: the LightGBM boosters flattened to
  arrays (feature, threshold, children, leaf values, missing-value rules).
- ``metrics.json``: the browser model's time-holdout error for every combination of
  present blocks, so the UI can quote the typical error for the inputs in use.
- ``fixtures.json``: raw test postings with the Python predictions and explanations,
  which ``web/test/verify.mjs`` replays through the JavaScript implementation.
- ``samples.json``: a handful of raw test postings for the "load an example" button.

:func:`predict_from_spec` is a NumPy reference evaluator for the flattened trees.
:func:`featurise_from_spec` is deliberately *not* provided: the JavaScript featuriser
is checked against the fitted scikit-learn transformer through the fixtures instead.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from salary_scout.cleaning import _HTML_TAG_RE, _SALARY_RE, SALARY_TOKEN
from salary_scout.data import REPO_ROOT, load_jobs
from salary_scout.dataset import TEXT_COLUMNS
from salary_scout.features import (
    BLOCK_NAMES,
    BLOCKS,
    REFERENCE_YEAR,
    SENIORITY_ORDER,
    FeatureBlocks,
    GroupTargetEncoder,
    MultiHotTopK,
    presence_column,
)
from salary_scout.models import MASKS, SalaryModel, score

DEFAULT_WEB_DIR = REPO_ROOT / "web" / "model"

# Smaller hashes for the browser: under 0.002 log MAE from the full widths (report §10).
BROWSER_FB_KWARGS = {"title_features": 2**14, "description_features": 2**16}

# Raw posting columns the browser form and the fixtures carry. Everything the feature
# blocks read is derived from these by ``prepare_inputs``.
RAW_INPUT_COLUMNS = [
    *TEXT_COLUMNS,
    "workplace_states",
    *[c for cols in BLOCKS.values() for c in cols
      if not c.endswith("_clean") and c not in ("primary_state", "n_states", "is_remote")],
]

_NUMERIC_FUNCS = {
    "_to_2d": "identity",
    "_log1p_2d": "log1p",
    "_company_age": "company_age",
    "_seniority_ordinal": "seniority",
}
_MISSING_TYPE = {"None": 0, "Zero": 1, "NaN": 2}


def _nan_key(value) -> str:
    return "__nan__" if isinstance(value, float) and math.isnan(value) else str(value)


# ---------------------------------------------------------------------------
# Featuriser spec
# ---------------------------------------------------------------------------


def _onehot_spec(enc: OneHotEncoder, columns: list[str]) -> dict:
    """Category -> output offset for each input column, in scikit-learn's column order:
    frequent categories in ``categories_`` order, then one infrequent column if any."""
    maps, offset = [], 0
    infrequent = enc.infrequent_categories_ or [None] * len(columns)
    for cats, infreq in zip(enc.categories_, infrequent):
        infreq_keys = {_nan_key(v) for v in (infreq if infreq is not None else [])}
        table: dict[str, int] = {}
        for cat in cats:
            key = _nan_key(cat)
            if key not in infreq_keys:
                table[key] = offset
                offset += 1
        entry = {"map": table, "infrequent": None}
        if infreq_keys:
            entry["infrequent"] = offset
            entry["infrequent_values"] = sorted(infreq_keys)
            offset += 1
        maps.append(entry)
    return {"kind": "onehot", "columns": columns, "features": maps, "width": offset}


def _numeric_spec(pipe: Pipeline, columns: list[str]) -> dict:
    func = pipe.steps[0][1].func.__name__
    imputer: SimpleImputer = pipe.steps[-1][1]
    medians = np.nan_to_num(imputer.statistics_.astype(float), nan=0.0).tolist()
    indicators = imputer.indicator_.features_.tolist() if imputer.indicator_ is not None else []
    return {
        "kind": "numeric",
        "func": _NUMERIC_FUNCS[func],
        "columns": columns,
        "medians": medians,
        "indicators": indicators,
        "width": len(columns) + len(indicators),
    }


def _text_hash_spec(pipe: Pipeline, column: str) -> dict:
    prep, vec = pipe.steps[0][1], pipe.steps[-1][1]
    assert vec.alternate_sign is False and vec.binary and vec.norm == "l2"
    return {
        "kind": "text_hash",
        "column": column,
        "n_features": vec.n_features,
        "ngram_range": list(vec.ngram_range),
        "stop_words": vec.stop_words == "english",
        "max_chars": (prep.kw_args or {}).get("max_chars"),
        "width": vec.n_features,
    }


def _multihot_spec(enc: MultiHotTopK, column: str) -> dict:
    return {
        "kind": "multihot",
        "column": column,
        "vocab": list(enc.vocab_),
        "n_hash": enc.n_hash,
        "lowercase": enc.lowercase,
        "width": len(enc.vocab_) + enc.n_hash + 1,
    }


def _target_enc_spec(enc: GroupTargetEncoder, columns: list[str]) -> dict:
    table = {
        _nan_key(k): [float((s + enc.smooth * enc.prior_) / (n + enc.smooth)), float(np.log1p(n))]
        for k, s, n in zip(enc.sums_.index, enc.sums_.to_numpy(), enc.counts_.to_numpy())
    }
    return {
        "kind": "target_enc",
        "column": columns[0],
        "prior": float(enc.prior_),
        "table": table,
        "width": 2,
    }


def _entry_for(name: str, transformer, columns) -> dict:
    cols = list(columns) if isinstance(columns, (list, tuple)) else columns
    family = name.split(":", 1)[1]
    if family == "present":
        return {"kind": "presence", "column": cols[0], "width": 1}
    if family == "len":
        return {"kind": "log_len", "column": cols, "width": 1}
    if isinstance(transformer, MultiHotTopK):
        return _multihot_spec(transformer, cols)
    if isinstance(transformer, GroupTargetEncoder):
        return _target_enc_spec(transformer, cols)
    if isinstance(transformer, Pipeline):
        last = transformer.steps[-1][1]
        if isinstance(last, OneHotEncoder):
            return _onehot_spec(last, cols)
        if isinstance(last, SimpleImputer):
            return _numeric_spec(transformer, cols)
        if last.__class__.__name__ == "HashingVectorizer":
            return _text_hash_spec(transformer, cols)
    raise TypeError(f"no exporter for transformer {name!r}: {transformer!r}")


def featuriser_spec(fb: FeatureBlocks) -> dict:
    """Serialisable description of a fitted :class:`FeatureBlocks`."""
    entries = []
    for name, transformer, columns in fb.ct_.transformers_:
        if name == "remainder":
            continue
        sl = fb.ct_.output_indices_[name]
        entry = _entry_for(name, transformer, columns)
        if entry["width"] != sl.stop - sl.start:
            raise RuntimeError(f"{name}: spec width {entry['width']} != {sl.stop - sl.start}")
        entries.append({"name": name, "block": name.split(":", 1)[0], "start": sl.start, **entry})
    return {
        "n_features": fb.n_features_out_,
        "blocks": {b: [sl.start, sl.stop] for b, sl in fb.block_slices_.items()},
        "block_columns": BLOCKS,
        "presence_columns": {b: presence_column(b) for b in BLOCK_NAMES},
        "stop_words": sorted(ENGLISH_STOP_WORDS),
        "seniority_order": SENIORITY_ORDER,
        "reference_year": REFERENCE_YEAR,
        "text_columns": TEXT_COLUMNS,
        "cleaning": {
            "html_tag_pattern": _HTML_TAG_RE.pattern,
            "salary_pattern": _SALARY_RE.pattern,
            "salary_token": SALARY_TOKEN,
        },
        "transformers": entries,
    }


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


def _flatten_tree(root: dict) -> dict:
    """Depth-first arrays. Children are node indices; a leaf is ``-(leaf_slot + 1)``."""
    feature, threshold, left, right, default_left, missing = [], [], [], [], [], []
    leaf_value: list[float] = []

    def visit(node: dict) -> int:
        if "leaf_value" in node:
            leaf_value.append(float(node["leaf_value"]))
            return -len(leaf_value)
        if node["decision_type"] != "<=":
            raise ValueError(f"unsupported decision type {node['decision_type']!r}")
        i = len(feature)
        feature.append(int(node["split_feature"]))
        threshold.append(float(node["threshold"]))
        default_left.append(int(bool(node["default_left"])))
        missing.append(_MISSING_TYPE[node["missing_type"]])
        left.append(0)
        right.append(0)
        left[i] = visit(node["left_child"])
        right[i] = visit(node["right_child"])
        return i

    if "leaf_value" in root:  # a constant tree
        return {"feature": [], "threshold": [], "left": [], "right": [], "default_left": [],
                "missing": [], "leaf_value": [float(root["leaf_value"])]}
    visit(root)
    return {"feature": feature, "threshold": threshold, "left": left, "right": right,
            "default_left": default_left, "missing": missing, "leaf_value": leaf_value}


def trees_spec(booster) -> dict:
    dump = booster.dump_model()
    if dump["num_tree_per_iteration"] != 1 or dump.get("average_output"):
        raise ValueError("expected a single-output regression booster")
    return {
        "objective": dump["objective"],
        "n_features": dump["max_feature_idx"] + 1,
        "trees": [_flatten_tree(t["tree_structure"]) for t in dump["tree_info"]],
    }


def _tree_value(tree: dict, x: np.ndarray) -> float:
    if not tree["feature"]:
        return tree["leaf_value"][0]
    i = 0
    while True:
        v = x[tree["feature"][i]]
        mt = tree["missing"][i]
        if mt != 2 and math.isnan(v):
            v = 0.0
        if (mt == 1 and abs(v) <= 1e-35) or (mt == 2 and math.isnan(v)):
            nxt = tree["left"][i] if tree["default_left"][i] else tree["right"][i]
        else:
            nxt = tree["left"][i] if v <= tree["threshold"][i] else tree["right"][i]
        if nxt < 0:
            return tree["leaf_value"][-nxt - 1]
        i = nxt


def predict_from_spec(spec: dict, X) -> np.ndarray:
    """Reference evaluator for the flattened trees (dense or sparse rows)."""
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=float)
    return np.array([sum(_tree_value(t, row) for t in spec["trees"]) for row in X])


# ---------------------------------------------------------------------------
# Metrics, fixtures, samples
# ---------------------------------------------------------------------------


def _present_key(present) -> str:
    return "+".join(b for b in BLOCK_NAMES if b in present)


def holdout_metrics(model: SalaryModel, test: pd.DataFrame) -> dict:
    """Time-holdout metrics for every non-empty combination of present blocks."""
    out = {}
    for r in range(1, len(BLOCK_NAMES) + 1):
        for present in itertools.combinations(BLOCK_NAMES, r):
            hide = [b for b in BLOCK_NAMES if b not in present]
            mid, spr = model.predict_targets(test, hide)
            out[_present_key(present)] = score(test, mid, spr)
    return {"n_test": len(test), "by_present_blocks": out}


def _raw_rows(derived: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    raw = load_jobs()
    raw = raw[raw["requisition_id"].isin(ids)].set_index("requisition_id")
    cols = [c for c in RAW_INPUT_COLUMNS if c in raw]
    return raw.loc[ids, cols].reset_index()


def _json_safe(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for rec in frame.to_dict(orient="records"):
        rows.append({k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                     for k, v in rec.items()})
    return rows


def _block_stats(model: SalaryModel, frame: pd.DataFrame) -> list[dict]:
    X = model.transform(frame)
    stats = []
    for i in range(X.shape[0]):
        row = X[i]
        per_block = {}
        for b, sl in model.fb_.block_slices_.items():
            part = row[:, sl]
            per_block[b] = {"nnz": int(part.nnz), "sum": float(part.sum())}
        stats.append(per_block)
    return stats


def build_fixtures(model: SalaryModel, derived: pd.DataFrame, n: int = 60, seed: int = 0) -> dict:
    """Raw test postings plus the Python outputs the JavaScript port must reproduce."""
    test = derived[derived["split"] == "test"]
    rng = np.random.default_rng(seed)
    pick = test.iloc[np.sort(rng.choice(len(test), size=min(n, len(test)), replace=False))]
    ids = pick["requisition_id"].tolist()
    raw = _raw_rows(derived, ids)
    prepared = SalaryModel.frame_from_postings(_json_safe(raw))
    expected = {}
    for pattern, hide in MASKS.items():
        mid, spr = model.predict_targets(prepared, hide)
        expected[pattern] = {"log_mid": mid.tolist(), "log_spread": spr.tolist()}
    n_explain = min(8, len(prepared))
    explain = model.explain(prepared.iloc[:n_explain], method="coalition")
    return {
        "inputs": _json_safe(raw),
        "true": {
            "low": pick["yearly_min_compensation"].tolist(),
            "high": pick["yearly_max_compensation"].tolist(),
        },
        "block_stats": _block_stats(model, prepared),
        "expected": expected,
        "explain_coalition": {c: explain[c].tolist() for c in explain.columns},
    }


def build_samples(derived: pd.DataFrame, n: int = 12, seed: int = 1) -> list[dict]:
    test = derived[derived["split"] == "test"]
    rng = np.random.default_rng(seed)
    pick = test.iloc[np.sort(rng.choice(len(test), size=min(n, len(test)), replace=False))]
    raw = _raw_rows(derived, pick["requisition_id"].tolist())
    rows = _json_safe(raw)
    for row, lo, hi in zip(rows, pick["yearly_min_compensation"], pick["yearly_max_compensation"]):
        row["true_low"], row["true_high"] = float(lo), float(hi)
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _dump(obj, path: Path) -> int:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path.stat().st_size


def export_browser_model(
    derived: pd.DataFrame,
    out_dir: Path | str = DEFAULT_WEB_DIR,
    model: SalaryModel | None = None,
    fixtures: bool = True,
) -> dict[str, int]:
    """Fit (or take) the browser-sized model and write every artefact. Returns file sizes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = derived[derived["split"] == "train"].reset_index(drop=True)
    test = derived[derived["split"] == "test"].reset_index(drop=True)
    if model is None:
        model = SalaryModel(fb_kwargs=BROWSER_FB_KWARGS).fit(train)

    spec = featuriser_spec(model.fb_)
    mid_spec = trees_spec(model.pair_.mid_.booster_)
    spr_spec = trees_spec(model.pair_.spread_.booster_)

    # The reference evaluator must agree with LightGBM before anything is written.
    probe = test.iloc[:20]
    X = model.transform(probe)
    lgb_mid, lgb_spr = model.pair_.predict(X)
    if not np.allclose(predict_from_spec(mid_spec, X), lgb_mid, atol=1e-9):
        raise RuntimeError("flattened log_mid trees disagree with LightGBM")
    if not np.allclose(predict_from_spec(spr_spec, X), lgb_spr, atol=1e-9):
        raise RuntimeError("flattened log_spread trees disagree with LightGBM")

    meta = {
        "baseline_log_mid": model.baseline_,
        "n_train": model.n_train_,
        "fb_kwargs": model.fb_kwargs,
        "lgbm_params": model.pair_.mid_.get_params(),
        "block_names": BLOCK_NAMES,
        "masks": MASKS,
    }
    sizes = {
        "featuriser.json": _dump({**spec, "meta": meta}, out_dir / "featuriser.json"),
        "trees_mid.json": _dump(mid_spec, out_dir / "trees_mid.json"),
        "trees_spread.json": _dump(spr_spec, out_dir / "trees_spread.json"),
        "metrics.json": _dump(holdout_metrics(model, test), out_dir / "metrics.json"),
    }
    if fixtures:
        sizes["fixtures.json"] = _dump(build_fixtures(model, derived), out_dir / "fixtures.json")
        sizes["samples.json"] = _dump(build_samples(derived), out_dir / "samples.json")
    return sizes


if __name__ == "__main__":
    from salary_scout.dataset import load_derived

    for name, size in export_browser_model(load_derived()).items():
        print(f"{name:20s} {size / 1e6:6.2f} MB")
