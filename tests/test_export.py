import json
from itertools import pairwise

import numpy as np
import pytest

from salary_scout.export import (
    RAW_INPUT_COLUMNS,
    featuriser_spec,
    holdout_metrics,
    predict_from_spec,
    trees_spec,
)
from salary_scout.features import BLOCK_NAMES, mask_blocks
from salary_scout.models import SalaryModel

SMALL_FB = {"title_features": 64, "description_features": 256, "tools_top_k": 3,
            "tools_hash": 8, "min_frequency": 2}
SMALL_LGBM = {"n_estimators": 25, "num_leaves": 7, "min_child_samples": 5}


@pytest.fixture(scope="module")
def model(toy):
    return SalaryModel(fb_kwargs=SMALL_FB, lgbm_params=SMALL_LGBM).fit(toy)


def test_featuriser_spec_covers_every_column(model):
    spec = featuriser_spec(model.fb_)
    assert spec["n_features"] == model.fb_.n_features_out_
    entries = spec["transformers"]
    assert entries[0]["start"] == 0
    for a, b in pairwise(entries):
        assert a["start"] + a["width"] == b["start"]
    assert entries[-1]["start"] + entries[-1]["width"] == spec["n_features"]
    kinds = {e["kind"] for e in entries}
    assert kinds == {"presence", "text_hash", "log_len", "onehot", "numeric", "multihot",
                     "target_enc"}
    json.dumps(spec)  # serialisable
    company = next(e for e in entries if e["kind"] == "target_enc")
    assert set(company["table"]) == {"Acme", "Globex", "Initech", "Umbrella"}
    onehot = next(e for e in entries if e["name"] == "role_meta:onehot")
    assert onehot["features"][0]["map"]  # seniority levels
    assert sorted(spec["blocks"]) == sorted(BLOCK_NAMES)


def test_onehot_spec_matches_transformer(model, toy):
    """Replay the one-hot map in Python and compare with scikit-learn's output."""
    spec = featuriser_spec(model.fb_)
    entry = next(e for e in spec["transformers"] if e["name"] == "location:onehot")
    X = model.transform(toy).toarray()[:, entry["start"]:entry["start"] + entry["width"]]
    for i, row in toy.iterrows():
        expected = np.zeros(entry["width"])
        for col, feat in zip(entry["columns"], entry["features"]):
            key = "__nan__" if row[col] is None or row[col] != row[col] else str(row[col])
            if key in feat["map"]:
                expected[feat["map"][key]] = 1
            elif feat["infrequent"] is not None and key in feat["infrequent_values"]:
                expected[feat["infrequent"]] = 1
        assert np.array_equal(X[i], expected), (i, dict(row[entry["columns"]]))


def test_trees_spec_reproduces_lightgbm(model, toy):
    frames = [toy, mask_blocks(toy, ["description", "company"]), mask_blocks(toy, BLOCK_NAMES[1:])]
    mid_spec, spr_spec = trees_spec(model.pair_.mid_.booster_), trees_spec(model.pair_.spread_.booster_)
    assert len(mid_spec["trees"]) == SMALL_LGBM["n_estimators"]
    for frame in frames:
        X = model.fb_.transform(frame)
        lgb_mid, lgb_spr = model.pair_.predict(X)
        assert np.allclose(predict_from_spec(mid_spec, X), lgb_mid, atol=1e-12)
        assert np.allclose(predict_from_spec(spr_spec, X), lgb_spr, atol=1e-12)


def test_holdout_metrics_all_patterns(model, toy):
    m = holdout_metrics(model, toy.iloc[:30])
    assert len(m["by_present_blocks"]) == 31
    assert "title+description+role_meta+location+company" in m["by_present_blocks"]
    assert all(0 <= v["mape"] < 5 for v in m["by_present_blocks"].values())


def test_raw_input_columns_are_raw():
    assert "description" in RAW_INPUT_COLUMNS and "workplace_states" in RAW_INPUT_COLUMNS
    assert not any(c.endswith("_clean") for c in RAW_INPUT_COLUMNS)
    assert "primary_state" not in RAW_INPUT_COLUMNS
