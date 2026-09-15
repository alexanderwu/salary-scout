from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from salary_scout.data import DEFAULT_DB_PATH, load_jobs
from salary_scout.dataset import assign_splits, build_derived, parse_json_list
from salary_scout.features import (
    BLOCK_NAMES,
    BLOCKS,
    FeatureBlocks,
    GroupTargetEncoder,
    MultiHotTopK,
    check_salary_leakage,
    dropout_blocks,
    mask_blocks,
    presence_column,
    salary_leak_mask,
)


def test_parse_json_list():
    assert parse_json_list('["Python","SQL"]') == ["Python", "SQL"]
    assert parse_json_list("[]") == []
    assert parse_json_list(None) == []
    assert parse_json_list("not json") == []


def test_prepare_inputs_scrubs_and_derives(toy):
    assert not salary_leak_mask(toy).any()
    assert "[SALARY]" in toy["description_clean"].iloc[0]
    assert set(toy["primary_state"].dropna()) <= {"California", "Texas"}
    assert toy["is_remote"].isin([0.0, 1.0]).all()


def test_leak_check_raises_on_raw_text(toy):
    raw = toy.assign(description_clean=toy["description"])
    with pytest.raises(ValueError):
        check_salary_leakage(raw)
    assert check_salary_leakage(toy) == 0.0


def test_assign_splits_grouped(toy):
    out = assign_splits(toy)
    side = out.groupby("collapse_key")["split"].nunique()
    assert (side == 1).all()
    frac = (out["split"] == "test").mean()
    assert 0.1 <= frac <= 0.2
    assert (out.loc[out["split"] == "test", "cv_fold"] == -1).all()
    train = out[out["split"] == "train"]
    assert (train.groupby("collapse_key")["cv_fold"].nunique() == 1).all()
    assert (train.groupby("company_name")["company_fold"].nunique() == 1).all()
    # newest postings are the test set
    assert out.loc[out["split"] == "test", "publish_ts"].min() > train["publish_ts"].median()


def test_mask_blocks_rows_and_presence(toy):
    rows = np.zeros(len(toy), dtype=bool)
    rows[:5] = True
    out = mask_blocks(toy, ["title"], rows=rows)
    assert out["title_clean"].iloc[:5].isna().all()
    assert out["title_clean"].iloc[5:].notna().all()
    assert out[presence_column("title")].iloc[:5].eq(0).all()
    assert out[presence_column("title")].iloc[5:].eq(1).all()
    with pytest.raises(KeyError):
        mask_blocks(toy, ["nope"])


def test_dropout_keeps_one_block(toy):
    out = dropout_blocks(toy, p=1.0, rng=0)
    present = out[[presence_column(b) for b in BLOCK_NAMES]]
    assert (present.sum(axis=1) == 1).all()
    some = dropout_blocks(toy, p=0.3, rng=0)
    present = some[[presence_column(b) for b in BLOCK_NAMES]]
    assert (present.sum(axis=1) >= 1).all()
    assert 0 < (present == 0).to_numpy().mean() < 0.5


def test_multihot_topk():
    enc = MultiHotTopK(top_k=2, n_hash=8).fit(pd.Series(['["a","b"]', '["a","c"]', '["a"]']))
    assert list(enc.vocab_) == ["a", "b"]
    out = enc.transform(pd.Series(['["a","c","c"]', None])).toarray()
    assert out.shape == (2, 2 + 8 + 1)
    assert out[0, 0] == 1 and out[0, 1] == 0 and out[0, 2:10].sum() == 1
    assert out[1].sum() == 0


def test_group_target_encoder_cross_fits():
    X = pd.DataFrame({"cat": ["A"] * 6 + ["B"] * 6, "grp": list("aabbcc") + list("ddeeff")})
    y = np.array([1, 1, 2, 2, 3, 3, 10, 10, 20, 20, 30, 30], dtype=float)
    enc = GroupTargetEncoder(smooth=0.0, n_splits=3)
    inner = enc.fit_transform(X, y)
    full = enc.transform(X)
    # cross-fitted rows never see their own group, so they differ from the full encoding
    assert not np.allclose(inner[:, 0], full[:, 0])
    assert np.allclose(full[:6, 0], 2.0) and np.allclose(full[6:, 0], 20.0)
    unknown = enc.transform(pd.DataFrame({"cat": ["Z", None], "grp": ["x", "y"]}))
    assert np.allclose(unknown[:, 0], enc.prior_) and np.allclose(unknown[:, 1], 0.0)


@pytest.fixture(scope="module")
def fitted(toy):
    fb = FeatureBlocks(title_features=64, description_features=256, tools_top_k=3, tools_hash=8,
                       min_frequency=2)
    Xt = fb.fit_transform(toy, toy[["log_mid", "log_spread"]])
    return fb, Xt


def test_block_slices_cover_output(fitted, toy):
    fb, Xt = fitted
    assert Xt.shape == (len(toy), fb.n_features_out_)
    slices = [fb.block_slices_[b] for b in BLOCK_NAMES]
    assert slices[0].start == 0
    for a, b in pairwise(slices):
        assert a.stop == b.start
    assert slices[-1].stop == fb.n_features_out_
    assert set(fb.layout_["block"]) == set(BLOCKS)
    assert fb.block_of(0) == "title"


def test_masking_only_touches_own_block(fitted, toy):
    fb, _ = fitted
    full = fb.transform(toy).toarray()  # not the fit_transform output: that one is cross-fitted
    for block in BLOCK_NAMES:
        masked = fb.transform(mask_blocks(toy, [block])).toarray()
        sl = fb.block_slices_[block]
        other = np.ones(full.shape[1], dtype=bool)
        other[sl] = False
        assert np.allclose(masked[:, other], full[:, other]), block
        assert (masked[:, sl.start] == 0).all()  # presence column leads the block
        assert not np.allclose(masked[:, sl], full[:, sl]), block


def test_transform_without_presence_columns_matches(fitted, toy):
    fb, Xt = fitted
    again = fb.transform(toy.drop(columns=["collapse_key"]))
    assert again.shape == Xt.shape
    assert np.allclose(again[:, fb.block_slices_["title"]].toarray(),
                       Xt[:, fb.block_slices_["title"]].toarray())


def test_sum_by_block(fitted):
    fb, _ = fitted
    summed = fb.sum_by_block(np.ones(fb.n_features_out_))
    assert list(summed.columns) == BLOCK_NAMES
    assert summed.iloc[0].sum() == fb.n_features_out_


@pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="jobs.duckdb not present")
def test_real_data_smoke():
    derived = build_derived(load_jobs(limit=800))
    assert "description" not in derived.columns
    assert check_salary_leakage(derived) <= 0.001
    fb = FeatureBlocks(title_features=2**10, description_features=2**12)
    Xt = fb.fit_transform(derived, derived["log_mid"])
    assert Xt.shape[0] == len(derived)
    assert fb.block_slices_["description"].stop - fb.block_slices_["description"].start > 2**12
