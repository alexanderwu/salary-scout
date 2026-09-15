import joblib
import numpy as np
import pytest

from salary_scout.features import BLOCK_NAMES, mask_blocks
from salary_scout.models import (
    MASKS,
    MODEL_ORDER,
    PATTERN_ORDER,
    SalaryModel,
    evaluate,
    fit_models,
    reconstruct_range,
    score,
    summarise,
    train_final,
)

SMALL_FB = {"title_features": 64, "description_features": 256, "tools_top_k": 3,
            "tools_hash": 8, "min_frequency": 2}
SMALL_LGBM = {"n_estimators": 15, "num_leaves": 7, "min_child_samples": 5}


def test_reconstruct_range_round_trip():
    lo, hi = np.array([80_000.0, 100_000.0]), np.array([120_000.0, 100_000.0])
    log_mid, log_spread = np.log((lo + hi) / 2), np.log(hi / lo)
    lo2, hi2, mid = reconstruct_range(log_mid, log_spread)
    assert np.allclose(lo2, lo) and np.allclose(hi2, hi) and np.allclose(mid, (lo + hi) / 2)
    # an inverted spread is clipped to a point range, never lo > hi
    lo3, hi3, _ = reconstruct_range(np.log(100_000.0), -0.5)
    assert lo3 == hi3 == pytest.approx(100_000.0)


def test_score_perfect_prediction(toy):
    s = score(toy, toy["log_mid"].to_numpy(), toy["log_spread"].to_numpy())
    assert s["log_mae"] == pytest.approx(0) and s["mape"] == pytest.approx(0)
    assert s["overlap"] == 1.0 and s["r2"] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def fitted(toy):
    return fit_models(toy, fb_kwargs=SMALL_FB, lgbm_params=SMALL_LGBM)


def test_fit_models_and_evaluate(fitted, toy):
    assert set(fitted["models"]) == set(MODEL_ORDER)
    res = evaluate(fitted, toy, "smoke", 0)
    assert len(res) == len(MODEL_ORDER) * len(MASKS)
    assert res["log_mae"].between(0, 5).all()
    table = summarise(res, "log_mae")
    assert list(table.index) == MODEL_ORDER and list(table.columns) == PATTERN_ORDER
    assert table.notna().all().all()


@pytest.fixture(scope="module")
def model(toy):
    return SalaryModel(fb_kwargs=SMALL_FB, lgbm_params=SMALL_LGBM).fit(toy)


def test_predict_returns_ordered_range(model, toy):
    pred = model.predict(toy)
    assert list(pred.columns) == ["low", "high", "mid", "log_mid", "log_spread"]
    assert (pred["low"] <= pred["high"]).all()
    assert np.allclose(pred["mid"], np.exp(pred["log_mid"]))
    # masking a block changes the prediction for at least some rows
    masked = model.predict(toy, masked=["description"])
    assert not np.allclose(masked["log_mid"], pred["log_mid"])
    # pre-masked input gives the same answer as the masked= argument
    again = model.predict(mask_blocks(toy, ["description"]))
    assert np.allclose(again["log_mid"], masked["log_mid"])


def test_explain_tree_shap_sums_to_prediction(model, toy):
    pred = model.predict(toy)
    exp = model.explain(toy, method="tree_shap")
    assert list(exp.columns) == BLOCK_NAMES + ["baseline"]
    assert np.allclose(exp.sum(axis=1), pred["log_mid"], atol=1e-6)


def test_explain_coalition_is_efficient_and_respects_masks(model, toy):
    sample = toy.iloc[:6]
    exp = model.explain(sample, method="coalition")
    pred = model.predict(sample)
    assert np.allclose(exp.sum(axis=1), pred["log_mid"], atol=1e-9)
    assert (exp["baseline"] == model.baseline_).all()
    masked = model.explain(sample, masked=["title", "company"], method="coalition")
    assert (masked[["title", "company"]] == 0).all().all()
    assert np.allclose(masked.sum(axis=1), model.predict(sample, masked=["title", "company"])["log_mid"])


def test_frame_from_postings_single_raw_posting(model):
    posting = {
        "title": "Senior Data Engineer",
        "description": "<p>Build pipelines. We pay $150,000 to $180,000.</p>",
        "workplace_type": "Remote",
        "workplace_states": '["California, US"]',
        "company_name": "Acme",
        "seniority_level": "Senior Level",
        "job_category": "Data and Analytics",
    }
    frame = SalaryModel.frame_from_postings([posting])
    assert "[SALARY]" in frame.loc[0, "description_clean"]
    pred = model.predict(frame)
    assert len(pred) == 1 and pred.loc[0, "low"] > 0


def test_save_load_round_trip(model, toy, tmp_path):
    path = model.save(tmp_path / "m.joblib")
    loaded = SalaryModel.load(path)
    assert np.allclose(loaded.predict(toy)["log_mid"], model.predict(toy)["log_mid"])
    joblib.dump({"not": "a model"}, tmp_path / "x.joblib")
    with pytest.raises(TypeError):
        SalaryModel.load(tmp_path / "x.joblib")


def test_train_final_uses_train_split(toy, tmp_path):
    df = toy.copy()
    df["split"] = np.where(np.arange(len(df)) % 5 == 0, "test", "train")
    m = train_final(df, path=tmp_path / "final.joblib", fb_kwargs=SMALL_FB, lgbm_params=SMALL_LGBM)
    assert m.n_train_ == (df["split"] == "train").sum()
    assert (tmp_path / "final.joblib").exists()
    m_all = train_final(df, path=None, rows="all", fb_kwargs=SMALL_FB, lgbm_params=SMALL_LGBM)
    assert m_all.n_train_ == len(df)
    with pytest.raises(ValueError):
        train_final(df, path=None, rows="nope")
