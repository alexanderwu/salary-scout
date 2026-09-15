"""Models: the benchmark set from ``notebooks/02_benchmark.ipynb`` and the deployable
block-dropout model behind both apps.

Two layers:

- Benchmark helpers (:func:`fit_models`, :func:`evaluate`, :func:`summarise`) are the
  notebook code moved here verbatim in behaviour, so the notebook and the apps cannot
  drift apart. They fit every model in ``MODEL_ORDER`` and score every mask pattern.
- :class:`SalaryModel` is the single model the apps ship: one :class:`FeatureBlocks`
  fitted on the dropout-augmented frame and one LightGBM per target (``log_mid`` and
  ``log_spread``). :func:`train_final` fits it on the train split and saves it.

Attributions come in two flavours. ``explain(method="tree_shap")`` sums LightGBM's
exact per-feature contributions per block; ``explain(method="coalition")`` computes
exact Shapley values over the five blocks by masking every subset (32 predictions),
which is what the browser demo does because it needs no TreeSHAP implementation.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from salary_scout.data import REPO_ROOT
from salary_scout.dataset import load_derived, prepare_inputs
from salary_scout.features import BLOCK_NAMES, FeatureBlocks, dropout_blocks, mask_blocks

SEED = 0
DROPOUT_P = 0.3
LGBM_PARAMS: dict = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "colsample_bytree": 0.3,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "verbose": -1,
    "random_state": SEED,
}

# Mask patterns reported everywhere: which blocks are *removed*.
MASKS: dict[str, list[str]] = {
    "full": [],
    "description_only": ["title", "role_meta", "location", "company"],
    "metadata_only": ["title", "description"],
    "title_only": ["description", "role_meta", "location", "company"],
}
METRICS = ["log_mae", "r2", "mae_usd", "mape", "overlap", "spread_mae"]
MODEL_ORDER = [
    "median_by_category",
    "ridge_metadata",
    "ridge_all_blocks",
    "lgbm_all_blocks",
    "lgbm_block_dropout",
]
PATTERN_ORDER = list(MASKS)

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "salary_model.joblib"


# ---------------------------------------------------------------------------
# Targets <-> ranges, scoring
# ---------------------------------------------------------------------------


def reconstruct_range(log_mid, log_spread) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(low, high, mid)`` in dollars from the two targets (ADR 0003).

    The spread is clipped at zero so the range can never be inverted.
    """
    mid = np.exp(np.asarray(log_mid, dtype=float))
    spread = np.exp(np.clip(np.asarray(log_spread, dtype=float), 0, None))
    lo = 2 * mid / (1 + spread)
    return lo, lo * spread, mid


def score(frame: pd.DataFrame, log_mid_pred, log_spread_pred) -> dict[str, float]:
    """The six benchmark metrics for predictions against ``frame``'s true range."""
    lo, hi, mid = reconstruct_range(log_mid_pred, log_spread_pred)
    y = frame["log_mid"].to_numpy()
    true_mid = frame["mid"].to_numpy()
    return {
        "log_mae": float(np.mean(np.abs(log_mid_pred - y))),
        "r2": float(r2_score(y, log_mid_pred)),
        "mae_usd": float(np.mean(np.abs(mid - true_mid))),
        "mape": float(np.mean(np.abs(mid - true_mid) / true_mid)),
        "overlap": float(
            np.mean(
                (lo <= frame["yearly_max_compensation"].to_numpy())
                & (hi >= frame["yearly_min_compensation"].to_numpy())
            )
        ),
        "spread_mae": float(
            np.mean(np.abs(np.clip(log_spread_pred, 0, None) - frame["log_spread"].to_numpy()))
        ),
    }


# ---------------------------------------------------------------------------
# Benchmark models
# ---------------------------------------------------------------------------


class CategoryMedian:
    """Baseline: median ``log_mid`` by (category, seniority), global median spread."""

    keys: ClassVar[list[str]] = ["job_category", "seniority_level"]

    def fit(self, frame: pd.DataFrame):
        self.global_mid_ = float(frame["log_mid"].median())
        self.global_spread_ = float(frame["log_spread"].median())
        self.table_ = frame.groupby(self.keys)["log_mid"].median().rename("m").reset_index()
        return self

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        merged = frame[self.keys].merge(self.table_, on=self.keys, how="left")
        mid = merged["m"].fillna(self.global_mid_).to_numpy()
        return mid, np.full(len(frame), self.global_spread_)


class PairModel:
    """Two regressors from one factory: one for ``log_mid``, one for ``log_spread``."""

    def __init__(self, factory: Callable[[], object]):
        self.factory = factory

    def fit(self, X, log_mid, log_spread):
        self.mid_ = self.factory().fit(X, log_mid)
        self.spread_ = self.factory().fit(X, log_spread)
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        return self.mid_.predict(X), self.spread_.predict(X)


def zero_blocks(X: sp.csr_matrix, fb: FeatureBlocks, blocks: Iterable[str]) -> sp.csr_matrix:
    """Zero out whole blocks of an already-transformed matrix (used for ridge_metadata)."""
    keep = np.ones(X.shape[1])
    for b in blocks:
        keep[fb.block_slices_[b]] = 0
    return (X @ sp.diags(keep)).tocsr()


class _LGBMFactory:
    """Picklable ``lambda: LGBMRegressor(**params)`` so a fitted model can be saved."""

    def __init__(self, params: dict | None = None):
        self.params = {**LGBM_PARAMS, **(params or {})}

    def __call__(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(**self.params)


def _lgbm_factory(params: dict | None) -> Callable[[], lgb.LGBMRegressor]:
    return _LGBMFactory(params)


def fit_models(
    fit_df: pd.DataFrame,
    seed: int = SEED,
    fb_kwargs: dict | None = None,
    lgbm_params: dict | None = None,
    dropout_p: float = DROPOUT_P,
) -> dict:
    """Fit all benchmark models on one frame.

    Returns ``{"fb", "fb_do", "models", "timings"}``: the feature transformer fitted on
    the plain frame, the one fitted on the dropout-augmented frame (used only by
    ``lgbm_block_dropout``), the models by name, and wall-clock seconds per stage.
    """
    fb_kwargs = {"random_state": seed, **(fb_kwargs or {})}
    timings: dict[str, float] = {}
    t = time.time()
    fb = FeatureBlocks(**fb_kwargs)
    X = fb.fit_transform(fit_df, fit_df["log_mid"])
    y_mid, y_spr = fit_df["log_mid"].to_numpy(), fit_df["log_spread"].to_numpy()
    timings["features"] = time.time() - t

    models: dict[str, object] = {}
    models["median_by_category"] = CategoryMedian().fit(fit_df)
    t = time.time()
    models["ridge_metadata"] = PairModel(lambda: Ridge(alpha=1.0)).fit(
        zero_blocks(X, fb, ["title", "description"]), y_mid, y_spr
    )
    models["ridge_all_blocks"] = PairModel(lambda: Ridge(alpha=1.0)).fit(X, y_mid, y_spr)
    timings["ridge"] = time.time() - t
    t = time.time()
    models["lgbm_all_blocks"] = PairModel(_lgbm_factory(lgbm_params)).fit(X, y_mid, y_spr)
    timings["lgbm"] = time.time() - t

    t = time.time()
    aug = pd.concat([fit_df, dropout_blocks(fit_df, p=dropout_p, rng=seed)], ignore_index=True)
    fb_do = FeatureBlocks(**fb_kwargs)
    X_aug = fb_do.fit_transform(aug, aug["log_mid"])
    models["lgbm_block_dropout"] = PairModel(_lgbm_factory(lgbm_params)).fit(
        X_aug, aug["log_mid"].to_numpy(), aug["log_spread"].to_numpy()
    )
    timings["lgbm_dropout"] = time.time() - t
    return {"fb": fb, "fb_do": fb_do, "models": models, "timings": timings}


def evaluate(fitted: dict, eval_df: pd.DataFrame, protocol: str, fold: int) -> pd.DataFrame:
    """Score every fitted model under every mask pattern; one row per (model, pattern)."""
    rows = []
    for pattern, blocks in MASKS.items():
        ev = mask_blocks(eval_df, blocks)
        X, X_do = fitted["fb"].transform(ev), fitted["fb_do"].transform(ev)
        for name, m in fitted["models"].items():
            if name == "median_by_category":
                mid, spr = m.predict(ev)
            else:
                mid, spr = m.predict(X_do if name == "lgbm_block_dropout" else X)
            rows.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "model": name,
                    "pattern": pattern,
                    **score(eval_df, mid, spr),
                }
            )
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame, metric: str, agg: str = "mean") -> pd.DataFrame:
    """Pivot a results table to models x patterns for one metric."""
    t = res.pivot_table(index="model", columns="pattern", values=metric, aggfunc=agg).reindex(
        index=MODEL_ORDER, columns=PATTERN_ORDER
    )
    t.columns.name = f"{metric} ({agg})"
    return t


# ---------------------------------------------------------------------------
# The deployable model
# ---------------------------------------------------------------------------


def _coalitions(blocks: list[str]) -> list[frozenset[str]]:
    out = []
    for r in range(len(blocks) + 1):
        out += [frozenset(c) for c in itertools.combinations(blocks, r)]
    return out


class SalaryModel:
    """Block-dropout LightGBM pair with its own :class:`FeatureBlocks`.

    ``fit`` appends one dropout-masked copy of every training row (ADR 0007) and fits
    the transformer on the augmented frame, so the company encoder is cross-fitted on
    it as well. ``predict`` takes a prepared frame (the derived-table columns, or the
    output of :func:`salary_scout.dataset.prepare_inputs`) with any blocks masked via
    :func:`mask_blocks`.
    """

    def __init__(
        self,
        fb_kwargs: dict | None = None,
        lgbm_params: dict | None = None,
        dropout_p: float = DROPOUT_P,
        seed: int = SEED,
    ):
        self.fb_kwargs = dict(fb_kwargs or {})
        self.lgbm_params = dict(lgbm_params or {})
        self.dropout_p = dropout_p
        self.seed = seed

    # -- fitting ------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> SalaryModel:
        t = time.time()
        aug = pd.concat([df, dropout_blocks(df, p=self.dropout_p, rng=self.seed)], ignore_index=True)
        self.fb_ = FeatureBlocks(**{"random_state": self.seed, **self.fb_kwargs})
        X = self.fb_.fit_transform(aug, aug["log_mid"])
        self.pair_ = PairModel(_lgbm_factory(self.lgbm_params)).fit(
            X, aug["log_mid"].to_numpy(), aug["log_spread"].to_numpy()
        )
        self.baseline_ = float(df["log_mid"].mean())
        self.baseline_spread_ = float(df["log_spread"].mean())
        self.n_train_ = len(df)
        self.fit_seconds_ = time.time() - t
        return self

    # -- prediction ---------------------------------------------------------

    def transform(self, df: pd.DataFrame, masked: Iterable[str] = ()) -> sp.csr_matrix:
        return self.fb_.transform(mask_blocks(df, list(masked)))

    def predict_targets(
        self, df: pd.DataFrame, masked: Iterable[str] = ()
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.pair_.predict(self.transform(df, masked))

    def predict(self, df: pd.DataFrame, masked: Iterable[str] = ()) -> pd.DataFrame:
        """Dollar range per row: columns ``low``, ``high``, ``mid``, ``log_mid``, ``log_spread``."""
        log_mid, log_spread = self.predict_targets(df, masked)
        lo, hi, mid = reconstruct_range(log_mid, log_spread)
        return pd.DataFrame(
            {"low": lo, "high": hi, "mid": mid, "log_mid": log_mid, "log_spread": log_spread},
            index=df.index,
        )

    @staticmethod
    def frame_from_postings(postings: list[dict]) -> pd.DataFrame:
        """Raw posting dicts (as a user or the service would send them) -> prepared frame."""
        return prepare_inputs(pd.DataFrame(postings))

    # -- explanations -------------------------------------------------------

    def explain(
        self, df: pd.DataFrame, masked: Iterable[str] = (), method: str = "tree_shap"
    ) -> pd.DataFrame:
        """Per-block contributions to ``log_mid``, one row per input row.

        Columns are the five blocks plus ``baseline``; ``baseline + sum(blocks)`` equals
        the predicted ``log_mid`` for both methods. ``exp(value) - 1`` is the effect on
        the midpoint as a fraction.
        """
        masked = list(masked)
        if method == "tree_shap":
            X = self.transform(df, masked)
            contrib = self.pair_.mid_.booster_.predict(X, pred_contrib=True)
            contrib = contrib.toarray() if hasattr(contrib, "toarray") else np.asarray(contrib)
            out = self.fb_.sum_by_block(contrib[:, :-1])
            out["baseline"] = contrib[:, -1]
        elif method == "coalition":
            out = self._coalition_shapley(df, masked)
        else:
            raise ValueError(f"unknown method {method!r}; use 'tree_shap' or 'coalition'")
        out.index = df.index
        return out

    def _coalition_shapley(self, df: pd.DataFrame, masked: list[str]) -> pd.DataFrame:
        """Exact Shapley values over the present blocks; masked blocks contribute 0.

        The value of a coalition is the model's ``log_mid`` with every block outside
        the coalition masked. The empty coalition is worth ``baseline_`` (the training
        mean) rather than an all-masked prediction the model never saw in training.
        """
        present = [b for b in BLOCK_NAMES if b not in masked]
        n = len(present)
        value: dict[frozenset[str], np.ndarray] = {}
        for coalition in _coalitions(present):
            if not coalition:
                value[coalition] = np.full(len(df), self.baseline_)
            else:
                hide = [b for b in BLOCK_NAMES if b not in coalition]
                value[coalition] = self.predict_targets(df, hide)[0]
        out = pd.DataFrame(0.0, index=range(len(df)), columns=BLOCK_NAMES)
        for b in present:
            others = [o for o in present if o != b]
            phi = np.zeros(len(df))
            for coalition in _coalitions(others):
                s = len(coalition)
                w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                phi += w * (value[coalition | {b}] - value[coalition])
            out[b] = phi
        out["baseline"] = self.baseline_
        return out

    # -- persistence --------------------------------------------------------

    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL_PATH) -> SalaryModel:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not hold a {cls.__name__}")
        return model


def train_final(
    derived: pd.DataFrame | None = None,
    path: Path | str | None = DEFAULT_MODEL_PATH,
    rows: str = "train",
    **kwargs,
) -> SalaryModel:
    """Fit the deployable model and save it.

    ``rows="train"`` fits on the train split only, so the time-holdout numbers in the
    report describe exactly this model. ``rows="all"`` uses every row. ``kwargs`` go to
    :class:`SalaryModel` (``fb_kwargs``, ``lgbm_params``, ...).
    """
    df = load_derived() if derived is None else derived
    if rows == "train":
        df = df[df["split"] == "train"].reset_index(drop=True)
    elif rows != "all":
        raise ValueError("rows must be 'train' or 'all'")
    model = SalaryModel(**kwargs).fit(df)
    if path is not None:
        model.save(path)
    return model


if __name__ == "__main__":
    m = train_final()
    print(f"trained on {m.n_train_:,} rows in {m.fit_seconds_:.0f}s -> {DEFAULT_MODEL_PATH}")
