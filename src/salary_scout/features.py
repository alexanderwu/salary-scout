"""Maskable feature blocks (ADR 0007) on top of the derived table.

Five named blocks map straight onto the UI: ``title``, ``description``,
``role_meta``, ``location``, ``company``. :class:`FeatureBlocks` is a thin wrapper
around a scikit-learn ``ColumnTransformer`` with one sub-transformer per feature
family, named ``<block>:<family>``, so that after fitting ``block_slices_`` says which
output columns belong to which block. Attributions are summed per block with it.

Masking contract
----------------
Each block has a presence column ``present_<block>`` (1.0 by default). Masking a
block, :func:`mask_blocks`, sets its source columns to NaN and the presence
column to 0. Every sub-transformer tolerates NaN input, so a masked block simply
produces zeros (hashed text, one-hot), imputed values plus a missing indicator
(numerics), or the prior (company target encoding). :func:`dropout_blocks` is the
training-time augmentation: it masks random blocks per row and always leaves at
least one block present.

Deployment constraint (ADR 0008): every text featuriser is hash based, so the
browser demo can reproduce it without shipping a vocabulary. Only the company
encoder and the top-K tool vocabulary need a lookup table.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from salary_scout.dataset import parse_json_list

# ---------------------------------------------------------------------------
# Block definitions
# ---------------------------------------------------------------------------

BLOCKS: dict[str, list[str]] = {
    "title": ["title_clean", "core_job_title_clean"],
    "description": ["description_clean", "requirements_summary_clean"],
    "role_meta": [
        "seniority_level",
        "job_category",
        "min_industry_and_role_yoe",
        "min_management_yoe",
        "commitment",
        "bachelors_degree_requirement",
        "masters_degree_requirement",
        "doctorate_degree_requirement",
        "security_clearance",
        "visa_sponsorship",
        "technical_tools",
    ],
    "location": [
        "workplace_type",
        "primary_state",
        "n_states",
        "workplace_countries",
        "latitude",
        "longitude",
        "is_remote",
    ],
    "company": [
        "company_name",
        "nb_employees",
        "year_founded",
        "organization_type",
        "company_sector_and_industry",
        "company_industries",
        "hq_country",
        "source",
    ],
}
BLOCK_NAMES = list(BLOCKS)

# Used only to cross-fit the company target encoder; never a feature, never masked.
GROUP_COLUMN = "collapse_key"

SENIORITY_ORDER = {
    "No Prior Experience Required": 0.0,
    "Entry Level": 1.0,
    "Mid Level": 2.0,
    "Senior Level": 3.0,
}
REFERENCE_YEAR = 2026


def presence_column(block: str) -> str:
    return f"present_{block}"


# ---------------------------------------------------------------------------
# Masking and dropout
# ---------------------------------------------------------------------------


def mask_blocks(
    X: pd.DataFrame, blocks: Iterable[str], rows: np.ndarray | pd.Index | None = None
) -> pd.DataFrame:
    """Return a copy of ``X`` with the given blocks masked, optionally on a row subset.

    ``rows`` is a boolean array aligned with ``X`` or an index label subset. A
    masked block has NaN in all its source columns and 0 in its presence column.
    """
    X = X.copy()
    for block in BLOCKS:
        if presence_column(block) not in X:
            X[presence_column(block)] = 1.0
    row_sel = slice(None) if rows is None else rows
    for block in blocks:
        if block not in BLOCKS:
            raise KeyError(f"unknown block {block!r}; expected one of {BLOCK_NAMES}")
        for col in BLOCKS[block]:
            if col in X:
                X.loc[row_sel, col] = np.nan
        X.loc[row_sel, presence_column(block)] = 0.0
    return X


def dropout_blocks(
    X: pd.DataFrame,
    p: float = 0.3,
    rng: np.random.Generator | int | None = None,
    min_present: int = 1,
) -> pd.DataFrame:
    """Training-time augmentation: mask each block per row with probability ``p``.

    Rows that would lose more than ``len(BLOCKS) - min_present`` blocks get random
    blocks restored so every row keeps at least ``min_present`` present.
    """
    rng = np.random.default_rng(rng)
    n, k = len(X), len(BLOCK_NAMES)
    drop = rng.random((n, k)) < p
    too_many = drop.sum(axis=1) > k - min_present
    for i in np.flatnonzero(too_many):
        keep = rng.choice(k, size=min_present, replace=False)
        drop[i, keep] = False
    out = X
    for j, block in enumerate(BLOCK_NAMES):
        if drop[:, j].any():
            out = mask_blocks(out, [block], rows=drop[:, j])
    if out is X:
        out = mask_blocks(X, [])  # ensure presence columns exist on a copy
    return out


# ---------------------------------------------------------------------------
# Small stateless helpers used inside FunctionTransformers
# ---------------------------------------------------------------------------


def _as_series(X) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0]
    if isinstance(X, pd.Series):
        return X
    return pd.Series(np.asarray(X).ravel())


def _text_list(X, max_chars: int | None = None) -> list[str]:
    s = _as_series(X).fillna("").astype(str)
    if max_chars:
        s = s.str.slice(0, max_chars)
    return s.tolist()


def _log_len(X) -> np.ndarray:
    s = _as_series(X).fillna("").astype(str)
    return np.log1p(s.str.len().to_numpy(dtype=float)).reshape(-1, 1)


def _to_2d(X) -> np.ndarray:
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _log1p_2d(X) -> np.ndarray:
    return np.log1p(np.clip(_to_2d(X), 0, None))


def _company_age(X) -> np.ndarray:
    age = REFERENCE_YEAR - _to_2d(X)
    age[(age < 0) | (age > 300)] = np.nan
    return age


def _seniority_ordinal(X) -> np.ndarray:
    return _as_series(X).map(SENIORITY_ORDER).to_numpy(dtype=float).reshape(-1, 1)


def _as_object_frame(X) -> pd.DataFrame:
    """One-hot input: object dtype with real NaN so OneHotEncoder treats missing uniformly."""
    df = pd.DataFrame(X).astype(object)
    return df.where(df.notna(), np.nan)


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------


class MultiHotTopK(BaseEstimator, TransformerMixin):
    """Multi-hot encode a JSON-list column: top ``top_k`` values by frequency get their
    own column; anything else is hashed into ``n_hash`` buckets. A final column holds
    log1p of the list length. NaN input yields an all-zero row."""

    def __init__(self, top_k: int = 100, n_hash: int = 0, lowercase: bool = True):
        self.top_k = top_k
        self.n_hash = n_hash
        self.lowercase = lowercase

    def _lists(self, X) -> list[list[str]]:
        items = _as_series(X).map(parse_json_list)
        if self.lowercase:
            items = items.map(lambda xs: [x.lower() for x in xs])
        return items.tolist()

    def fit(self, X, y=None):
        counts = pd.Series([v for xs in self._lists(X) for v in xs]).value_counts()
        self.vocab_ = {v: i for i, v in enumerate(counts.index[: self.top_k])}
        if self.n_hash:
            self.hasher_ = FeatureHasher(
                n_features=self.n_hash, input_type="string", alternate_sign=False
            )
        return self

    def transform(self, X) -> sp.csr_matrix:
        lists = self._lists(X)
        rows, cols = [], []
        rest: list[list[str]] = []
        for i, xs in enumerate(lists):
            leftovers = []
            for v in xs:
                j = self.vocab_.get(v)
                if j is None:
                    leftovers.append(v)
                else:
                    rows.append(i)
                    cols.append(j)
            rest.append(leftovers)
        n = len(lists)
        top = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(self.vocab_)))
        top.data[:] = 1.0  # duplicates within one list collapse to 1
        parts = [top]
        if self.n_hash:
            hashed = self.hasher_.transform(rest).tocsr()
            hashed.data[:] = 1.0
            parts.append(hashed)
        length = sp.csr_matrix(np.log1p([len(xs) for xs in lists]).reshape(-1, 1))
        parts.append(length)
        return sp.hstack(parts, format="csr")

    def get_feature_names_out(self, input_features=None):
        names = [f"top:{v}" for v in self.vocab_]
        if self.n_hash:
            names += [f"hash:{i}" for i in range(self.n_hash)]
        return np.asarray(names + ["log_count"], dtype=object)


class GroupTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed mean-target encoding of a high-cardinality category, cross-fitted by
    group during ``fit_transform`` so sibling postings cannot leak their salary
    into each other's encoding.

    Expects two input columns: ``[category, group]``. Emits two output columns:
    the encoded target and log1p of the category's training frequency. Unknown or
    missing categories get the prior and a frequency of zero.
    """

    def __init__(self, smooth: float = 10.0, n_splits: int = 5, random_state: int = 0):
        self.smooth = smooth
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _split(X) -> tuple[pd.Series, pd.Series]:
        df = pd.DataFrame(X)
        cat = df.iloc[:, 0].astype(object).where(df.iloc[:, 0].notna(), np.nan)
        grp = df.iloc[:, 1].astype(object).fillna("(no group)")
        return cat.reset_index(drop=True), grp.reset_index(drop=True)

    def _stats(self, cat: pd.Series, y: np.ndarray) -> tuple[pd.Series, pd.Series]:
        frame = pd.DataFrame({"cat": cat, "y": y}).dropna(subset=["cat"])
        agg = frame.groupby("cat")["y"].agg(["sum", "count"])
        return agg["sum"], agg["count"]

    def _encode(self, cat: pd.Series, sums: pd.Series, counts: pd.Series, prior: float):
        s = cat.map(sums).astype(float).fillna(0.0).to_numpy()
        n = cat.map(counts).astype(float).fillna(0.0).to_numpy()
        denom = n + self.smooth
        enc = np.where(denom > 0, (s + self.smooth * prior) / np.where(denom > 0, denom, 1), prior)
        return np.column_stack([enc, np.log1p(n)])

    def fit(self, X, y):
        cat, _ = self._split(X)
        y = np.asarray(y, dtype=float).ravel()
        self.prior_ = float(np.mean(y))
        self.sums_, self.counts_ = self._stats(cat, y)
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        cat, grp = self._split(X)
        y = np.asarray(y, dtype=float).ravel()
        out = np.empty((len(cat), 2))
        k = min(self.n_splits, grp.nunique())
        if k < 2:
            return self._encode(cat, self.sums_, self.counts_, self.prior_)
        gkf = GroupKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        for fit_idx, enc_idx in gkf.split(cat, groups=grp):
            sums, counts = self._stats(cat.iloc[fit_idx], y[fit_idx])
            prior = float(y[fit_idx].mean())
            out[enc_idx] = self._encode(cat.iloc[enc_idx], sums, counts, prior)
        return out

    def transform(self, X):
        cat, _ = self._split(X)
        return self._encode(cat, self.sums_, self.counts_, self.prior_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["target_enc", "log_freq"], dtype=object)


# ---------------------------------------------------------------------------
# The block transformer
# ---------------------------------------------------------------------------


def _numeric(func, **kw):
    """FunctionTransformer -> median imputer with a missing indicator."""
    return make_pipeline(
        FunctionTransformer(func, kw_args=kw or None),
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
    )


def _presence():
    return FunctionTransformer(_to_2d)


class FeatureBlocks(BaseEstimator, TransformerMixin):
    """Sparse feature matrix organised in five maskable blocks.

    Parameters control the hash sizes so the browser model can be made smaller
    than the service model without changing the code path.

    After ``fit``:

    - ``block_slices_``: ``{block: slice}`` into the output columns, contiguous
      and disjoint, covering every column.
    - ``layout_``: DataFrame with one row per sub-transformer (block, name, start,
      stop, width) for inspection in notebooks.
    """

    def __init__(
        self,
        title_features: int = 2**16,
        description_features: int = 2**18,
        description_max_chars: int = 30_000,
        tools_top_k: int = 300,
        tools_hash: int = 2**12,
        industries_top_k: int = 100,
        min_frequency: int = 20,
        company_smooth: float = 10.0,
        random_state: int = 0,
    ):
        self.title_features = title_features
        self.description_features = description_features
        self.description_max_chars = description_max_chars
        self.tools_top_k = tools_top_k
        self.tools_hash = tools_hash
        self.industries_top_k = industries_top_k
        self.min_frequency = min_frequency
        self.company_smooth = company_smooth
        self.random_state = random_state

    # -- construction -------------------------------------------------------

    def _onehot(self):
        return make_pipeline(
            FunctionTransformer(_as_object_frame),
            OneHotEncoder(handle_unknown="ignore", min_frequency=self.min_frequency),
        )

    def _hashing(self, n_features: int, ngram_range=(1, 1), max_chars: int | None = None):
        return make_pipeline(
            FunctionTransformer(_text_list, kw_args={"max_chars": max_chars}),
            HashingVectorizer(
                n_features=n_features,
                ngram_range=ngram_range,
                alternate_sign=False,
                binary=True,
                norm="l2",
                stop_words="english" if ngram_range == (1, 1) else None,
            ),
        )

    def _transformers(self) -> list[tuple[str, object, list[str] | str]]:
        t: list[tuple[str, object, list[str] | str]] = []
        # title
        t += [
            ("title:present", _presence(), [presence_column("title")]),
            ("title:hash", self._hashing(self.title_features, (1, 2)), "title_clean"),
            ("title:core_hash", self._hashing(self.title_features, (1, 2)), "core_job_title_clean"),
            ("title:len", FunctionTransformer(_log_len), "title_clean"),
        ]
        # description
        t += [
            ("description:present", _presence(), [presence_column("description")]),
            (
                "description:hash",
                self._hashing(self.description_features, (1, 1), self.description_max_chars),
                "description_clean",
            ),
            (
                "description:req_hash",
                self._hashing(self.description_features // 4, (1, 1)),
                "requirements_summary_clean",
            ),
            ("description:len", FunctionTransformer(_log_len), "description_clean"),
        ]
        # role_meta
        t += [
            ("role_meta:present", _presence(), [presence_column("role_meta")]),
            (
                "role_meta:onehot",
                self._onehot(),
                [
                    "seniority_level",
                    "job_category",
                    "bachelors_degree_requirement",
                    "masters_degree_requirement",
                    "doctorate_degree_requirement",
                    "security_clearance",
                    "visa_sponsorship",
                ],
            ),
            ("role_meta:seniority", _numeric(_seniority_ordinal), ["seniority_level"]),
            (
                "role_meta:yoe",
                _numeric(_to_2d),
                ["min_industry_and_role_yoe", "min_management_yoe"],
            ),
            ("role_meta:commitment", MultiHotTopK(top_k=10), "commitment"),
            (
                "role_meta:tools",
                MultiHotTopK(top_k=self.tools_top_k, n_hash=self.tools_hash),
                "technical_tools",
            ),
        ]
        # location
        t += [
            ("location:present", _presence(), [presence_column("location")]),
            ("location:onehot", self._onehot(), ["workplace_type", "primary_state"]),
            ("location:countries", MultiHotTopK(top_k=20), "workplace_countries"),
            ("location:numeric", _numeric(_to_2d), ["latitude", "longitude", "n_states"]),
            ("location:remote", _numeric(_to_2d), ["is_remote"]),
        ]
        # company
        t += [
            ("company:present", _presence(), [presence_column("company")]),
            (
                "company:name",
                GroupTargetEncoder(smooth=self.company_smooth, random_state=self.random_state),
                ["company_name", GROUP_COLUMN],
            ),
            ("company:headcount", _numeric(_log1p_2d), ["nb_employees"]),
            ("company:age", _numeric(_company_age), ["year_founded"]),
            (
                "company:onehot",
                self._onehot(),
                ["organization_type", "company_sector_and_industry", "hq_country", "source"],
            ),
            (
                "company:industries",
                MultiHotTopK(top_k=self.industries_top_k, n_hash=2**8),
                "company_industries",
            ),
        ]
        return t

    # -- sklearn API --------------------------------------------------------

    @staticmethod
    def _target(y) -> np.ndarray | None:
        """The company encoder is fitted on ``log_mid`` only."""
        if y is None:
            return None
        if isinstance(y, pd.DataFrame):
            y = y["log_mid"] if "log_mid" in y else y.iloc[:, 0]
        y = np.asarray(y, dtype=float)
        return y[:, 0] if y.ndim == 2 else y

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill in presence columns, the group column, and any absent source columns."""
        X = X.copy()
        for block, cols in BLOCKS.items():
            if presence_column(block) not in X:
                X[presence_column(block)] = 1.0
            for col in cols:
                if col not in X:
                    X[col] = np.nan
        if GROUP_COLUMN not in X:
            X[GROUP_COLUMN] = np.nan
        return X

    def _finish_fit(self):
        starts: dict[str, list[int]] = {b: [] for b in BLOCKS}
        stops: dict[str, list[int]] = {b: [] for b in BLOCKS}
        rows = []
        for name, sl in self.ct_.output_indices_.items():
            if name == "remainder":
                continue
            block = name.split(":", 1)[0]
            starts[block].append(sl.start)
            stops[block].append(sl.stop)
            rows.append((block, name, sl.start, sl.stop, sl.stop - sl.start))
        self.layout_ = pd.DataFrame(rows, columns=["block", "transformer", "start", "stop", "width"])
        self.block_slices_ = {}
        for block in BLOCK_NAMES:
            lo, hi = min(starts[block]), max(stops[block])
            width = sum(b - a for a, b in zip(starts[block], stops[block]))
            if width != hi - lo:
                raise RuntimeError(f"block {block!r} is not contiguous in the output")
            self.block_slices_[block] = slice(lo, hi)
        self.n_features_out_ = int(self.layout_["stop"].max())
        return self

    def fit(self, X: pd.DataFrame, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> sp.csr_matrix:
        self.ct_ = ColumnTransformer(
            self._transformers(), remainder="drop", sparse_threshold=1.0, n_jobs=None
        )
        out = self.ct_.fit_transform(self._prepare(X), self._target(y))
        self._finish_fit()
        return sp.csr_matrix(out)

    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        return sp.csr_matrix(self.ct_.transform(self._prepare(X)))

    # -- block helpers ------------------------------------------------------

    def block_of(self, column: int) -> str:
        for block, sl in self.block_slices_.items():
            if sl.start <= column < sl.stop:
                return block
        raise IndexError(column)

    def sum_by_block(self, values: np.ndarray) -> pd.DataFrame:
        """Sum per-feature attributions (rows x features) into rows x blocks."""
        values = np.asarray(values)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return pd.DataFrame(
            {b: values[:, sl].sum(axis=1) for b, sl in self.block_slices_.items()}
        )


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------

DEFAULT_LEAK_COLUMNS = [f"{c}_clean" for c in ("title", "description", "requirements_summary")]


def _figure_forms(value: float) -> list[str]:
    if not np.isfinite(value) or value <= 0:
        return []
    forms = []
    if float(value).is_integer():
        v = int(value)
        forms += [f"{v:,}", str(v)]
        if v % 1000 == 0:
            forms.append(f"{v // 1000}k")
    else:
        forms += [f"{value:,.2f}", f"{value:.2f}"]
    return forms


def salary_leak_mask(df: pd.DataFrame, text_columns: list[str] | None = None) -> pd.Series:
    """True where a row's own min or max salary figure survives in its scrubbed text."""
    cols = [c for c in (text_columns or DEFAULT_LEAK_COLUMNS) if c in df]
    text = df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    hits = np.zeros(len(df), dtype=bool)
    lo = df["yearly_min_compensation"].to_numpy(dtype=float)
    hi = df["yearly_max_compensation"].to_numpy(dtype=float)
    for i, (t, a, b) in enumerate(zip(text, lo, hi)):
        hits[i] = any(f.lower() in t for v in (a, b) for f in _figure_forms(v))
    return pd.Series(hits, index=df.index, name="salary_leak")


def check_salary_leakage(
    df: pd.DataFrame, max_rate: float = 0.001, text_columns: list[str] | None = None
) -> float:
    """Return the leak rate; raise if more than ``max_rate`` of rows still show the figure."""
    rate = float(salary_leak_mask(df, text_columns).mean()) if len(df) else 0.0
    if rate > max_rate:
        raise ValueError(
            f"salary figure survives in {rate:.2%} of scrubbed rows (limit {max_rate:.2%})"
        )
    return rate
