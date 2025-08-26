# unified_preprocessor.py
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# =========================
# Custom transformers
# =========================
# preprocessing.py — REPLACE the whole DateCyclicalFeatures class with this version
class DateCyclicalFeatures(BaseEstimator, TransformerMixin):
    """Add calendar/cyclical features from DatetimeIndex or a 'date' column.
       Safe: will NOT overwrite if columns already exist (to avoid duplicates)."""
    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X.index, pd.DatetimeIndex):
            dates = X.index
        else:
            if self.date_col not in X.columns:
                raise KeyError(
                    f"DateCyclicalFeatures: '{self.date_col}' not in columns "
                    "and index is not DatetimeIndex."
                )
            dates = pd.to_datetime(X[self.date_col])

        year      = dates.year
        month     = dates.month
        week      = dates.isocalendar().week.astype(int)
        dayofweek = dates.dayofweek
        dayofyear = dates.dayofyear

        def _set(col, series):
            if col not in X.columns:
                X[col] = series

        # raw calendar
        _set("year", year)
        _set("month", month)
        _set("week", week)
        _set("dayofweek", dayofweek)
        _set("dayofyear", dayofyear)

        # cyclical encodings
        _set("month_sin", np.sin(2 * np.pi * month / 12))
        _set("month_cos", np.cos(2 * np.pi * month / 12))
        _set("week_sin",  np.sin(2 * np.pi * week / 52))
        _set("week_cos",  np.cos(2 * np.pi * week / 52))
        _set("doy_sin",   np.sin(2 * np.pi * dayofyear / 365.25))
        _set("doy_cos",   np.cos(2 * np.pi * dayofyear / 365.25))
        return X


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Drop highly correlated features (keep one from any correlated pair)."""
    def __init__(self, threshold: float = 0.95, verbose: bool = True):
        self.threshold = threshold
        self.verbose = verbose
        self.selected_: Optional[list[str]] = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        # keep only numeric for correlation
        num = X_df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            self.selected_ = list(X_df.columns)
            return self
        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > self.threshold)]
        kept = [c for c in X_df.columns if c not in drop]
        self.selected_ = kept
        if self.verbose and drop:
            print(f"🔎 CorrelationSelector dropped {len(drop)} features: {drop}")
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_]


# =========================
# Main unified class
# =========================
@dataclass
class Preprocessor:
    filepath: str | Path
    date_col: str = "date"
    target_col: str = "RRP"
    # columns that may leak label info
    leaky_cols: Iterable[str] = field(default_factory=lambda: (
        "RRP_positive", "RRP_negative",
        "demand_pos_RRP", "demand_neg_RRP",
        "frac_at_neg_RRP",
    ))
    # mapping for binary columns; normalized to uppercase first
    bool_maps: dict = field(default_factory=lambda: {
        "holiday": {"Y": 1, "N": 0},
        "school_day": {"Y": 1, "N": 0},
    })
    corr_threshold: float = 0.95
    add_date_features: bool = True
    random_state: int = 42

    # internal
    df: Optional[pd.DataFrame] = None
    pipeline_: Optional[Pipeline] = None

    # -------------------------
    # Data loading / cleaning
    # -------------------------

    # preprocessing.py — ADD this method inside the Preprocessor class
    def set_data(self, df: pd.DataFrame):
        """Supply an already loaded/cleaned DataFrame (e.g., from load.Loading)."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if self.date_col in df.columns:
                df = df.sort_values(self.date_col).set_index(self.date_col)
            else:
                raise ValueError(
                    f"set_data: DataFrame must have a DatetimeIndex or a '{self.date_col}' column."
                )
        self.df = df
        return self

    # -------------------------
    # Pipeline builder
    # -------------------------
    def build_pipeline(self, estimator=None) -> Pipeline:
        """
        Build an end-to-end pipeline:
          [Date features] -> ColumnTransformer -> CorrelationSelector -> Estimator
        """
        # OneHotEncoder compatibility for sklearn versions:
        ohe_kwargs = dict(handle_unknown="ignore", drop="first")
        try:
            # sklearn >= 1.2
            ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)
        except TypeError:
            # older sklearn
            ohe = OneHotEncoder(sparse=False, **ohe_kwargs)

        num_tf = make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler()
        )
        cat_tf = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="missing"),
            ohe
        )

        num_sel = make_column_selector(dtype_include=["int64", "float64", "Int8", "Int16", "Int32", "Int64", "float32", "float64"])
        cat_sel = make_column_selector(dtype_include=["object", "string", "category"])

        pre_ct = make_column_transformer(
            (num_tf, num_sel),
            (cat_tf, cat_sel),
            remainder="drop"
        )

        steps = []
        if self.add_date_features:
            steps.append(("date_features", DateCyclicalFeatures(self.date_col)))
        steps.append(("pre", pre_ct))
        steps.append(("corr_prune", CorrelationSelector(threshold=self.corr_threshold, verbose=True)))
        steps.append(("est", estimator if estimator is not None else LinearRegression()))

        self.pipeline_ = Pipeline(steps)
        return self.pipeline_

    # -------------------------
    # Train & evaluate
    # -------------------------
    def evaluate(self, n_splits: int = 5, estimator=None):
        """
        TimeSeriesSplit cross-validation with R^2 and RMSE.
        Returns scores and a fitted pipeline on the full data.
        """
        if self.df is None:
            self.load_data()
        if self.pipeline_ is None or estimator is not None:
            self.build_pipeline(estimator=estimator)

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        tscv = TimeSeriesSplit(n_splits=n_splits)

        scoring = {
            "r2": make_scorer(r2_score),
            # RMSE scorer compatible with older sklearn (no 'squared' kw) and supports sample_weight
            "rmse": make_scorer(
                lambda yt, yp, **kw: float(
                    np.sqrt(
                        np.average((np.asarray(yt) - np.asarray(yp)) ** 2, weights=kw.get("sample_weight"))
                    )
                )
            ),
        }



        # cross-validate the full pipeline (no leakage)
        cv = cross_validate(self.pipeline_, X, y, cv=tscv, scoring=scoring, n_jobs=None, return_train_score=False)
        # fit on full data for downstream use
        self.pipeline_.fit(X, y)

        return {
            "pipeline": self.pipeline_,
            "r2_scores": cv["test_r2"],
            "r2_mean": float(np.mean(cv["test_r2"])),
            "rmse_scores": cv["test_rmse"],
            "rmse_mean": float(np.mean(cv["test_rmse"])),
        }
