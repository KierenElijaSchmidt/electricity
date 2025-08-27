# model.py — drop-in replacement

from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# reuse the unified pipeline builder (date features + preprocessing + corr pruning)
from electricity.preprocessing import Preprocessor

# ---- Optional models ----
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

try:
    import tensorflow as tf
    try:
        # First try the new standalone Keras (>=3.x)
        from keras import Sequential
        from keras.layers import LSTM, Dense
    except ImportError:
        # Fallback to bundled tensorflow.keras
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# models.py — REPLACE the whole _tscv_scores(...) helper with this version

def _tscv_scores(preprocessor, X: pd.DataFrame, y: pd.Series, model, n_splits: int = 5):
    """
    Helper: run TimeSeriesSplit CV with R2 and RMSE on a pipeline that consists of
    the provided preprocessor followed by the specified model.
    Uses an RMSE scorer that doesn't rely on sklearn's 'squared' argument.

    Args:
        preprocessor: The preprocessing step to use in the pipeline.
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        model: The regression model to use (e.g., LinearRegression(), RandomForestRegressor(), etc.).
        n_splits (int): Number of splits for TimeSeriesSplit.

    Returns:
        dict: Dictionary with R2 and RMSE scores (per split and mean).
    """
    from sklearn.pipeline import make_pipeline

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scoring = {
        "r2": make_scorer(r2_score),
        "rmse": make_scorer(
            lambda yt, yp, **kw: float(
                np.sqrt(
                    np.average((np.asarray(yt) - np.asarray(yp)) ** 2, weights=kw.get("sample_weight"))
                )
            )
        ),
    }
    # Compose the pipeline: preprocessor + user-specified model
    pipeline = make_pipeline(preprocessor, model)
    cv = cross_validate(
        pipeline, X, y, cv=tscv, scoring=scoring, return_estimator=False, n_jobs=None
    )
    return {
        "r2_scores": cv["test_r2"],
        "r2_mean": float(np.mean(cv["test_r2"])),
        "rmse_scores": cv["test_rmse"],
        "rmse_mean": float(np.mean(cv["test_rmse"])),
    }


