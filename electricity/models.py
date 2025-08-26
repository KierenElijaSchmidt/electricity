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


def run_lstm(
    preprocessor,
    X: pd.DataFrame,
    y: pd.Series,
    window: int = 30,
    epochs: int = 5,
    batch_size: int = 32,
    n_splits: int = 5,
) -> pd.DataFrame | None:
    """
    LSTM baseline using the provided preprocessor and feature set, analogous to _tscv_scores.
    Runs TimeSeriesSplit CV, applies the preprocessor, and fits an LSTM on the transformed data.
    Only supports univariate prediction (y).
    """
    if not TF_AVAILABLE:
        return None
    from sklearn.model_selection import TimeSeriesSplit

    # Prepare results
    r2_scores = []
    rmse_scores = []

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X, y))
    if len(splits) == 0:
        return None

    for train_idx, test_idx in splits:
        # Preprocess X
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]

        # Fit preprocessor on train, transform both
        X_train = preprocessor.fit_transform(X_train_raw, y_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        # If X_train is a DataFrame, convert to ndarray
        if hasattr(X_train, "to_numpy"):
            X_train = X_train.to_numpy()
        if hasattr(X_test, "to_numpy"):
            X_test = X_test.to_numpy()

        # Univariate: use only the first column if X_train is 2D with more than 1 feature
        if X_train.ndim == 2 and X_train.shape[1] > 1:
            X_train_seq = X_train[:, 0]
            X_test_seq = X_test[:, 0]
        else:
            X_train_seq = X_train.ravel()
            X_test_seq = X_test.ravel()

        # Prepare sequences for LSTM
        def _prepare_seq(series, window):
            Xs, Ys = [], []
            for i in range(len(series) - window):
                Xs.append(series[i : i + window])
                Ys.append(series[i + window])
            return np.array(Xs)[..., np.newaxis], np.array(Ys)

        Xs_train, Ys_train = _prepare_seq(X_train_seq, window)
        Xs_test, Ys_test = _prepare_seq(X_test_seq, window)

        # If not enough data for a split, skip
        if len(Xs_train) == 0 or len(Xs_test) == 0:
            continue

        # Build and fit LSTM
        model = Sequential([LSTM(64, input_shape=(window, 1)), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(Xs_train, Ys_train, epochs=epochs, batch_size=batch_size, verbose=0)

        preds = model.predict(Xs_test, verbose=0).flatten()
        r2 = r2_score(Ys_test, preds)
        rmse = float(np.sqrt(mean_squared_error(Ys_test, preds)))
        r2_scores.append(r2)
        rmse_scores.append(rmse)

    if not r2_scores:
        return None

    return pd.DataFrame([{
        "Model": "LSTM",
        "R2_mean": float(np.mean(r2_scores)),
        "R2_std": float(np.std(r2_scores)),
        "RMSE_mean": float(np.mean(rmse_scores)),
        "RMSE_std": float(np.std(rmse_scores)),
    }])
