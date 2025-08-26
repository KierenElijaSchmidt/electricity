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

def run_ml_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    corr_threshold: float = 0.95,
    add_date_features: bool = True,
):
    """
    Build the unified preprocessing pipeline (from preprocessing.Preprocessor)
    and evaluate several classical ML models with TimeSeriesSplit CV.

    Returns:
        summary_df (pd.DataFrame), fitted_pipelines (dict[str, object])
    """
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42,
            tree_method="hist", verbosity=0
        )

    fitted = {}
    rows = []

    for name, est in models.items():
        # Build a fresh pipeline per estimator using the unified Preprocessor
        pre = Preprocessor(
            filepath="",  # not used when we pass data directly
            add_date_features=add_date_features,
            corr_threshold=corr_threshold,
        )
        pre.set_data(pd.concat([X, y.rename(pre.target_col)], axis=1))  # ensures DatetimeIndex etc.
        pipe = pre.build_pipeline(estimator=est)

        scores = _tscv_scores(pipe, X, y, n_splits=n_splits)
        # Fit on full data for downstream use
        pipe.fit(X, y)
        fitted[name] = pipe

        rows.append({
            "Model": name,
            "R2_mean": scores["r2_mean"],
            "R2_std": float(np.std(scores["r2_scores"])),
            "RMSE_mean": scores["rmse_mean"],
            "RMSE_std": float(np.std(scores["rmse_scores"])),
        })

    summary = pd.DataFrame(rows).sort_values("R2_mean", ascending=False).reset_index(drop=True)
    return summary, fitted


def run_arima(y: pd.Series, horizon: int = 30, order=(5, 1, 0)) -> pd.DataFrame | None:
    """
    Simple ARIMA baseline on the target series (train/test split by last `horizon` points).
    """
    if not STATSMODELS_AVAILABLE:
        return None
    if len(y) <= horizon:
        return None

    train, test = y.iloc[:-horizon], y.iloc[-horizon:]
    model = sm.tsa.ARIMA(train, order=order)
    res = model.fit()
    preds = res.forecast(steps=len(test))
    r2 = r2_score(test, preds)
    rmse = float(np.sqrt(mean_squared_error(test, preds)))
    return pd.DataFrame([{
        "Model": "ARIMA",
        "R2_mean": float(r2),
        "R2_std": 0.0,
        "RMSE_mean": rmse,
        "RMSE_std": 0.0,
    }])


def _prepare_seq(series: np.ndarray, window: int = 30):
    Xs, Ys = [], []
    for i in range(len(series) - window):
        Xs.append(series[i:i + window])
        Ys.append(series[i + window])
    Xs = np.array(Xs).reshape(-1, window, 1)
    Ys = np.array(Ys)
    return Xs, Ys


def run_lstm(y: pd.Series, window: int = 30, epochs: int = 5, batch_size: int = 32) -> pd.DataFrame | None:
    """
    Lightweight univariate LSTM baseline on the target series.
    """
    if not TF_AVAILABLE:
        return None
    series = y.values.astype("float32")
    if len(series) <= window + 10:
        return None

    Xs, Ys = _prepare_seq(series, window)
    split = int(len(Xs) * 0.8)
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = Ys[:split], Ys[split:]

    model = Sequential([LSTM(64, input_shape=(window, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    preds = model.predict(X_test, verbose=0).flatten()
    r2 = r2_score(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    return pd.DataFrame([{
        "Model": "LSTM",
        "R2_mean": float(r2),
        "R2_std": 0.0,
        "RMSE_mean": rmse,
        "RMSE_std": 0.0,
    }])


def run_all_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    include_arima: bool = True,
    include_lstm: bool = True,
) -> pd.DataFrame:
    """
    Orchestrates ML pipelines (unified preprocessing), plus optional ARIMA and LSTM.
    Returns a single summary DataFrame.
    """
    results = []

    ml_summary, _ = run_ml_models(X, y, n_splits=n_splits)
    results.append(ml_summary)

    if include_arima:
        arima_res = run_arima(y)
        if arima_res is not None:
            results.append(arima_res)

    if include_lstm:
        lstm_res = run_lstm(y)
        if lstm_res is not None:
            results.append(lstm_res)

    return pd.concat(results, ignore_index=True)
