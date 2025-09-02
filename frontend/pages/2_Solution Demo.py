import os, io, requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

API_URL = os.getenv("API_URL", "http://localhost:8080/predict")

st.title("📡 Forecast via FastAPI (RNN)")
st.caption(f"Using API: {API_URL}")

# -------------------------------
# Helpers
# -------------------------------
def load_array_from_file(f, label):
    ext = os.path.splitext(f.name)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(f)
        st.write(f"{label} CSV preview:", df.head())
        numeric = df.select_dtypes(include=[np.number]).to_numpy()
        return numeric
    elif ext == ".npy":
        return np.load(f, allow_pickle=False)
    elif ext == ".npz":
        npz = np.load(f, allow_pickle=False)
        keys = list(npz.keys())
        st.write(f"Keys in {label} npz:", keys)
        key = keys[0] if len(keys) == 1 else st.selectbox(f"Select array key for {label}:", keys, key=f"{label}_key")
        return npz[key]
    else:
        st.error(f"Unsupported file type for {label}")
        st.stop()

def ensure_3d_X(X):
    X = np.asarray(X)
    if X.ndim == 3:
        return X
    if X.ndim == 2:
        st.info("Detected 2D array for X_test. Provide timesteps to reshape to (n_samples, timesteps, n_features).")
        timesteps = st.number_input("Timesteps (for reshaping X_test)", min_value=1, value=1, step=1, key="ts")
        if timesteps <= 1:
            st.warning("Timesteps=1 → model sees single-step sequences. If your model needs T>1, increase this.")
        n_samples = X.shape[0]
        n_features_total = X.shape[1]
        if n_features_total % timesteps != 0:
            st.error(f"Cannot reshape: features={n_features_total} not divisible by timesteps={timesteps}.")
            st.stop()
        n_features = n_features_total // timesteps
        return X.reshape(n_samples, timesteps, n_features)
    if X.ndim == 1:
        st.error("X_test must be at least 2D to form sequences. Upload a pre-windowed 3D array or CSV with enough columns to reshape.")
        st.stop()

def reduce_preds_to_1d(preds):
    """
    Convert various RNN output shapes to a 1D series for plotting/metrics.
    - (N,) -> return as is
    - (N,1) -> squeeze
    - (N,H) multi-horizon -> choose horizon h
    - (N,T,1) -> take last step by default (selectable)
    - (N,T,H) -> choose step and/or horizon
    """
    preds = np.asarray(preds)
    if preds.ndim == 1:
        return preds, {"mode": "1d", "info": preds.shape}
    if preds.ndim == 2:
        if preds.shape[1] == 1:
            return preds.ravel(), {"mode": "Nx1", "info": preds.shape}
        # Multi-horizon
        h = st.number_input("Select horizon index (0-based)", min_value=0, max_value=preds.shape[1]-1, value=0, step=1, key="h_idx")
        return preds[:, h], {"mode": "NxH", "h": h, "info": preds.shape}
    if preds.ndim == 3:
        N, T, K = preds.shape
        if K == 1:
            step = st.number_input("Which time step to plot? (0..T-1)", min_value=0, max_value=T-1, value=T-1, step=1, key="t_idx")
            return preds[:, step, 0], {"mode": "NxTx1", "t": step, "info": preds.shape}
        # General (N,T,K): pick step and horizon
        step = st.number_input("Select time step (0..T-1)", min_value=0, max_value=T-1, value=T-1, step=1, key="t_idx2")
        hor = st.number_input("Select horizon/feature (0..K-1)", min_value=0, max_value=K-1, value=0, step=1, key="k_idx2")
        return preds[:, step, hor], {"mode": "NxTxK", "t": step, "k": hor, "info": preds.shape}
    st.error(f"Unsupported prediction shape: {preds.shape}")
    st.stop()

def reduce_y_test_to_1d(y, target_len, hint=None):
    y = np.asarray(y)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        if y.shape[1] == 1:
            return y.ravel()
        h = st.number_input("Select horizon for y_test (0-based)", min_value=0, max_value=y.shape[1]-1, value=hint.get("h", 0) if hint else 0, step=1, key="y_h_idx")
        return y[:, h]
    if y.ndim == 3:
        N, T, K = y.shape
        t_def = hint.get("t", T-1) if hint else T-1
        k_def = hint.get("k", 0) if hint else 0
        t = st.number_input("Select time step for y_test", min_value=0, max_value=T-1, value=t_def, step=1, key="y_t_idx")
        k = st.number_input("Select horizon/feature for y_test", min_value=0, max_value=K-1, value=k_def, step=1, key="y_k_idx")
        return y[:, t, k]
    st.error(f"Unsupported y_test shape: {y.shape}")
    st.stop()

# -------------------------------
# 1) Upload X_test (3D for RNN)
# -------------------------------
x_file = st.file_uploader("Upload X_test (.csv, .npy, .npz) — expects (N, T, F)", type=["csv", "npy", "npz"])

y_pred = None
pred_info = None
X = None

if x_file is not None:
    X_raw = load_array_from_file(x_file, "X_test")
    X = ensure_3d_X(X_raw)
    st.write("X_test shape (sending to API):", X.shape)

    # Serialize as .npy to the API
    buf = io.BytesIO()
    np.save(buf, X)
    buf.seek(0)
    files = {"file": ("input.npy", buf, "application/octet-stream")}

    try:
        with st.spinner("Contacting API and running prediction..."):
            r = requests.post(API_URL, files=files, timeout=60)
            r.raise_for_status()
            resp = r.json()
        preds = np.array(resp.get("prediction", []))
        if preds.size == 0:
            st.error("API returned no predictions.")
            st.stop()
        # Normalize predictions to 1D for plots/metrics
        y_pred, pred_info = reduce_preds_to_1d(preds)
        if "input_shape" in resp:
            st.caption(f"API reported input_shape: {resp['input_shape']}")
        st.success("Prediction complete via API!")
        st.subheader("Predictions (y_pred) — first few")
        st.write(y_pred[:10], "...")
        st.caption(f"Raw prediction shape: {preds.shape} | reduced via {pred_info}")
    except requests.exceptions.RequestException as re:
        st.error(f"API request failed: {re}")
        st.info("Check that your FastAPI service is running and API_URL is correct.")
    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# 2) (Optional) Upload y_test for evaluation/plot
# -------------------------------
if y_pred is not None:
    y_file = st.file_uploader("Optional: Upload y_test (.csv, .npy, .npz)", type=["csv", "npy", "npz"], key="y_file")
    if y_file is not None:
        y_raw = load_array_from_file(y_file, "y_test")
        y_series = reduce_y_test_to_1d(y_raw, target_len=len(y_pred), hint=pred_info)

        if len(y_series) != len(y_pred):
            st.warning(f"Length mismatch: y_test={len(y_series)} vs y_pred={len(y_pred)}. Skipping comparison.")
        else:
            mae = float(np.mean(np.abs(y_series - y_pred)))
            rmse = float(np.sqrt(np.mean((y_series - y_pred)**2)))
            ss_res = float(np.sum((y_series - y_pred)**2))
            ss_tot = float(np.sum((y_series - np.mean(y_series))**2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            st.subheader("Evaluation")
            st.write({"MAE": mae, "RMSE": rmse, "R2": r2})

            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.plot(y_series, label="Actual (y_test)")
            ax.plot(y_pred, label="Predicted (y_pred)")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)
