import os, io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tensorflow as tf  # ← NEW: we run the model locally

# --- Settings / quiet logs (same intent as your API) -------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ✅ Make this page behave like the others
st.set_page_config(page_title="Live AI Demo", page_icon="📡", layout="wide", initial_sidebar_state="collapsed")

# ✅ Same title CSS used everywhere
st.markdown("""
    <style>
    .section-title {
        font-size: 34px;
        font-weight: 800;
        margin: 12px 0 24px 0;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #ffffff;
    }
    .section-title span.icon {
        font-size: 34px; /* same as text */
        line-height: 1;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title (same structure as other pages) ---
st.markdown('<div class="section-title"><span class="icon">📡</span> Live AI Demo</div>', unsafe_allow_html=True)

# -------------------------------
# Model discovery + loading (mirrors your API)
# -------------------------------
def _find_default_model() -> Path:
    """Look for a model in common places so it works locally and in containers."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "results" / "rnn_model.keras",          # ./results/rnn_model.keras
        here / "rnn_model.keras",                      # ./rnn_model.keras
        here.parent / "results" / "rnn_model.keras",   # ../results/rnn_model.keras
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # default suggestion even if it doesn't exist yet

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_find_default_model()))).expanduser()

@st.cache_resource(show_spinner=True)
def load_keras_model(model_path: Path):
    # (Optional) friendlier GPU behavior
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

try:
    with st.spinner("Loading model…"):
        model = load_keras_model(MODEL_PATH)
    st.caption(f"Model loaded from: {MODEL_PATH}")
    # Helpful debug: what the model expects/produces
    try:
        st.caption(f"Model input_shape: {model.input_shape} → output_shape: {model.output_shape}")
    except Exception:
        pass
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# -------------------------------
# Helpers (unchanged)
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
        # If model has a fixed timestep, offer it as a default
        default_ts = 1
        try:
            # model.input_shape often looks like (None, T, F)
            if isinstance(model.input_shape, tuple) and len(model.input_shape) >= 3 and model.input_shape[1] is not None:
                default_ts = int(model.input_shape[1])
        except Exception:
            pass

        timesteps = st.number_input("Timesteps (for reshaping X_test)", min_value=1, value=default_ts, step=1, key="ts")
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
    preds = np.asarray(preds)
    if preds.ndim == 1:
        return preds, {"mode": "1d", "info": preds.shape}
    if preds.ndim == 2:
        if preds.shape[1] == 1:
            return preds.ravel(), {"mode": "Nx1", "info": preds.shape}
        h = st.number_input("Select horizon index (0-based)", min_value=0, max_value=preds.shape[1]-1, value=0, step=1, key="h_idx")
        return preds[:, h], {"mode": "NxH", "h": h, "info": preds.shape}
    if preds.ndim == 3:
        N, T, K = preds.shape
        if K == 1:
            step = st.number_input("Which time step to plot? (0..T-1)", min_value=0, max_value=T-1, value=T-1, step=1, key="t_idx")
            return preds[:, step, 0], {"mode": "NxTx1", "t": step, "info": preds.shape}
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
x_file = st.file_uploader("Upload your input data with features such as weather, past prices or date.", type=["csv", "npy", "npz"])

y_pred = None
pred_info = None
X = None

if x_file is not None:
    X_raw = load_array_from_file(x_file, "X_test")
    X = ensure_3d_X(X_raw)
    st.write("The shape (e.g., dimensions) of your data is:", X.shape)

    # --- LOCAL INFERENCE (no API) -------------------------------------------
    try:
        # dtype often matters for TF speed/compat
        X = np.asarray(X).astype(np.float32)

        with st.spinner("Running prediction locally…"):
            preds = model.predict(X, verbose=0)

        preds = np.array(preds)
        if preds.size == 0:
            st.error("Model returned no predictions.")
            st.stop()

        # Normalize predictions to 1D for plots/metrics
        y_pred, pred_info = reduce_preds_to_1d(preds)

        st.success("Prediction complete!")
        st.subheader("First 10 Predictions")
        st.write(y_pred[:10], "…")
        st.caption(f"Raw prediction tensor shape: {preds.shape}")

    except Exception as e:
        st.error(f"Error during local inference: {e}")

# -------------------------------
# 2) (Optional) Upload y_test for evaluation/plot
# -------------------------------
if y_pred is not None:
    y_file = st.file_uploader("Optional: Upload data showing the actual prices", type=["csv", "npy", "npz"], key="y_file")
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

            baseline_mae = st.number_input(
                "Baseline MAE",
                min_value=0.0,
                value=float(44.58),
                step=0.01,
                help="Baseline reference to compare against."
            )

            mae_now = float(mae)
            delta_abs = mae_now - baseline_mae           # negative = improvement
            improve_pct = (baseline_mae - mae_now) / baseline_mae * 100.0 if baseline_mae > 0 else float("nan")

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE (↓ better)", f"{mae_now:.2f}", delta=f"{delta_abs:+.2f}", delta_color="inverse")
            c2.metric("Baseline MAE", f"{baseline_mae:.2f}")
            if baseline_mae > 0:
                c3.metric("Improvement vs baseline", f"{improve_pct:.1f}%", delta=f"{improve_pct:+.1f}%", delta_color="normal")
            else:
                c3.metric("Improvement vs baseline", "—")

            st.subheader("Chart: Actual vs Predicted Values")
            _y_true = np.array(y_series).reshape(-1)
            _y_pred = np.array(y_pred).reshape(-1)
            n_total = int(min(_y_true.shape[0], _y_pred.shape[0]))

            if n_total == 0:
                st.warning("No data to plot. Check y_test / y_pred.")
            else:
                default_n = 200 if n_total >= 200 else n_total
                n_last = st.slider(
                    "Show last N days",
                    min_value=10,
                    max_value=n_total,
                    value=default_n,
                    step=10,
                    help="Adjust to zoom into the recent portion of the series."
                )

                y_true_zoom = _y_true[-n_last:]
                y_pred_zoom = _y_pred[-n_last:]

                fig_zoom, ax_zoom = plt.subplots(figsize=(6, 3), dpi=120)
                ax_zoom.plot(y_true_zoom, label="Actual prices")
                ax_zoom.plot(y_pred_zoom, label="Predicted prices")
                ax_zoom.set_xlabel("Last N Days")
                ax_zoom.set_ylabel("Electricity Price (RRP)")
                ax_zoom.legend(fontsize=9)
                ax_zoom.tick_params(labelsize=9)
                fig_zoom.tight_layout()
                st.pyplot(fig_zoom, use_container_width=False)

                st.subheader("Learning Curves: Showing the error of the model over time")
                img_path = Path(__file__).resolve().parents[1] / "frontend" / "assets" / "curves" / "output.png"
                try:
                    image = Image.open(img_path)
                    st.image(image, caption="Training vs. Validation (Learning Curves)", width=560, use_container_width=False)
                except FileNotFoundError:
                    st.warning(f"Image not found at: {img_path}\nMake sure the file exists and the path is correct.")
