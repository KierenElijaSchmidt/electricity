import os
import io
import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

API_URL = os.getenv("API_URL", "http://localhost:8080/predict")

st.title("📡 Forecast via FastAPI")
st.caption(f"Using API: {API_URL}")

# -------------------------------
# 1) Upload X_test (features)
# -------------------------------
x_file = st.file_uploader("Upload X_test (.csv, .npy, .npz)", type=["csv", "npy", "npz"])

def load_X_from_file(f):
    ext = os.path.splitext(f.name)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(f)
        st.write("CSV preview:", df.head())
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found for X_test.")
            st.stop()
        feat_cols = st.multiselect("Select feature columns for X_test:", numeric_cols, default=numeric_cols)
        if not feat_cols:
            st.error("Please select at least one feature column.")
            st.stop()
        X = df[feat_cols].to_numpy()
        return X
    elif ext == ".npy":
        X = np.load(f, allow_pickle=False)
        return X
    elif ext == ".npz":
        npz = np.load(f, allow_pickle=False)
        keys = list(npz.keys())
        st.write("Keys in npz:", keys)
        key = keys[0] if len(keys) == 1 else st.selectbox("Select array key for X_test:", keys)
        X = npz[key]
        return X
    else:
        st.error("Unsupported file type for X_test")
        st.stop()

def ensure_2d_X(X):
    X = np.asarray(X)
    if X.ndim == 1:
        st.warning("Uploaded array is 1-D; treating as a single feature. Reshaping to (-1, 1).")
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        st.error(f"X_test must be 2-D (n_samples, n_features). Got shape {X.shape}.")
        st.stop()
    return X

y_pred = None
X = None

if x_file is not None:
    # Load and validate X_test
    X = load_X_from_file(x_file)
    X = ensure_2d_X(X)
    st.write("X_test shape being sent to API:", X.shape)

    # Serialize X_test as .npy in-memory
    buf = io.BytesIO()
    np.save(buf, X)
    buf.seek(0)
    files = {"file": ("input.npy", buf, "application/octet-stream")}

    # Call the API
    try:
        with st.spinner("Contacting API and running prediction..."):
            r = requests.post(API_URL, files=files, timeout=60)
            r.raise_for_status()
            resp = r.json()
        y_pred = np.array(resp.get("prediction", []))
        input_shape_from_api = resp.get("input_shape", None)
        if input_shape_from_api:
            st.caption(f"API reported input_shape: {input_shape_from_api}")

        if y_pred.size == 0:
            st.error("API returned no predictions.")
            st.stop()

        st.success("Prediction complete via API!")
        st.subheader("Predictions (y_pred)")
        st.write(y_pred[:10], "...")
    except requests.exceptions.RequestException as re:
        st.error(f"API request failed: {re}")
        st.info("Check that your FastAPI service is running and API_URL is correct.")
    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# 2) (Optional) Upload y_test for evaluation
# -------------------------------
if y_pred is not None:
    y_file = st.file_uploader("Optional: Upload y_test to evaluate (.csv, .npy, .npz)", type=["csv", "npy", "npz"], key="y_file")

    def load_y_from_file(f):
        ext = os.path.splitext(f.name)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(f)
            st.write("y_test CSV preview:", df.head())
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric numeric column found for y_test in CSV.")
                st.stop()
            target_col = st.selectbox("Select y_test column:", numeric_cols)
            y = df[target_col].to_numpy().reshape(-1)
            return y
        elif ext == ".npy":
            y = np.load(f, allow_pickle=False)
            return y.reshape(-1)
        elif ext == ".npz":
            npz = np.load(f, allow_pickle=False)
            keys = list(npz.keys())
            st.write("Keys in npz:", keys)
            key = keys[0] if len(keys) == 1 else st.selectbox("Select array key for y_test:", keys)
            y = npz[key].reshape(-1)
            return y
        else:
            st.error("Unsupported file type for y_test")
            st.stop()

    if y_file is not None:
        y_test = load_y_from_file(y_file)
        if len(y_test) != len(y_pred):
            st.warning(f"Length mismatch: y_test={len(y_test)} vs y_pred={len(y_pred)}. Skipping comparison.")
        else:
            # Metrics
            mae = float(np.mean(np.abs(y_test - y_pred)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred)**2)))
            # r2
            ss_res = float(np.sum((y_test - y_pred)**2))
            ss_tot = float(np.sum((y_test - np.mean(y_test))**2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            st.subheader("Evaluation")
            st.write({"MAE": mae, "RMSE": rmse, "R2": r2})

            # Plot actual vs predicted
            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.plot(y_test, label="Actual (y_test)")
            ax.plot(y_pred, label="Predicted (y_pred)")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)
