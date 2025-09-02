import os
import io
import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------
# Config: where is your API?
# --------------------------------
# Set API_URL in your environment for prod, e.g.:
#   export API_URL="https://your-domain/predict"
API_URL = os.getenv("API_URL", "http://localhost:8080/predict")

st.title("📡 Forecast via FastAPI")
st.caption(f"Using API: {API_URL}")

uploaded_file = st.file_uploader("Upload dataset (.csv, .npy, .npz)", type=["csv", "npy", "npz"])

if uploaded_file is not None:
    try:
        # -------------------------------
        # Detect and parse the upload
        # -------------------------------
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(uploaded_file)
            st.write("Data preview:", df.head())

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                st.error("No numeric columns found in CSV.")
                st.stop()

            target_col = st.selectbox("Select target column (y):", numeric_cols)
            y_raw = df[target_col].values

        elif ext == ".npy":
            y_raw = np.load(uploaded_file, allow_pickle=False)
            st.write("Loaded NumPy array with shape:", y_raw.shape)

        elif ext == ".npz":
            npzfile = np.load(uploaded_file, allow_pickle=False)
            keys = list(npzfile.keys())
            st.write("Keys in npz file:", keys)
            key = keys[0] if len(keys) == 1 else st.selectbox("Select array key:", keys)
            y_raw = npzfile[key]
            st.write("Loaded NumPy array with shape:", y_raw.shape)

        else:
            st.error("Unsupported file type")
            st.stop()

        # -------------------------------
        # Shape to something reasonable
        # -------------------------------
        X = np.asarray(y_raw)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # keep your previous behavior

        st.write("Input array shape being sent to API:", X.shape)

        # -------------------------------
        # Serialize as .npy in-memory
        # -------------------------------
        buf = io.BytesIO()
        np.save(buf, X)          # write .npy bytes
        buf.seek(0)              # rewind before read

        files = {"file": ("input.npy", buf, "application/octet-stream")}

        # -------------------------------
        # Call the API
        # -------------------------------
        with st.spinner("Contacting API and running prediction..."):
            r = requests.post(API_URL, files=files, timeout=60)
            # If FastAPI returned an error code, raise here
            r.raise_for_status()
            resp = r.json()

        # -------------------------------
        # Consume response
        # -------------------------------
        preds = np.array(resp.get("prediction", []))
        input_shape_from_api = resp.get("input_shape", None)

        if input_shape_from_api:
            st.caption(f"API reported input_shape: {input_shape_from_api}")

        if preds.size == 0:
            st.error("API returned no predictions.")
            st.stop()

        st.success("Prediction complete via API!")

        # Show prediction values
        st.subheader("Predictions")
        st.write(preds[:10], "...")  # first few

        # --- Plot actual vs predicted ---
        if len(preds) == len(y_raw):
            st.subheader("Comparison: Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.plot(y_raw, label="Actual")
            ax.plot(preds, label="Predicted")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning(
                f"Input and prediction lengths differ (actual={len(y_raw)}, predicted={len(preds)}). "
                "Skipping comparison plot."
            )

    except requests.exceptions.RequestException as re:
        st.error(f"API request failed: {re}")
        st.info("Check that your FastAPI service is running and API_URL is correct.")
    except Exception as e:
        st.error(f"Error: {e}")
