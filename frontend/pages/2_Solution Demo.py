import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# -------------------------------
# Load trained model (same as API)
# -------------------------------
@st.cache_resource
def load_model():
    # Go up two levels to repo root
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(REPO_ROOT, "results", "rnn_model.keras"))
    return tf.keras.models.load_model(MODEL_PATH)



# -------------------------------
# Streamlit Page
# -------------------------------
st.title("📊 Forecast with Keras Model")

uploaded_file = st.file_uploader("Upload dataset (.csv, .npy, .npz)", type=["csv", "npy", "npz"])

if uploaded_file is not None:
    try:
        # Detect file type
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(uploaded_file)
            st.write("Data preview:", df.head())

            # Let user pick the target column
            target_col = st.selectbox(
                "Select target column (y):",
                df.select_dtypes(include=[np.number]).columns
            )
            y_raw = df[target_col].values

        elif ext == ".npy":
            y_raw = np.load(uploaded_file, allow_pickle=False)
            st.write("Loaded NumPy array with shape:", y_raw.shape)

        elif ext == ".npz":
            npzfile = np.load(uploaded_file, allow_pickle=False)
            st.write("Keys in npz file:", list(npzfile.keys()))
            y_raw = npzfile[list(npzfile.keys())[0]]
            st.write("Loaded NumPy array with shape:", y_raw.shape)

        else:
            st.error("Unsupported file type")
            st.stop()

        # Ensure array shape is correct
        X = np.asarray(y_raw)
        if X.ndim == 1:
            # If 1D, reshape into (n, 1) for model
            X = X.reshape(-1, 1)

        # Load model
        model = load_model()

        # Predict
        preds = model.predict(X).flatten()
        st.success("Prediction complete!")

        # Show prediction values
        st.subheader("Predictions")
        st.write(preds[:10], "...")  # show first few

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
            st.warning("Input and prediction lengths differ, cannot plot comparison.")

    except Exception as e:
        st.error(f"Error: {e}")
