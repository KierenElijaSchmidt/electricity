import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------
# Helper function to create sequences
# -------------------------------
def create_sequences(series, seq_length=60):
    """Turn 1D series into 3D sequences for RNN input."""
    sequences = []
    for i in range(len(series) - seq_length):
        seq = series[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "../results/rnn_model.keras"
    )

# -------------------------------
# Streamlit App
# -------------------------------
st.title("Next Day Forecast with Keras Model")

uploaded_file = st.file_uploader("Upload dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:", df.head())

        # Let user pick the target column
        target_col = st.selectbox("Select target column (y):", df.select_dtypes(include=[np.number]).columns)

        # Extract the series
        y_raw = df[target_col].values

        # Build sequences
        SEQ_LEN = 60
        X = create_sequences(y_raw, SEQ_LEN).reshape(-1, SEQ_LEN, 1)
        y = y_raw[SEQ_LEN:]

        st.write("X shape:", X.shape)
        st.write("y shape:", y.shape)

        # Load model
        model = load_model()

        # Predict the next day using the last sequence
        last_seq = y_raw[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        next_pred = model.predict(last_seq)[0][0]

        # Show prediction
        st.subheader("Next Day Prediction")
        st.write(f"Predicted value: {next_pred:.2f}")

        # --- Plot ---
        st.subheader("Last 100 Days + Next Day Prediction")

        last_100 = y[-100:]
        extended = np.append(last_100, next_pred)

        fig, ax = plt.subplots()
        ax.plot(range(len(last_100)), last_100, label="Target (y)")
        ax.plot(len(last_100), next_pred, "ro", label="Next Day Prediction")
        ax.plot(range(len(extended)), extended, "--", color="gray", alpha=0.7)

        ax.set_xlabel("Days (last 100)")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
