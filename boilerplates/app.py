import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load your trained Keras model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "/Users/Alessio/code/KierenElijaSchmidt/electricity/results/rnn_model.keras"
    )

st.title("Keras Model Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload a .npy file", type=["npy"])

if uploaded_file is not None:
    try:
        # Load numpy data
        data = np.load(uploaded_file)
        st.write("Dataset shape:", data.shape)

        # Load model
        model = load_model()

        # Make prediction
        preds = model.predict(data)

        st.subheader("Predictions (array):")
        st.write(preds)

        # --- Plot results ---
        st.subheader("Predictions Plot")

        fig, ax = plt.subplots()
        if preds.ndim == 1 or preds.shape[1] == 1:  # regression or single output
            ax.plot(preds, label="Predictions")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Predicted Value")
        else:  # classification probs or multi-output
            for i in range(preds.shape[1]):
                ax.plot(preds[:, i], label=f"Class {i}")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Probability")

        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
