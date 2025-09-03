import os
import io
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- 0) Settings -------------------------------------------------------------
# Quieter TF logs (optional)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def _find_default_model() -> Path:
    """Look for a model in common places so it works locally and in containers."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "results" / "rnn_model.keras",     # ./results/rnn_model.keras
        here / "rnn_model.keras",                 # ./rnn_model.keras
        here.parent / "results" / "rnn_model.keras",  # ../results/rnn_model.keras
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # default suggestion even if it doesn't exist yet

DEFAULT_MODEL = _find_default_model()
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL))).expanduser()

# --- 1) App + lifecycle ------------------------------------------------------
model = None  # will hold the loaded Keras model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load once at startup (fast fail if the model is missing)."""
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    yield
    # (optional) cleanup on shutdown
    model = None

app = FastAPI(lifespan=lifespan)

# --- 2) CORS so your Streamlit app can call this API -------------------------
ALLOWED_ORIGINS = [
    "http://localhost:8501",                       # local Streamlit
    "https://<your-streamlit-app>.streamlit.app",  # <-- replace with your deployed UI URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3) Health endpoint ------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None,
    }

# --- 4) Predict endpoint -----------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a .npy or .npz (NumPy) file containing your input array X.
    Returns a flat list of predictions.
    """
    try:
        contents = await file.read()
        arr = np.load(io.BytesIO(contents), allow_pickle=False)

        # Accept .npz (pick first array) or .npy
        if isinstance(arr, np.lib.npyio.NpzFile):
            first_key = list(arr.files)[0]
            X = arr[first_key]
        else:
            X = arr

        X = np.asarray(X)
        preds = model.predict(X, verbose=0)
        preds = np.asarray(preds).reshape(-1).tolist()

        return {"input_shape": list(X.shape), "prediction": preds}

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# --- 5) Local dev entrypoint -------------------------------------------------
if __name__ == "__main__":
    # For local testing: python api.py
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
