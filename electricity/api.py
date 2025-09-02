import os, io
from pathlib import Path
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

BASE_DIR = Path(__file__).resolve().parent              # .../electricity/electricity
REPO_ROOT = BASE_DIR.parent                             # .../electricity
DEFAULT_MODEL = REPO_ROOT / "results" / "rnn_model.keras"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL))).expanduser()
print(f"[api] MODEL_PATH={MODEL_PATH} exists={MODEL_PATH.exists()}")

# Optional: lazy-load so the server starts even if the path is wrong
_model = None
def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "welcome to our app"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        X = np.load(io.BytesIO(contents), allow_pickle=False)
        if isinstance(X, np.lib.npyio.NpzFile):
            X = X[list(X.files)[0]]
        X = np.asarray(X)
        y = get_model().predict(X).flatten().tolist()
        return {"input_shape": list(X.shape), "prediction": y}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("electricity.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
