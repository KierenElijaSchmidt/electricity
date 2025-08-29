from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import sys
import os
from flask import Flask, request, jsonify, render_template
from darts_xgboost import XGBoostModel
from darts_preprocessing import DataPreprocessor
import joblib
app = Flask(__name__)

# Load model and preprocessor
MODEL_PATH = "artifacts/xgboost_1.0/model.joblib"
PREPROCESSOR_PATH = "artifacts/xgboost_1.0/preprocessor.joblib"

try:
    model = XGBoostModel()
    model.load_model(MODEL_PATH)
    preprocessor = DataPreprocessor.load_preprocessor(PREPROCESSOR_PATH)
    print("Model and preprocessor loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get data from request
        data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Preprocess data
        processed_data = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(processed_data)

        return jsonify({
            'prediction': float(prediction[0]),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get data from request
        data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame(data)

        # Preprocess data
        processed_data = preprocessor.transform(input_df)

        # Make predictions
        predictions = model.predict(processed_data)

        return jsonify({
            'predictions': predictions.tolist(),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
