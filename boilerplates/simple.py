from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import joblib
import numpy as np
import io
from darts_preprocessing import loader, preprocessor
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {"status": "API is running"}

@app.post("/predict/csv")
async def predict_csv():
    try:
        print("=" * 50)
        print("Starting prediction process...")
        logger.info("Starting CSV prediction process")

        # Read CSV
        file_path = "complete_dataset.csv"
        print(f"Loading data from: {file_path}")
        loaded_data = loader(file_path)
        print(f"Data loaded successfully. Shape: {loaded_data.shape if hasattr(loaded_data, 'shape') else 'Unknown'}")
        logger.info(f"Data loaded from {file_path}")

        # Preprocessing dei dati
        print("Starting preprocessing...")
        _, series, covariates, _ = preprocessor(loaded_data)
        print(f"Preprocessing completed.")
        print(f"Series shape: {series.shape if hasattr(series, 'shape') else type(series)}")
        print(f"Covariates shape: {covariates.shape if hasattr(covariates, 'shape') else type(covariates)}")
        logger.info("Data preprocessing completed")

        # Split dei dati in training e test
        print("Splitting data...")
        train_size = int(len(series) * 0.8)
        y_train, y_test = series[:train_size], series[train_size:]
        X_train, X_test = covariates[:train_size], covariates[train_size:len(series)]

        print(f"Train size: {train_size}")
        print(f"y_train shape: {y_train.shape if hasattr(y_train, 'shape') else len(y_train)}")
        print(f"y_test shape: {y_test.shape if hasattr(y_test, 'shape') else len(y_test)}")
        print(f"X_train shape: {X_train.shape if hasattr(X_train, 'shape') else len(X_train)}")
        print(f"X_test shape: {X_test.shape if hasattr(X_test, 'shape') else len(X_test)}")

        # Deep inspection of X_test structure
        print("\n--- DETAILED X_test INSPECTION ---")
        print(f"X_test type: {type(X_test)}")
        print(f"X_test length: {len(X_test) if hasattr(X_test, '__len__') else 'No length'}")

        if hasattr(X_test, 'shape'):
            print(f"X_test shape: {X_test.shape}")

        if hasattr(X_test, 'ndim'):
            print(f"X_test dimensions: {X_test.ndim}")

        # Check first few elements
        print("First 3 elements of X_test:")
        for i, element in enumerate(X_test[:3]):
            print(f"  Element {i}: type={type(element)}, shape={element.shape if hasattr(element, 'shape') else 'no shape'}")
            if hasattr(element, 'shape') and len(element.shape) > 0:
                print(f"    Value sample: {element[:5] if len(element) > 5 else element}")

        # Check for nested structures
        if len(X_test) > 0:
            first_elem = X_test[0]
            print(f"First element detailed inspection:")
            print(f"  Type: {type(first_elem)}")
            if hasattr(first_elem, '__len__') and not isinstance(first_elem, (str, bytes)):
                print(f"  Length: {len(first_elem)}")
                if hasattr(first_elem, '__iter__') and len(first_elem) > 0:
                    sub_elem = first_elem[0]
                    print(f"  Sub-element type: {type(sub_elem)}")
                    if hasattr(sub_elem, '__len__') and not isinstance(sub_elem, (str, bytes)):
                        print(f"  Sub-element length: {len(sub_elem)}")

        print("--- END X_test INSPECTION ---\n")
        logger.info(f"Data split completed - train_size: {train_size}")

        # Try to convert X_test to proper format
        print("Attempting to fix X_test format...")
        try:
            X_test_array = np.array(X_test)
            print(f"X_test converted to numpy array. Shape: {X_test_array.shape}")
        except Exception as e:
            print(f"Failed to convert X_test to numpy array: {e}")
            # Try flattening or reshaping
            try:
                if hasattr(X_test, 'values'):
                    X_test_array = X_test.values
                    print(f"Used .values attribute. Shape: {X_test_array.shape}")
                else:
                    X_test_array = np.concatenate([np.array(x).flatten() for x in X_test]).reshape(len(X_test), -1)
                    print(f"Manually reshaped X_test. Shape: {X_test_array.shape}")
            except Exception as e2:
                print(f"Also failed to manually reshape: {e2}")
                X_test_array = X_test

        # Load models
        print("Loading models...")
        lr_model_path = "/Users/Alessio/code/KierenElijaSchmidt/electricity/results/autoregressor.pkl"
        rf_model_path = "/Users/Alessio/code/KierenElijaSchmidt/electricity/results/random_forest.pkl"
        xgb_model_path = "/Users/Alessio/code/KierenElijaSchmidt/electricity/results/xgboost_model.pkl"

        lr_model = joblib.load(lr_model_path)
        print(f"Linear Regression model loaded: {type(lr_model)}")

        rf_model = joblib.load(rf_model_path)
        print(f"Random Forest model loaded: {type(rf_model)}")

        xgb_model = joblib.load(xgb_model_path)
        print(f"XGBoost model loaded: {type(xgb_model)}")

        logger.info("All models loaded successfully")

        # Get predictions
        print("Making predictions...")

        print("LR Model prediction...")
        try:
            lr_pred = lr_model.predict(X_test_array)
            print(f"LR prediction completed. Shape: {lr_pred.shape}, Type: {type(lr_pred)}")
            print(f"LR prediction sample (first 5): {lr_pred[:5] if len(lr_pred) > 5 else lr_pred}")
        except Exception as e:
            print(f"LR prediction failed: {e}")
            # Try with original X_test
            try:
                print("Trying with original X_test...")
                lr_pred = lr_model.predict(X_test)
                print(f"LR prediction with original X_test completed. Shape: {lr_pred.shape}")
            except Exception as e2:
                print(f"LR prediction with original X_test also failed: {e2}")
                raise e2

        print("RF Model prediction...")
        try:
            rf_pred = rf_model.predict(X_test_array)
            print(f"RF prediction completed. Shape: {rf_pred.shape}, Type: {type(rf_pred)}")
            print(f"RF prediction sample (first 5): {rf_pred[:5] if len(rf_pred) > 5 else rf_pred}")
        except Exception as e:
            print(f"RF prediction failed: {e}")
            try:
                print("Trying with original X_test...")
                rf_pred = rf_model.predict(X_test)
                print(f"RF prediction with original X_test completed. Shape: {rf_pred.shape}")
            except Exception as e2:
                print(f"RF prediction with original X_test also failed: {e2}")
                raise e2

        print("XGB Model prediction...")
        try:
            xgb_pred = xgb_model.predict(X_test_array)
            print(f"XGB prediction completed. Shape: {xgb_pred.shape}, Type: {type(xgb_pred)}")
            print(f"XGB prediction sample (first 5): {xgb_pred[:5] if len(xgb_pred) > 5 else xgb_pred}")
        except Exception as e:
            print(f"XGB prediction failed: {e}")
            try:
                print("Trying with original X_test...")
                xgb_pred = xgb_model.predict(X_test)
                print(f"XGB prediction with original X_test completed. Shape: {xgb_pred.shape}")
            except Exception as e2:
                print(f"XGB prediction with original X_test also failed: {e2}")
                raise e2

        logger.info("All predictions completed")

        # Convert predictions to lists (with flattening)
        print("Converting predictions to lists...")

        try:
            lr_forecast = lr_pred.flatten().tolist()
            print(f"LR forecast converted. Length: {len(lr_forecast)}")
        except Exception as e:
            print(f"Error converting LR prediction: {e}")
            lr_forecast = []

        try:
            rf_forecast = rf_pred.flatten().tolist()
            print(f"RF forecast converted. Length: {len(rf_forecast)}")
        except Exception as e:
            print(f"Error converting RF prediction: {e}")
            rf_forecast = []

        try:
            xgb_forecast = xgb_pred.flatten().tolist()
            print(f"XGB forecast converted. Length: {len(xgb_forecast)}")
        except Exception as e:
            print(f"Error converting XGB prediction: {e}")
            xgb_forecast = []

        print("Predictions converted successfully!")
        print("=" * 50)
        logger.info("Prediction process completed successfully")

        return {
            "lr_forecast": lr_forecast,
            "rf_forecast": rf_forecast,
            "xgb_forecast": xgb_forecast,
            "metadata": {
                "total_predictions": len(lr_forecast),
                "lr_shape": str(lr_pred.shape),
                "rf_shape": str(rf_pred.shape),
                "xgb_shape": str(xgb_pred.shape)
            }
        }

    except Exception as e:
        print("=" * 50)
        print(f"ERROR OCCURRED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        print("=" * 50)
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server with debug logging...")
    print("Access the API at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
