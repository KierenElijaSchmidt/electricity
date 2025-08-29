import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from darts import TimeSeries
import joblib

class XGBoostModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state, objective='reg:squarederror')
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
    def evaluate(self, y_test: np.ndarray, predictions: np.ndarray) -> dict:
        return {"mae": mean_absolute_error(y_test, predictions),
                "mse": mean_squared_error(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions)}

def run_xgboost(
    y_train: TimeSeries, y_test: TimeSeries,
    X_train: TimeSeries, X_test: TimeSeries
) -> dict:
    print("Running XGBoost...")
    X_train_np, X_test_np = X_train.values(), X_test.values()
    y_train_np = y_train.values().flatten()
    y_test_np = y_test.values().flatten()

    model = XGBoostModel()
    model.train(X_train_np, y_train_np)
    predictions_np = model.predict(X_test_np)
    results = model.evaluate(y_test_np, predictions_np)
    # salvataggio modello
    joblib.dump(model, "xgboost_model.pkl")
    print("XGBoost Results:", results)
    return results
