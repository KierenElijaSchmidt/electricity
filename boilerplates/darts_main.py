import pandas as pd
import numpy as np
from darts import TimeSeries

from darts_preprocessing import loader, preprocessor
from darts_linear_regression import run_linear_regression
from darts_randomforest import run_randomforest
from darts_xgboost import run_xgboost

def main():
    """Esegue la pipeline Darts."""
    print("Starting Darts analysis...")
    file_path = "complete_dataset.csv"
    loaded_data = loader(file_path)

    # Preprocessing dei dati
    _, series, covariates, _ = preprocessor(loaded_data)

    # Split dei dati in training e test
    train_size = int(len(series) * 0.8)
    y_train, y_test = series[:train_size], series[train_size:]
    X_train, X_test = covariates[:train_size], covariates[train_size:len(series)]

    print("\n--- Running Models ---")
    models_results = {
        # Linear Regression non ha hyperparameter tuning in questo contesto
        'Linear Regression': run_linear_regression(y_train, y_test, X_train, X_test),
        # Attiva l'hyperparameter tuning per Random Forest
        'Random Forest': run_randomforest(y_train, y_test, X_train, X_test, tune_hyperparameters=True),
        # Attiva l'hyperparameter tuning per XGBoost
        'XGBoost': run_xgboost(y_train, y_test, X_train, X_test)
    }

    # Estrai i punteggi per una stampa più chiara
    lr_scores = models_results['Linear Regression']
    rf_scores = models_results['Random Forest']
    xgb_scores = models_results['XGBoost']

    print("Linear Regression Scores:", lr_scores)
    print("Random Forest Scores:", rf_scores)
    print("XGBoost Scores:", xgb_scores)

    print("\n--- Comparing Results ---")
    # Stampa i risultati confrontando le metriche
    print(f"Linear Regression - MAE: {lr_scores['mae']:.4f}, RMSE: {lr_scores['rmse']:.4f}, R2: {lr_scores['r2']:.4f}")
    print(f"Random Forest     - MAE: {rf_scores['mae']:.4f}, RMSE: {rf_scores['rmse']:.4f}, R2: {rf_scores['r2']:.4f}")
    print(f"XGBoost           - MAE: {xgb_scores['mae']:.4f}, RMSE: {xgb_scores['rmse']:.4f}, R2: {xgb_scores['r2']:.4f}")

    # Trova il modello migliore
    best_model_mae = min(models_results.keys(), key=lambda k: models_results[k]['mae'])
    best_model_rmse = min(models_results.keys(), key=lambda k: models_results[k]['rmse'])
    best_model_r2 = max(models_results.keys(), key=lambda k: models_results[k]['r2'])

    print(f"\nBest model by MAE: {best_model_mae}")
    print(f"Best model by RMSE: {best_model_rmse}")
    print(f"Best model by R2: {best_model_r2}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
