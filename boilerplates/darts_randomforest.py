import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from darts import TimeSeries
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import joblib
class RandomForestModel:
    """Modello Random Forest."""
    def __init__(self, **params):
        # I parametri del modello ora possono essere passati durante l'inizializzazione
        self.model = RandomForestRegressor(random_state=42, **params)
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
    def evaluate(self, y_test: np.ndarray, predictions: np.ndarray) -> dict:
        return {"mae": mean_absolute_error(y_test, predictions),
                "mse": mean_squared_error(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions)}

def run_randomforest(
    y_train: TimeSeries, y_test: TimeSeries,
    X_train: TimeSeries, X_test: TimeSeries,
    tune_hyperparameters: bool = False # Nuovo parametro per attivare il tuning
) -> dict:
    """Esegue Random Forest, con tuning opzionale."""
    print("Running Random Forest...")
    X_train_np, X_test_np = X_train.values(), X_test.values()
    y_train_np = y_train.values().flatten()
    y_test_np = y_test.values().flatten()

    if tune_hyperparameters:
        print("  Inizio Hyperparameter Tuning per Random Forest...")
        param_dist = {
            'n_estimators': sp_randint(50, 200), # Numero di alberi
            'max_depth': sp_randint(5, 20),      # Profondità massima degli alberi
            'min_samples_split': sp_randint(2, 10), # Numero minimo di campioni per dividere un nodo
            'min_samples_leaf': sp_randint(1, 5)   # Numero minimo di campioni per foglia
        }

        # Inizializza un modello base per RandomizedSearchCV
        base_model = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=20, # Numero di combinazioni di parametri da provare
            cv=3,       # Cross-validation a 3 fold
            verbose=0,  # Ridotto a 0 per output sintetico
            random_state=42,
            n_jobs=-1   # Usa tutti i core disponibili
        )

        random_search.fit(X_train_np, y_train_np)
        best_params = random_search.best_params_
        print(f"  Migliori iperparametri per Random Forest: {best_params}")
        model = RandomForestModel(**best_params) # Inizializza il modello con i migliori parametri
    else:
        model = RandomForestModel() # Usa i parametri di default

    model.train(X_train_np, y_train_np)
    predictions_np = model.predict(X_test_np)
    results = model.evaluate(y_test_np, predictions_np)
    print("Random Forest Results:", results)
    joblib.dump(model, "random_forest.pkl")
    return results
