# Configuration settings for the darts project
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
MODEL_CONFIGS = {
    'linear_regression': {
        'fit_intercept': True,
        'normalize': False
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

# Preprocessing settings
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'target_column': 'score'  # Adjust based on your data
}

# Training settings
TRAINING_CONFIG = {
    'validation_size': 0.1,
    'random_state': 42
}
