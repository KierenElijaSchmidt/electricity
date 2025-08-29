import pandas as pd
import joblib
from darts import TimeSeries
import numpy as np

# --- Preprocessing identico a training ---
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Addressed FutureWarning: Using .ffill() directly is the modern approach
    df = df.ffill()

    # Based on your traceback, 'demand' is the actual name of your value column
    target_col = "demand" # Define the target column name here for consistency

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame during preprocessing.")

    # Create lag features (keep only the most important ones)
    df["lag1"] = df[target_col].shift(1)
    df["rolling_mean"] = df[target_col].rolling(window=3).mean()

    # --- TIME-BASED FEATURES (select only the most important ones) ---
    # Assuming you have a date column - adjust the column name if different
    if 'date' in df.columns:
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # Extract time-based features - only the most critical ones
        df['hour_of_day'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

        # Cyclical encoding for hour (most important for electricity demand)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # Additional rolling features
        df['rolling_mean_7'] = df[target_col].rolling(window=7).mean()  # Weekly average
        df['rolling_std'] = df[target_col].rolling(window=3).std()

        # Temperature interaction
        if 'max_temperature' in df.columns:
            df['temp_demand_ratio'] = df['max_temperature'] / (df[target_col] + 1)
        else:
            df['temp_demand_ratio'] = 1

        # Weather range
        if 'min_temperature' in df.columns and 'max_temperature' in df.columns:
            df['temp_range'] = df['max_temperature'] - df['min_temperature']
        else:
            df['temp_range'] = 0

        # Price volatility
        if 'RRP_positive' in df.columns and 'RRP_negative' in df.columns:
            df['price_volatility'] = abs(df['RRP_positive'] - df['RRP_negative'])
        else:
            df['price_volatility'] = 0

        # Demand change
        df['demand_change'] = df[target_col].diff()

        # Lag 2 for additional context
        df["lag2"] = df[target_col].shift(2)

        # One more feature to reach exactly 26
        df['rolling_max_3'] = df[target_col].rolling(window=3).max()

    else:
        print("Warning: No 'date' column found. Creating essential dummy time features.")
        # Create only essential dummy features
        df['hour_of_day'] = 12
        df['day_of_week'] = 2
        df['month'] = 6
        df['is_weekend'] = 0
        df['hour_sin'] = 0
        df['hour_cos'] = 1
        df['rolling_mean_7'] = df[target_col].rolling(window=7).mean()
        df['rolling_std'] = df[target_col].rolling(window=3).std()
        df['temp_demand_ratio'] = 1
        df['temp_range'] = 0
        df['price_volatility'] = 0
        df['demand_change'] = df[target_col].diff()
        df["lag2"] = df[target_col].shift(2)
        df['rolling_max_3'] = df[target_col].rolling(window=3).max()

    # Fill any remaining NaN values that might have been created
    df = df.dropna()

    print(f"After preprocessing, DataFrame has {len(df.columns)} columns")
    print(f"Column names: {list(df.columns)}")

    return df

def verify_input(df, expected_features: int):
    # 'RRP' is mentioned in your error but 'demand' seems to be the actual target
    target_col = "demand"  # Changed back to demand based on your error message

    if target_col not in df.columns:
        raise ValueError(f"The DataFrame must contain a '{target_col}' column.")

    # Drop the target column to get only the features
    # Also, explicitly drop 'date' if it's a timestamp column
    columns_to_drop = [target_col]
    if 'date' in df.columns:
        print("Info: 'date' column found. Dropping it from features for model input.")
        columns_to_drop.append('date')

    feature_df = df.drop(columns=columns_to_drop, errors='ignore')

    # Convert all remaining columns to numeric, coercing errors to NaN, then drop NaNs
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

    # Drop columns that became all NaN due to non-numeric conversion
    feature_df = feature_df.dropna(axis=1)

    # If we still don't have enough features, create additional dummy features
    if len(feature_df.columns) < expected_features:
        print(f"Still missing {expected_features - len(feature_df.columns)} features. Adding dummy features.")
        for i in range(expected_features - len(feature_df.columns)):
            feature_df[f'dummy_feature_{i}'] = 0

    X_new = feature_df.values

    print(f"Features actually being passed to the model: {list(feature_df.columns)}")
    print(f"Number of features actually being passed: {X_new.shape[1]}")

    if X_new.shape[1] != expected_features:
        raise ValueError(
            f"Wrong number of features: expected {expected_features}, got {X_new.shape[1]}. "
            f"This likely means your input CSV is missing columns that were used as features during training, "
            f"or your `preprocess` function isn't generating all required features. "
            f"Current features derived and passed to model: {list(feature_df.columns)}. "
            f"Please ensure your input data and preprocessing steps match your training pipeline precisely."
        )
    return X_new

def predict(df):
    # Load trained models
    # Ensure these paths are correct relative to where you run the script
    lr_model = joblib.load("/Users/Alessio/code/KierenElijaSchmidt/electricity/results/autoregressor.pkl")
    rf_model = joblib.load("/Users/Alessio/code/KierenElijaSchmidt/electricity/results/random_forest.pkl")
    xgb_model = joblib.load("/Users/Alessio/code/KierenElijaSchmidt/electricity/results/xgboost_model.pkl")

    # --- Apply preprocessing ---
    df = preprocess(df)

    # Check expected feature count (from training)
    expected_features = lr_model.model.n_features_in_
    print(f"Model expects {expected_features} features")

    X_new = verify_input(df, expected_features)

    # Forecast on the last 10 samples
    if len(X_new) < 10:
        print("Warning: Not enough data points for forecasting the last 10 samples. Forecasting on all available samples.")
        lr_forecast = lr_model.predict(X_new)
        xgb_forecast = xgb_model.predict(X_new)
        rf_forecast = rf_model.predict(X_new)
    else:
        lr_forecast = lr_model.predict(X_new[-10:])
        xgb_forecast = xgb_model.predict(X_new[-10:])
        rf_forecast = rf_model.predict(X_new[-10:])

    print("Linear Regression Forecast:", lr_forecast)
    print("XGBoost Forecast:", xgb_forecast)
    print("Random Forest Forecast:", rf_forecast)

def main():
    # Load raw data
    # Ensure this path is correct
    df = pd.read_csv("/Users/Alessio/Desktop/complete_dataset.csv")

    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Send it through preprocessing + predictions
    predict(df)

if __name__ == "__main__":
    main()
