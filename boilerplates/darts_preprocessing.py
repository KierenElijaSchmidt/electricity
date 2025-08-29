import pandas as pd
import numpy as np
from darts import TimeSeries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def loader(file_path: str) -> pd.DataFrame:
    file_path = "/Users/Alessio/Desktop/complete_dataset.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        data = {
            'date': pd.to_datetime(pd.date_range(start='2015-01-01', periods=100, freq='D')),
            'demand': np.random.rand(100) * 100000,
            'RRP': np.random.rand(100) * 50,
            'min_temperature': np.random.rand(100) * 15 + 10,
            'max_temperature': np.random.rand(100) * 15 + 25,
            'school_day': np.random.choice(['N', 'Y'], 100),
            'holiday': np.random.choice(['N', 'Y'], 100)
        }
        return pd.DataFrame(data)

def preprocessor(df: pd.DataFrame):
    """Pre-elabora per Darts TimeSeries."""
    df_processed = df.copy()
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.set_index('date')
    else:
        df_processed.index = pd.to_datetime(pd.date_range(start='2015-01-01', periods=len(df_processed), freq='D'))

    # Gestione valori mancanti (iniziale)
    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True)).fillna(df_processed.mode().iloc[0])

    # --- Ingegneria delle Feature per Serie Temporali ---
    # Feature basate sul tempo (stagionalità e tendenza)
    df_processed['day_of_week'] = df_processed.index.dayofweek
    df_processed['month'] = df_processed.index.month
    df_processed['year'] = df_processed.index.year
    df_processed['day_of_year'] = df_processed.index.dayofyear # Per la stagionalità annuale

    # Lagged features per RRP (target) e Demand (covariata importante)
    df_processed['RRP_lag_1'] = df_processed['RRP'].shift(1)
    df_processed['RRP_lag_2'] = df_processed['RRP'].shift(2) # Nuova feature di lag
    df_processed['RRP_lag_3'] = df_processed['RRP'].shift(3) # Nuova feature di lag
    df_processed['RRP_lag_7'] = df_processed['RRP'].shift(7) # Valore della settimana precedente

    df_processed['demand_lag_1'] = df_processed['demand'].shift(1)
    df_processed['demand_lag_7'] = df_processed['demand'].shift(7)

    # Rolling Mean per RRP (Nuova feature: media mobile per catturare trend locali)
    df_processed['RRP_rolling_mean_3'] = df_processed['RRP'].rolling(window=3).mean().shift(1)
    df_processed['RRP_rolling_mean_7'] = df_processed['RRP'].rolling(window=7).mean().shift(1)


    # Riempi i NaN creati dallo shift e dal rolling (per i primi giorni) con la media delle colonne
    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))

    target_col = 'RRP'
    # Tutte le colonne tranne il target_col sono ora covariate
    all_covariate_cols = [col for col in df_processed.columns if col != target_col]

    numerical_cols = df_processed[all_covariate_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_processed[all_covariate_cols].select_dtypes(include=['object', 'bool']).columns.tolist()

    preprocessor_pipeline = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )

    transformed_covariates_array = preprocessor_pipeline.fit_transform(df_processed[all_covariate_cols])
    feature_cols = numerical_cols + preprocessor_pipeline.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()

    covariates_df = pd.DataFrame(transformed_covariates_array, index=df_processed.index, columns=feature_cols)
    series = TimeSeries.from_dataframe(df_processed, value_cols=[target_col])
    covariates = TimeSeries.from_dataframe(covariates_df)

    return df_processed, series, covariates, feature_cols

def extend_covariates(covariates: TimeSeries, horizon: int) -> TimeSeries:
    """Estende le covariate per previsione."""
    last_val = covariates.last_values()
    future_index = pd.date_range(start=covariates.end_time() + pd.Timedelta(days=1), periods=horizon, freq=covariates.freq_str())
    future_df = pd.DataFrame(np.tile(last_val, (horizon, 1)), index=future_index, columns=covariates.components)
    return covariates.append_values(TimeSeries.from_dataframe(future_df))
