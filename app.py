import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

st.title("ELECTRICITY - RRP Forecast - Linear Regression Model")

uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Encode categorical features as in the notebook
    df['holiday'] = df['holiday'].map({'Y': 1, 'N': 0})
    df['school_day'] = df['school_day'].map({'Y': 1, 'N': 0})

    # Split features and target
    X = df.drop(columns=['RRP', 'date'])
    y = df['RRP']

    # Preprocessing + model pipeline
    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='median'), RobustScaler()), X.select_dtypes('number').columns),
        (OneHotEncoder(handle_unknown='ignore'), X.select_dtypes('object').columns)
    )

    model = make_pipeline(preprocessor, LinearRegression())
    model.fit(X, y)

    # Forecast for the next day (based on the last row as input)
    last_row = X.iloc[[-1]]
    next_pred = model.predict(last_row)[0]
    next_date = df['date'].iloc[-1] + pd.Timedelta(days=1)

    st.write(f"📅 Last available date: **{df['date'].iloc[-1].date()}**")
    st.write(f"🔹 Last observed value: **{y.iloc[-1]:.2f}**")
    st.write(f"📅 Forecast for the next day ({next_date.date()}):")
    st.success(f"**{next_pred:.2f}**")

    # Plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df['date'], y, label="Observed")
    ax.scatter(next_date, next_pred, color="red", label="Forecast", zorder=5)
    ax.legend()
    st.pyplot(fig)
