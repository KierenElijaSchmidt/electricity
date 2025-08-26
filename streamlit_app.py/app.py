
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error
st.title(":blitzschnell: Electricity Price Prediction")
# Load data
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    # Simple baseline
    df['prediction'] = df['RRP'].shift(1)
    df = df.dropna()
    # Metrics
    mae = mean_absolute_error(df['RRP'], df['prediction'])
    st.metric("MAE", f"{mae:.2f}")
    # Simple chart
    st.line_chart(df.set_index('date')[['RRP', 'prediction']])
    # Latest predictions
    st.write("Latest 10 predictions:")
    st.dataframe(df[['date', 'RRP', 'prediction']].tail(10))
