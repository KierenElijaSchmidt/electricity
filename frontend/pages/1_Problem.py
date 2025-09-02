import streamlit as st

st.set_page_config(
    page_title="Electricity Price Prediction",
    page_icon="⚡",
    layout="wide",
)

<<<<<<< HEAD:frontend/pages/1_Problem.py
st.title("⚡ Electricity Price Prediction")

# Two columns layout
left, right = st.columns(2)

with left:
    st.subheader("📉 Problem Description")
=======
# Create 3 columns: empty - content - empty
_, center, _ = st.columns([1, 2, 1])

with center:
    st.subheader("⚠️ Problem Description")
>>>>>>> 38a27cb (added grpahs to problem page):frontend/pages/2_Problem.py
    st.markdown(
        """
        Electricity markets are highly volatile due to:
        - Fluctuations in supply and demand
        - Weather conditions and renewable energy integration
        - Policy, regulations, and market behavior

        This volatility makes **predicting electricity prices** a challenging task.

        Accurate forecasts are crucial for:
        - Energy companies optimizing operations
        - Traders managing risk
        - Consumers benefiting from better pricing
        """
    )

<<<<<<< HEAD:frontend/pages/1_Problem.py
with right:
    st.subheader("🤖 Solution with Machine Learning / Deep Learning")
    st.markdown(
        """
        To tackle the problem, we can apply **data-driven predictive models**:

        - **Machine Learning** methods such as:
          - Linear Regression
          - Random Forests
          - Gradient Boosting

        - **Deep Learning** methods for complex temporal patterns:
          - Recurrent Neural Networks (RNNs)
          - Long Short-Term Memory (LSTM) networks
          - Transformers for time-series forecasting

        These models learn from **historical electricity price data** along with
        external features (weather, demand, generation mix).

        The result → more **accurate and robust forecasts**, enabling smarter
        decision-making in energy markets.
        """
    )
=======
st.image("assets/barcharts/rrp_vs_date.png", use_container_width=True)

st.image("assets/barcharts/rrp_weather_2018_2020.png", use_container_width=True)

st.image("assets/barcharts/rrp_hazelwood.png", use_container_width=True)
>>>>>>> 38a27cb (added grpahs to problem page):frontend/pages/2_Problem.py
