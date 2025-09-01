import streamlit as st

st.set_page_config(
    page_title="Electricity Price Prediction",
    page_icon="⚡",
    layout="wide",
)

#st.title("Problem")

# Two columns layout
left, right = st.columns(2)

with left:
    st.subheader("⚠️ Problem Description")
    st.markdown(
        """
        Electricity markets are highly volatile due to:
        - Fluctuations in supply and demand
        - Weather conditions and renewable energy integration
        - Policy, regulations, and market behaviour

        This volatility makes **predicting electricity prices** a challenging task.
        Accurate forecasts are crucial for:
        - Energy companies optimising operations
        - Traders managing risk
        - Consumers benefiting from better pricing
        """
    )

with right:
    st.subheader("💡 Solution with AI")
    st.markdown(
        """
        To tackle the problem, we used **data-driven predictive models**:

        - **Machine Learning**:
          - Linear Regression
          - Random Forests


        - **Deep Learning** methods for complex temporal patterns:
          - Dense Neural Networks (RNNs)
          - Recurrent Neural Networks (RNNs)
          - Time-series forecasting with Long Short-Term Memory (LSTM) networks

        We trained these models with **historical electricity price data** along with
        external features (weather, demand, generation mix).

        The result → more accurate and robust forecasts, enabling smarter
        decision-making in energy markets.
        """
    )
