import streamlit as st

st.set_page_config(layout="wide")

st.subheader("⚠️ Problem Description")

# --- First bullet + chart ---
st.markdown("""
- **Fluctuations in supply and demand**
""")
st.image("assets/barcharts/rrp_vs_date.png", use_container_width=True, caption="Electricity Price (RRP) over Time")

# --- Second bullet + chart ---
st.markdown("""
- **Weather conditions and renewable energy integration**
""")
st.image("assets/barcharts/rrp_weather_2018_2020.png", use_container_width=True, caption="Electricity Price vs Weather Conditions")

# --- Third bullet + chart ---
st.markdown("""
- **Policy, regulations, and market behaviour**
""")
st.image("assets/barcharts/rrp_hazelwood.png", use_container_width=True, caption="Impact of Hazelwood Closure on RRP")

# --- Why forecasts are crucial ---
st.markdown("""
This volatility makes **predicting electricity prices** a challenging task.

Accurate forecasts are crucial for:
- Energy companies optimising operations
- Traders managing risk
- Consumers benefiting from better pricing
""")
