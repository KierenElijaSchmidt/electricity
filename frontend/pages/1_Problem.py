import streamlit as st

st.set_page_config(layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .section-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 30px;
        color: #ffffff;
    }
    .custom-card {
        background-color: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 40px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .card-title {
        font-size: 22px;
        font-weight: 600;
        color: #ffcc00;
        margin-bottom: 15px;
    }
    .card-text {
        font-size: 18px;
        color: #cccccc;
        line-height: 1.6;
        margin-bottom: 15px;
    }
    .highlight-box {
        background-color: #333333;
        border-left: 5px solid #ffcc00;
        padding: 24px;
        margin-top: 50px;
        border-radius: 8px;
        font-size: 18px;
        color: #ffffff;
    }
    ul {
        margin-top: 10px;
        font-size: 18px;
        color: #cccccc;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="section-title">⚠️ Problem Description</div>', unsafe_allow_html=True)

# --- First card: Supply & Demand ---
with st.container():
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">1) Fluctuations in supply and demand</div>
            <div class="card-text">
                Electricity demand spikes during hot summers (air conditioning) and cold winters (heating).
                Supply shocks (generator outages, low renewable input) can also cause sudden price surges.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("assets/barcharts/rrp_vs_date.png", use_container_width=True, caption="Electricity Price (RRP) over Time")

# --- Second card: Weather & Renewables ---
with st.container():
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">2) Weather conditions and renewable energy integration</div>
            <div class="card-text">
                Weather drives both demand (hot days → higher usage) and supply (solar & wind variability).
                This dual effect makes forecasting more complex.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("assets/barcharts/rrp_weather_2018_2020.png", use_container_width=True, caption="Electricity Price vs Weather Conditions")

# --- Third card: Policy & Market Behaviour ---
with st.container():
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">3) Policy, regulations, and market behaviour</div>
            <div class="card-text">
                Regulatory changes and power plant closures can reshape the market overnight.
                Example: the Hazelwood coal plant closure (2017) led to a significant price increase.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("assets/barcharts/rrp_hazelwood.png", use_container_width=True, caption="Impact of Hazelwood Closure on RRP")

# --- Highlight section: Why forecasts matter ---
st.markdown("""
<div class="highlight-box">
    This volatility makes <b>predicting electricity prices</b> a challenging task. <br><br>
    Accurate forecasts are crucial for:
    <ul>
        <li>⚡ Energy companies optimising operations</li>
        <li>📈 Traders managing risk</li>
        <li>🏠 Consumers benefiting from better pricing</li>
    </ul>
</div>
""", unsafe_allow_html=True)
