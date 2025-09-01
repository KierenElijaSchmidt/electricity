import streamlit as st

# Configure page
st.set_page_config(
    page_title="Electricity Price Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }

    /* Hide default streamlit elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}

    /* Custom card styling */
    .custom-card {
        background-color: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .icon-container {
        background-color: #404040;
        border-radius: 8px;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 16px;
        font-size: 24px;
    }

    .icon-red {
        background-color: #4a2c2c;
        color: #ff6b6b;
    }

    .section-title {
        color: #ffffff;
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        margin: 60px 0 40px 0;
        position: relative;
    }

    .section-title::after {
        content: '';
        display: block;
        width: 80px;
        height: 4px;
        background-color: #ff6b6b;
        margin: 16px auto 0;
        border-radius: 2px;
    }

    .card-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .card-subtitle {
        color: #888888;
        font-size: 14px;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .card-description {
        color: #cccccc;
        font-size: 16px;
        line-height: 1.6;
    }

    .vision-card {
        background-color: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 12px;
        padding: 32px;
        margin: 16px 0;
        text-align: center;
    }

    .vision-title {
        color: #ff6b6b;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }

    .vision-text {
        color: #ffffff;
        font-size: 20px;
        font-style: italic;
        line-height: 1.4;
    }

    /* Metric styling */
    .metric-highlight {
        color: #ff6b6b;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Objectives and Measurement Section
st.markdown('<h1 class="section-title">Objectives and Measurement</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("""
    <div class="custom-card">
        <div class="icon-container icon-red">
            🎯
        </div>
        <h2 class="card-title">Our Goal</h2>
        <p class="card-description">
            To develop a simple and clean UI with inputs for date, temperature forecast and
            past demand. The output will be a predicted electricity price, visualized and numeric.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-card">
        <div class="icon-container">
            📊
        </div>
        <h2 class="card-title">Measurement</h2>
        <p class="card-description">
            Project will be measured by <span class="metric-highlight">RMSE</span>, <span class="metric-highlight">MAE</span>,
            and <span class="metric-highlight">MAPE</span>.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="vision-card">
        <div class="vision-title">
            📈 Vision
        </div>
        <p class="vision-text">
            "Vision for the Electricity Price Predictor"
        </p>
    </div>
    """, unsafe_allow_html=True)

# Deliverables Section
st.markdown('<h1 class="section-title">Deliverables</h1>', unsafe_allow_html=True)

# First row of deliverables
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="custom-card">
        <div class="icon-container">
            💾
        </div>
        <h2 class="card-title">Clean Database</h2>
        <p class="card-subtitle">Data Processing</p>
        <p class="card-description">
            Processing over 1.3m rows, 10 columns for time-series data
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-card">
        <div class="icon-container">
            ⚡
        </div>
        <h2 class="card-title">Application Consolidation</h2>
        <p class="card-subtitle">Final Implementation</p>
        <p class="card-description">
            Showing final UI and model performance
        </p>
    </div>
    """, unsafe_allow_html=True)

# Second row of deliverables
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="custom-card">
        <div class="icon-container icon-red">
            📈
        </div>
        <h2 class="card-title">Exploratory Data Analysis</h2>
        <p class="card-subtitle">Key Insights</p>
        <p class="card-description">
            Visuals with key insights on demand, price, temperature
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-card">
        <div class="icon-container">
            ➕
        </div>
        <h2 class="card-title">More Everything</h2>
        <p class="card-subtitle">Added Value</p>
        <p class="card-description">
            Everything that adds value on the way
        </p>
    </div>
    """, unsafe_allow_html=True)
