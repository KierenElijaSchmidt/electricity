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

    .block-container .stColumns {
    display: flex;
    align-items: stretch;  /* ensures children stretch equally */
}

    /* Custom card styling */
    .custom-card {
    background-color: #2a2a2a;
    border: 1px solid #404040;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);

    /* Add this */
    min-height: 220px;  /* ensures equal height */
    display: flex;
    flex-direction: column;
    justify-content: space-between;
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
</style>
""", unsafe_allow_html=True)

# Deliverables Section
st.markdown('<h1 class="section-title">Deliverables</h1>', unsafe_allow_html=True)

# First row of deliverables
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="custom-card">
        <h2 class="card-title">Type of Data</h2>
        <ul class="card-description">
            <li>Tabular Data (Pandas DataFrame)</li>
            <li>29.498 data entries</li>
            <li>Features: dates, demand, price, temperature (min and max), solar exposure, rainfall, schoolday, holiday </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


with col2:
    st.markdown("""
    <div class="custom-card">
        <h2 class="card-title">Task Definition</h2>
        <ul class="card-description">
            <li>Regression Task</li>
            <li>Predict the next days electricity price</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Second row of deliverables
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="custom-card">
        <h2 class="card-title">Goals and Deliverables</h2>
        <ul class="card-description">
            <li>Showing model performance on User Interface</li>
            <li>Visuals with key insights on demand and price</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-card">
        <h2 class="card-title">Performance Measurement</h2>
        <ul class="card-description">
            <li>Mean Squared Error</li>
            <li>Learning Curve</li>
            <li>Visuals of predivted vs actual values</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
