import streamlit as st
# Custom CSS for styling
st.markdown("""
<style>
    .deliverable-card {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #DC3545;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .objectives-box {
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }
    .phase-circle {
        display: inline-block;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #DC3545;
        color: white;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)
# Page title
st.title(":clipboard: Project Deliverables")
# Objectives and Measurement Section
st.markdown('<div class="objectives-box">', unsafe_allow_html=True)
st.markdown("### :dart: Objectives and Measurement")
st.markdown("""
**Our goal is to develop a simple and clean UI with inputs for date, temperature forecast and past
demand. The output will be a predicted electricity price, visualized and numerical. The success of our
project will be measured by RMSE, MAE, and MAPE.**
*- Vision for the Electricity Price Predictor*
""")
st.markdown('</div>', unsafe_allow_html=True)
# Project Phases
st.markdown("## :arrows_counterclockwise: Project Phases")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="phase-circle">01</div>', unsafe_allow_html=True)
    st.markdown("**Phase name**")
with col2:
    st.markdown('<div class="phase-circle">02</div>', unsafe_allow_html=True)
    st.markdown("**Phase name**")
with col3:
    st.markdown('<div class="phase-circle">03</div>', unsafe_allow_html=True)
    st.markdown("**Phase name**")
with col4:
    st.markdown('<div class="phase-circle">04</div>', unsafe_allow_html=True)
    st.markdown("**Phase name**")
st.markdown("---")
# Main Deliverables Section
st.markdown("## :rocket: Key Deliverables")
# Create 2x2 grid for deliverables
col1, col2 = st.columns(2)
with col1:
    # Clean Database
    st.markdown('<div class="deliverable-card">', unsafe_allow_html=True)
    st.markdown("### :card_file_box: Clean Database")
    st.markdown("""
    **Preprocessing**
    - Over 1.3m rows
    - 10 columns for time-series
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    # Presentation
    st.markdown('<div class="deliverable-card">', unsafe_allow_html=True)
    st.markdown("### :bar_chart: Presentation")
    st.markdown("""
    **Consolidation**
    - Showing final UI and model performance
    """)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    # Exploratory Data Analysis
    st.markdown('<div class="deliverable-card">', unsafe_allow_html=True)
    st.markdown("### :chart_with_upwards_trend: Exploratory Data Analysis")
    st.markdown("""
    **Visuals with key insights on:**
    - Demand, price, temperature
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    # More Everything
    st.markdown('<div class="deliverable-card">', unsafe_allow_html=True)
    st.markdown("### :heavy_plus_sign: More Everything")
    st.markdown("""
    **That adds value on the way**
    """)
    st.markdown('</div>', unsafe_allow_html=True)
