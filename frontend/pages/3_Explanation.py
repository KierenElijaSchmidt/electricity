import streamlit as st

st.set_page_config(page_title="Next Steps", page_icon="🚀", layout="wide")

# --------- CSS ----------
st.markdown("""
<style>
    .section-title{
        font-size: 34px; font-weight: 800; margin: 8px 0 24px 0;
    }
    .subtitle{
        color:#cfcfcf; font-size:18px; margin-bottom:28px;
    }
    .card{
        background:#2a2a2a; border:1px solid #3b3b3b; border-radius:14px;
        padding:22px; height:100%; box-shadow:0 6px 14px rgba(0,0,0,.25);
    }
    .card h3{
        margin:0 0 10px 0; font-size:22px; font-weight:700;
    }
    .pill{
        display:inline-block; padding:6px 10px; font-size:12px; font-weight:700;
        border-radius:999px; background:#3b3b3b; color:#ffd166; margin-bottom:12px;
        letter-spacing:.3px; text-transform:uppercase;
    }
    .card ul{margin:10px 0 0 18px; font-size:16px; line-height:1.6; color:#e8e8e8;}
    .footer{
        margin-top:26px; color:#bdbdbd; font-size:14px;
    }
</style>
""", unsafe_allow_html=True)

# --------- Header ----------
st.markdown('<div class="section-title">🚀 Next Steps & Roadmap</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Where we’re taking the predictor—from flexible inputs to real-world pilots and commercialization.</div>', unsafe_allow_html=True)

# --------- Cards ----------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
        <div class="pill">Flexibility</div>
        <h3>Make the Predictor More Flexible</h3>
        <ul>
            <li><b>Currently:</b> accepts NumPy vectors.</li>
            <li><b>Next:</b> accept any data type (CSV, JSON, DB tables).</li>
            <li><b>Bonus:</b> text input via a simple chatbot interface.</li>
        </ul>
        <div class="footer">Goal: easier inputs, faster experimentation.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
        <div class="pill">Collaboration</div>
        <h3>Work with Real Companies</h3>
        <ul>
            <li>Analyze their <b>specific datasets</b>.</li>
            <li>Enrich with <b>external data sources</b> (weather, market, policy).</li>
            <li>Build a <b>customized pilot</b> that fits their workflow.</li>
        </ul>
        <div class="footer">Outcome: measurable value on real operations.</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
        <div class="pill">Commercialization</div>
        <h3>From Pilot to Product</h3>
        <ul>
            <li>Scale a successful pilot into production.</li>
            <li>Offer <b>forecasting solutions</b> as a service or license.</li>
            <li>Provide support, SLAs, and dashboards.</li>
        </ul>
        <div class="footer">Path: pilot ➜ scale ➜ sell.</div>
    </div>
    """, unsafe_allow_html=True)
