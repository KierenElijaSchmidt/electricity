import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(layout="wide")

# --- Shared CSS (smaller cards + general layout) ---
st.markdown("""
    <style>
    .section-title {
        font-size: 34px;
        font-weight: 800;
        margin: 12px 0 24px 0;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #ffffff;
    }
    .section-title span.icon {
        font-size: 34px;
        line-height: 1;
    }
    .custom-card {
        background-color: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 12px;
        padding: 16px;                 /* smaller padding */
        margin-bottom: 24px;           /* smaller margin */
        box-shadow: 0 4px 6px rgba(0,0,0,0.25);
        max-width: 520px;              /* cap so charts get more width */
    }
    .card-title {
        font-size: 20px;               /* slightly smaller */
        font-weight: 700;
        color: #ffcc00;
        margin-bottom: 10px;
    }
    .card-text {
        font-size: 16px;               /* slightly smaller */
        color: #cccccc;
        line-height: 1.55;
        margin-bottom: 8px;
    }
    .highlight-box {
        background-color: #333333;
        border-left: 5px solid #ffcc00;
        padding: 18px;                 /* smaller */
        margin-top: 28px;              /* tighter spacing */
        border-radius: 8px;
        font-size: 16px;
        color: #ffffff;
        max-width: 980px;              /* keep highlight readable */
    }
    ul {
        margin-top: 8px;
        font-size: 16px;
        color: #cccccc;
    }
    .chart-caption {
        color: #bdbdbd;
        font-size: 13px;
        margin-top: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="section-title"><span class="icon">⚠️</span> Problem Statement</div>', unsafe_allow_html=True)

# ---------- Helpers ----------
def _resolve_asset(rel_path: str) -> Path:
    """Resolve asset path robustly from /pages or root."""
    rel = Path(rel_path)
    if rel.exists():
        return rel
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / rel,
        here.parent / rel,            # same dir as page
        here.parents[1] / rel,        # project root if page under /pages
        here.parents[2] / rel,        # one higher (just in case)
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Asset not found: {rel_path} (tried: {[str(c) for c in candidates]})")

def chart(img_rel_path: str, caption: str, target_height: int = 420):
    """
    Show image via st.image (works for local files).
    Resizes to a uniform target height to make layouts consistent.
    """
    try:
        p = _resolve_asset(img_rel_path)
        img = Image.open(p).convert("RGBA")
        if target_height is not None and img.height != target_height:
            # keep aspect ratio when resizing to the target height
            new_w = max(1, round(img.width * (target_height / img.height)))
            img = img.resize((new_w, target_height), Image.LANCZOS)
        st.image(img, caption=caption, use_container_width=True)
        st.markdown('<div class="chart-caption"></div>', unsafe_allow_html=True)
    except FileNotFoundError as e:
        st.warning(str(e))

def text_card(title, text):
    st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">{title}</div>
            <div class="card-text">{text}</div>
        </div>
    """, unsafe_allow_html=True)

# --- First: Supply & Demand ---
with st.container():
    col_text, col_chart = st.columns([0.8, 1.6])  # give charts more space
    with col_text:
        text_card(
            "1) Fluctuations in supply and demand",
            """Electricity demand spikes during hot summers (air conditioning) and cold winters (heating).
               Supply shocks (generator outages, low renewable input) can also cause sudden price surges."""
        )
    with col_chart:
        chart("assets/barcharts/rrp_vs_date.png", "Electricity Price (RRP) over Time")

# --- Second: Weather & Renewables ---
with st.container():
    col_text, col_chart = st.columns([0.8, 1.6])
    with col_text:
        text_card(
            "2) Weather conditions and renewable energy integration",
            """Weather drives both demand (hot days → higher usage) and supply (solar & wind variability).
               This dual effect makes forecasting more complex."""
        )
    with col_chart:
        chart("assets/barcharts/rrp_weather_2018_2020.png", "Electricity Price vs Weather Conditions")

# --- Third: Policy & Market Behaviour ---
with st.container():
    col_text, col_chart = st.columns([0.8, 1.6])
    with col_text:
        text_card(
            "3) Policy, regulations, and market behaviour",
            """Regulatory changes and power plant closures can reshape the market overnight.
               Example: the Hazelwood coal plant closure (2017) led to a significant price increase."""
        )
    with col_chart:
        chart("assets/barcharts/rrp_hazelwood.png", "Impact of Hazelwood Closure on RRP")
