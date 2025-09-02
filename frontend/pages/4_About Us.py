import streamlit as st
from streamlit_image_select import image_select
import os
import os.path

# -------------------------------
# Page Config
# -------------------------------

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
        font-size: 34px; /* same as text */
        line-height: 1;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="section-title"><span class="icon">👥</span> About Us</div>', unsafe_allow_html=True)

# -------------------------------
# Example team members
# -------------------------------
# Put your team pictures in a folder, e.g. "assets/team/"
# Use relative paths from repo root
# st.write("Current working directory:", os.getcwd())
# st.write("Files in current directory:", os.listdir('.'))

# # Check if your path exists

# img_path = "frontend/assets/team/Alessio.png"
# st.write(f"Path exists: {os.path.exists(img_path)}")

TEAM = [
    {
        "name": "Alessio",
        "img": "assets/team/Alessio.png",
        "bio": "Alessio focused on the frontend integration.",
    },
     {
         "name": "Niko",
         "img": "assets/team/Niko.png",
         "bio": "Niko focused Machine learning models.",
     },
     {
         "name": "Kieren",
         "img": "assets/team/Kieren.png",
         "bio": "Kieren focused on Deep learinng models and ML Ops.",
     },
     {
         "name": "Filippa",
         "img": "assets/team/Filippa.png",
         "bio": "Filippa focused on Deep learning models and ML Ops",
     },
]

# -------------------------------
# Image Selector
# -------------------------------
selected_index = image_select(
    label="Click on a picture to learn more:",
    images=[member["img"] for member in TEAM],
    captions=[member["name"] for member in TEAM],
    index=0,
    return_value="index",
)

# -------------------------------
# Show details of selected member
# -------------------------------
member = TEAM[selected_index]
st.write(member["bio"])
