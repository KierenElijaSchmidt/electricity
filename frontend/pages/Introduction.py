import streamlit as st
from streamlit_image_select import image_select

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Team Introduction",
    page_icon="👥",
    initial_sidebar_state="collapsed",
)

st.markdown("# 👋 Meet the Team")

# -------------------------------
# Example team members
# -------------------------------
# Put your team pictures in a folder, e.g. "assets/team/"
# Use relative paths from repo root
TEAM = [
    {
        "name": "Alessio",
        "img": "frontend/assets/team/Alessio.png",
        "bio": "Alessio focused on the frontend integration.",
    },
     {
         "name": "Niko",
         "img": "frontend/assets/team/Niko.png",
         "bio": "Niko focused Machine learning models.",
     },
     {
         "name": "Kieren",
         "img": "frontend/assets/team/Kieren.png",
         "bio": "Kieren focused on Deep learinng models and ML Ops.",
     },
     {
         "name": "Filippa",
         "img": "frontend/assets/team/Filippa.png",
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
#st.subheader(f"{member['name']} — {member['role']}")
st.write(member["bio"])
