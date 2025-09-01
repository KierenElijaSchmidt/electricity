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
        "name": "Alice",
        "role": "Data Scientist",
        "img": "assets/team/alice.png",
        "bio": "Alice specializes in deep learning and model deployment.",
    },
    {
        "name": "Bob",
        "role": "Project Manager",
        "img": "assets/team/bob.png",
        "bio": "Bob keeps the team on track and ensures smooth delivery.",
    },
    {
        "name": "Charlie",
        "role": "Frontend Engineer",
        "img": "assets/team/charlie.png",
        "bio": "Charlie designs and builds the user-facing applications.",
    },
    {
        "name": "Diana",
        "role": "Research Analyst",
        "img": "assets/team/diana.png",
        "bio": "Diana works on data gathering, research, and insights.",
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
st.subheader(f"{member['name']} — {member['role']}")
st.write(member["bio"])
