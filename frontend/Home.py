import streamlit as st

st.set_page_config(
    page_title="Electricity Price Prediction",
    #page_icon=":pickup_truck:",
    initial_sidebar_state="expanded",
)

st.write("# Electricity Price Prediction")

st.sidebar.success("Select a page above.")

st.markdown(
    """
##### We are predicting Electricity prices based on the analysis of weather data, passed electricity prices and seasonal features.

In the following pages we will present the problem that price prediction poses for companies, our solution and a functional demo.""")



st.image("assets/title/electricity.jpg", use_container_width=True)

st.markdown('''All the code can be accessed here: https://github.com/KierenElijaSchmidt/electricity''')
