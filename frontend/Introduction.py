import streamlit as st

st.set_page_config(
    page_title="Electricity Price Prediction",
    #page_icon=":pickup_truck:",
    initial_sidebar_state="expanded",
)

st.write("# Electricity Price Prediction")

# st.markdown(
#     """
# Electricity markets are shaped by volatile factors such as weather, demand cycles, and policy changes. Anticipating these fluctuations is essential for companies to manage costs and plan effectively.
# This project applies machine and deep learning to predict electricity prices using historical data, weather conditions, and seasonal patterns. The following pages outline the problem statement, a live demonstration, and a plan for next steps.
# """)



st.image("assets/title/electricity.jpg", use_container_width=True)

st.markdown('''The codebase for this project can be accessed here: https://github.com/KierenElijaSchmidt/electricity''')
