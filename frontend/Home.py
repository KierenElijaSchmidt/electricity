import streamlit as st

st.set_page_config(
    page_title="Vehicle Incidents in England",
    page_icon=":pickup_truck:",
    initial_sidebar_state="expanded",
)

st.write("# VEHICLE INCIDENTS IN ENGLAND :pickup_truck:")

st.sidebar.success("Select a page above.")

st.markdown(
    """
##### We are exploring Vehicle incidents across England whilst utilising the capabilities of Streamlit, Snowflake, Carto and Tableau.

Click through the pages to view all towns in England based on real data, then see how this relates to Integrated Care Boards.

Explore the details of your chosen city and see how this is indexed into the H3 geospatial grid (Hexagons). Carto's Toolkit shared within Snowflake has been utilised for this.

Finally, explore how vehicle incidents relate to fire service areas.

All the raw data has been curated, engineered and processed with Snowflake.
Enjoy :smile:

All code can be accessed here:
https://github.com/beckyoconnor/Vehicle_Incidents_uk/
"""
)
