import streamlit as st

from calc import calc
from obj import obj
from dash import main
import time

# Set the page config to wide mode
st.set_page_config(
    page_title="Sorad AI",
     page_icon="./logo.png",
    layout="wide",  # This makes the layout wide
    initial_sidebar_state="expanded"  # Optional: This expands the sidebar by default
)


# Create a sidebar
st.sidebar.image("./logo.png", width=250)
st.sidebar.title("Navigation")
if st.sidebar.button("Dashboard"):
    main()
if st.sidebar.button("Virtual Calculator"):
    time.sleep(1)
    calc()
if st.sidebar.button("Object Detector"):
    time.sleep(1)
    obj()

i = 0
if i <1:
    main()
    i = i +1
