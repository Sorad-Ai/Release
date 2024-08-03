import streamlit as st1

def main():
    st1.header("Dashboard")
    st1.markdown("<br><br>", unsafe_allow_html=True)
    st1.button('Virtual Calculator', key='calc1')
    st1.text("Calculator made with computer vision technology.")
    st1.text("Python modules like opencv, cvzone, numpy, mediapipe are used")
    st1.markdown("<br>", unsafe_allow_html=True)

    st1.button("Objext Delection")
    st1.text("Realtime object detector with computer vision technology.")
    st1.text("Python modules like opencv, yolo, are used")
    st1.markdown("<br>", unsafe_allow_html=True)