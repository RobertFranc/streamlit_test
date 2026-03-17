import streamlit as st

st.title("Streamlit testing")
st.write("This is a test app for deployment.")

name = st.text_input("Write your name?")
if name:
    st.success(f"Welcome to the cloud, {name}!")
