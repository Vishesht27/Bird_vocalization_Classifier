import streamlit as st
from backend.serving import serving

# Heading of the app
st.title("Bird Vocal Classifier")

#Taking Audio Input file from user
uploaded_file = st.file_uploader("Choose an audio file", type="wav")
# convert wav to array
# result = infer(array)