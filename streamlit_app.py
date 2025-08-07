
import streamlit as st
from PIL import Image
import numpy as np
import leaf  # your custom prediction logic

st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a leaf image to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    with st.spinner("Predicting..."):
        result = leaf.predict_image(image)  # Adjust based on your method
        st.success(f"Prediction: {result}")
