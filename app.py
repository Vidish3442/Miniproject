import streamlit as st
import gdown
import os
from keras.models import load_model
from PIL import Image
import numpy as np
import time

# Page configuration
st.set_page_config(page_title="Diabetic Retinopathy Detector", page_icon="🩺", layout="centered")

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #f4f6f8;
}
.stButton>button {
    background-color: #007bff;
    color: white;
    height: 3em;
    width: 12em;
    font-size: 18px;
    border-radius: 10px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #0056b3;
    transform: scale(1.05);
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 30px;
    text-align: center;
}
.conf-bar {
    border-radius: 10px;
    height: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;color:#333;'>🩺 Diabetic Retinopathy Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;'>Upload a retina image to check DR severity</p>", unsafe_allow_html=True)

# Model setup
MODEL_URL = "https://drive.google.com/uc?id=11DVFnbesDNaxrqCSAwgC8t76Ntgdu2q_"
MODEL_PATH = "dr_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (200MB)..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

model = load_model(MODEL_PATH)
classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Colors & emojis for severity
colors = {
    "No_DR": "#28a745",          # Green
    "Mild": "#17a2b8",           # Blue
    "Moderate": "#ffc107",       # Yellow
    "Severe": "#fd7e14",         # Orange
    "Proliferate_DR": "#dc3545"  # Red
}
emojis = {
    "No_DR": "🟢",
    "Mild": "🙂",
    "Moderate": "😐",
    "Severe": "😟",
    "Proliferate_DR": "😢"
}

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Card layout
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Animated prediction bar
    with st.spinner("Predicting..."):
        progress_bar = st.progress(0)
        for i in range(0, 101, 10):
            progress_bar.progress(i)
            time.sleep(0.05)
        preds = model.predict(img_array)

    pred_class = classes[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100

    # Prediction with color + emoji
    st.markdown(f"<h2 style='color:{colors[pred_class]};'>{emojis[pred_class]} Prediction: {pred_class}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

    # Dynamic confidence bar
    bar_color = colors[pred_class]
    st.markdown(f"""
    <div class='conf-bar' style='background-color:{bar_color}; width:{int(confidence)}%'></div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
