import streamlit as st
import gdown
import os
from keras.models import load_model
from PIL import Image
import numpy as np

# App title
st.title("🩺 Diabetic Retinopathy Detector")
st.write("Upload a retina image to detect the severity of DR.")

# Model download
MODEL_URL = "https://drive.google.com/uc?id=11DVFnbesDNaxrqCSAwgC8t76Ntgdu2q_"
MODEL_PATH = "dr_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (200MB)..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

# Load model
model = load_model(MODEL_PATH)
classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    preds = model.predict(img_array)
    pred_class = classes[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100

    # Show results
    st.markdown(f"**Prediction:** {pred_class}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
