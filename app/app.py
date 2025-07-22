import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Config
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠")

# Load model
model = tf.keras.models.load_model("../models/brain_tumor_mobilenet_model_with_unknown.keras")
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary', 'Unknown']

IMG_SIZE = 150

# Title
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload a brain MRI image and the model will predict the **tumor type**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### 🧠 Predicted Tumor Type: `{predicted_class}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")

    if predicted_class == "Unknown":
        st.warning("⚠️ This image does not appear to be a valid MRI. Please try again with a proper brain scan.")

