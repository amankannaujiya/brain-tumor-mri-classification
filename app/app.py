import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the trained model
model = tf.keras.models.load_model("../models/brain_tumor_mobilenet_model.keras")
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE = 150

# App title
st.title("🧠 Brain Tumor MRI Image Classifier")
st.markdown("Upload a brain MRI image and the model will predict the **tumor type**.")

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    - Upload an MRI image in **.jpg**, **.jpeg**, or **.png** format.
    - The image will be analyzed by the AI model.
    - You'll get the predicted **tumor type** with confidence score.
    """)

# Upload MRI image
uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼 Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    try:
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"### 🧠 Tumor Type: `{predicted_class}`")
        st.info(f"### 📊 Confidence: `{confidence:.2f}%`")

    except Exception as e:
        st.error("⚠️ Oops! Something went wrong during prediction.")
        st.code(str(e))

