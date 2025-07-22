# 🧠 Brain Tumor MRI Image Classifier

This deep learning project classifies brain MRI images into multiple tumor categories using transfer learning with MobileNetV2.

## 🧪 Classes
- Glioma
- Meningioma
- Pituitary
- No Tumor
- Unknown (non-MRI images)

## 📸 Live Demo
👉 [Click here to open Streamlit App](https://your-deployment-url)

## 📁 Project Structure

brain_tumor_project/
├── app/
│ └── app.py # Streamlit UI
├── models/
│ └── brain_tumor_mobilenet_model.keras
├── notebooks/
│ └── model_training.ipynb # Full training + evaluation
├── data/
│ └── BrainTumorData/ # glioma, meningioma, etc.
├── requirements.txt
├── README.md

markdown
Copy
Edit

## 🚀 Tech Stack
- Python
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- Streamlit
- Data Augmentation
- Matplotlib + Seaborn

## 🏥 Use Cases
- Assisting radiologists in tumor detection
- Triage patients based on tumor risk
- Automating second-opinion tools in telemedicine

## ✅ Final Notes
Model trained with EarlyStopping, data augmentation, and class weights.
Includes an "unknown" class to detect non-MRI uploads.

