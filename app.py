import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model(r"C:\Users\anmol\Desktop\brain_tumor_detection-using_CNN\model.h5")


# Class labels
labels = ['Healthy', 'Tumor']

def preprocess_image(img):
    img = img.resize((64, 64))
    img = np.array(img)
    if img.shape[-1] == 4:  # handle PNG with alpha
        img = img[:, :, :3]
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to check for a tumor.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    if st.button("Predict"):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)

        st.write("### Prediction:")
        st.success(f"🧾 {labels[class_index]} ({prediction[0][class_index]*100:.2f}%)")
