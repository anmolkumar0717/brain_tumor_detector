import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import streamlit.components.v1 as components

# Load model
model = load_model(r"C:\Users\anmol\Desktop\brain_tumor_detection\model.h5")

# Page settings
st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="wide")

# Labels
labels = ['Healthy', 'Tumor']

# Match background to 3D object (assumed #1A1A1A)
st.markdown("""
    <style>
    /* Override default styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #1A1A1A !important;  /* Match to Spline background */
        color: #ffffff !important;
    }
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 18px;
        color: #cccccc;
        margin-bottom: 25px;
    }
    .section {
        background-color: #262626;  /* Slight contrast for sections */
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .result-text {
        font-size: 28px;
        font-weight: 600;
        color: #4cc9f0;
        margin-top: 20px;
    }
    .confidence-bar {
        margin-top: 15px;
    }
    footer {
        text-align: center;
        margin-top: 50px;
        color: #999;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Preprocess function
def preprocess_image(img):
    img = img.resize((64, 64))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Layout: 5 columns to space things out
spacer1, left_col, mid_spacer, right_col, spacer2 = st.columns([0.5, 2.5, 0.5, 3, 0.5])

# LEFT COLUMN (Uploader & Prediction)
with left_col:
    st.markdown('<div class="main-title">🧠 Brain Tumor Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload an MRI scan to detect signs of a brain tumor using AI</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded MRI Image", use_column_width=True)

        if st.button("🔬 Predict"):
            with st.spinner("Analyzing..."):
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)[0]
                class_index = np.argmax(prediction)
                confidence = prediction[class_index] * 100

            st.markdown(f'<div class="result-text">🧾 Prediction: {labels[class_index]}</div>', unsafe_allow_html=True)
            st.write(f"🧠 Confidence: **{confidence:.2f}%**")

            st.markdown('<div class="confidence-bar">', unsafe_allow_html=True)
            for i, label in enumerate(labels):
                st.write(f"**{label}**")
                st.progress(int(prediction[i] * 100))
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN (3D View)
with right_col:
    st.markdown("### 🧬 Explore Brain in 3D")
    components.html(
        """
        <iframe src='https://my.spline.design/particleaibrain-5f7ifU1Bz1nrsYdrJHdFGYlP/' 
                frameborder='0' 
                width='100%' 
                height='700px' 
                style='border-radius: 12px; background-color: transparent;'>
        </iframe>
        """,
        height=700,
    )

# Footer
st.markdown("<footer>© 2025 BrainScan AI · Built with ❤️ using Streamlit</footer>", unsafe_allow_html=True)
