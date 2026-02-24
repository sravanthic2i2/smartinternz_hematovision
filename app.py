import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(page_title="HematoVision", page_icon="🩸")

st.title("🩸 HematoVision - Blood Cell Classification")
st.write("Upload a microscopic blood cell image to classify the cell type.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("blood_cell_final.h5")

model = load_model()

class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")