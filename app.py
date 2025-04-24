import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

st.title("Rice Grain Type Classifier")

model = load_model("models/rice_model.h5")
class_names = sorted(os.listdir("data/train")) 

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((250, 250))
    st.image(img, caption='Uploaded Image',  use_container_width=True)
    

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    st.success(f"Prediction: **{class_names[np.argmax(pred)]}**")
