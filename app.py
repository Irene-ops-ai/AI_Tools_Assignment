# streamlit_app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🧠")
st.title("🧠 MNIST Digit Classifier")
st.write("Upload a digit image (28x28 grayscale) to predict its value.")

# Show a helpful message before upload
uploaded_file = st.file_uploader("📁 Upload a digit image...", type=["png", "jpg", "jpeg"])

# Check model existence before using it
model_path = "mnist_model.h5"
if not os.path.exists(model_path):
    st.error("Model file 'mnist_model.h5' not found! Please upload it or include it in your app folder.")
else:
    # Only run prediction when file is uploaded
    if uploaded_file is None:
        st.info("👆 Please upload an image to get a prediction.")
    else:
        try:
            # Load and preprocess image
            image = Image.open(uploaded_file).convert("L").resize((28, 28))
            img_array = np.array(image) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            # Load model
            model = tf.keras.models.load_model(model_path)

            # Predict
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)

            # Display
            st.image(image, caption="🖼 Uploaded Image", width=150)
            st.success(f"### Predicted Digit: {predicted_digit}")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

