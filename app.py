import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from PIL import Image
import os

# Load the trained model
MODEL_PATH = "braintumormodel.h5"
model = load_model(MODEL_PATH)

# Streamlit app
st.title("Brain Tumor Detection")
st.write("You can use various MRI images from kaggle dataset to test the model")
st.write("Upload an image to determine if it indicates a brain tumor or is healthy.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    """
    Preprocess the uploaded image to match the model's input size.
    Args:
        image: PIL Image
    Returns:
        Preprocessed image ready for model prediction.
    """
    # Convert PIL Image to OpenCV format
    image = np.array(image)
    
    # Check if the image has only 1 channel (grayscale), and convert it to 3 channels (RGB)
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    
    # Resize image to model's expected size (224x224)
    image = cv2.resize(image, (224, 224))
    
    # Convert to array and preprocess
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    try:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = (model.predict(preprocessed_image) > 0.5).astype(int)[0, 0]

        # Map prediction to class label
        if prediction == 0:
            st.write("Healthy Brain")
        else:
            st.write("The Brain has Tumor")
    except Exception as e:
        st.error(f"Error processing the image: {e}")

# Footer
st.write("Developed by Omkar")
