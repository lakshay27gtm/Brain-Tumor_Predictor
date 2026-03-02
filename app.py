import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image # Import Pillow for image handling
import io # Import io for handling byte streams

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="brain_tumor_model.h5")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get input and output details for the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title('Brain Tumor Detection')
st.write("Upload an MRI image to predict if it shows a Brain Tumor or Not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image file as bytes
    image_bytes = uploaded_file.getvalue()
    # Convert bytes to numpy array
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for matplotlib/streamlit display
        st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            # Preprocess the image for the model
            image_to_predict = cv2.resize(image, (224, 224))
            image_to_predict = np.expand_dims(image_to_predict, axis=0) # Add batch dimension
            image_to_predict = image_to_predict.astype(np.float32) / 255.0 # Normalize pixel values and ensure float32

            # Make prediction using TFLite interpreter
            interpreter.set_tensor(input_details[0]['index'], image_to_predict)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0] # Get output and access the single value

            confidence = float(prediction) # The prediction is already a single value

            # Interpret the prediction
            if confidence > 0.5:
                st.error("Tumor Detected")
            else:
                st.success("No Tumor Detected")

            st.write(f'Confidence (probability of Tumor): {confidence:.4f}')
    else:
        st.error("Could not load the image. Please try again with a different file.")
