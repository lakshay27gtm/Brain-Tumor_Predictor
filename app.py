import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Define the directories (assuming they are set up as per previous steps)
train_dir = "/content/brain_tumor_dataset/train"
test_dir = "/content/brain_tumor_dataset/test"

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

# Get lists of image files from train and test directories
all_train_images = []
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            all_train_images.append(os.path.join(class_path, fname))

all_test_images = []
for class_name in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_name)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            all_test_images.append(os.path.join(class_path, fname))

all_images = all_train_images + all_test_images

# Create a dropdown for image selection
selected_image_path = st.selectbox('Select an image for prediction:', all_images)

if selected_image_path:
    st.write(f'Selected image: {selected_image_path}')
    # Display the selected image
    image = cv2.imread(selected_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for matplotlib/streamlit display
    st.image(image_rgb, caption='Selected Image', use_column_width=True)

    if st.button('Predict'):
        # Preprocess the image for the model
        image_to_predict = cv2.imread(selected_image_path)
        image_to_predict = cv2.resize(image_to_predict, (224, 224))
        image_to_predict = np.expand_dims(image_to_predict, axis=0) # Add batch dimension
        image_to_predict = image_to_predict.astype(np.float32) / 255.0 # Normalize pixel values and ensure float32

        # Make prediction using TFLite interpreter
        interpreter.set_tensor(input_details[0]['index'], image_to_predict)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0] # Get output and access the single value

        # Interpret the prediction
        if prediction > 0.5:
            st.write('Prediction: **Tumor**')
        else:
            st.write('Prediction: **Not a Tumor**')
        st.write(f'Confidence (probability of Tumor): {prediction:.4f}')
