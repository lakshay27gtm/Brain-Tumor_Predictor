# Brain Tumor Detection with Deep Learning

This project implements a Convolutional Neural Network (CNN) for the detection of brain tumors from MRI images. The model leverages transfer learning with a VGG16 base, incorporates data augmentation, hyperparameter tuning, and early stopping to achieve an optimized performance.

## Features:
-   **Deep Learning Model**: Utilizes a custom CNN built on top of a pre-trained VGG16 model (Transfer Learning).
-   **Data Augmentation**: Employs aggressive data augmentation techniques to enhance model generalization.
-   **Hyperparameter Tuning**: Fine-tuned optimizer (Adam with a specific learning rate) and model architecture (dense layers).
-   **Callbacks**: Implements EarlyStopping and ModelCheckpoint for robust training and saving the best performing model.
-   **Interactive Dashboard**: A Streamlit application is provided to interactively classify images (Tumor or Not a Tumor).

## Technologies Used:
-   Python
-   TensorFlow / Keras
-   OpenCV
-   Scikit-learn
-   Matplotlib
-   Streamlit

## Getting Started:
1.  **Clone the Repository (if applicable)**
2.  **Install Dependencies**: Use the `requirements.txt` file to install all necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare Dataset**: Ensure your dataset is organized into `train` and `test` directories, with `tumor` and `normal` subdirectories, as handled by the initial setup script.
4.  **Run the Streamlit Application**: To launch the interactive dashboard, navigate to the project directory in your terminal and run:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser, allowing you to select images and get predictions from the model.
