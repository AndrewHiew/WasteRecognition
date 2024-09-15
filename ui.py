import streamlit as st
from streamlit_modal import Modal
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import tensorflow as tf
import concurrent.futures
import joblib
from skimage.feature import hog
import matplotlib.pyplot as plt

# Define a function to load the selected model
def load_selected_model(model_path):
    if model_path.endswith('.h5'):
        return load_model(model_path)
    elif model_path.endswith('.joblib'):
        return joblib.load(model_path)
    else:
        raise ValueError("Unsupported model format")

# Predict image method
def predict_image(image, model, model_type):
    resize = tf.image.resize(image, (256, 256))
    class_names = ['compostable', 'nonrecyclable', 'recyclable']
    if model_type == 'keras':
        predictions = model.predict(np.expand_dims(resize / 255.0, axis=0))
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        return predicted_class, predictions, confidence
    elif model_type == 'svm':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize or pad the image to ensure it has the same dimensions as the training images
        target_size = (256, 256)
        resized_image = cv2.resize(gray_image, target_size)

        # Extract HOG features from the resized image
        hog_features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

        # Make predictions using the loaded SVM classifier
        predictions = model.predict([hog_features])[0]
        predicted_class = class_names[predictions]
        return predicted_class, None, None

# Function to resize image using matplotlib (For Displaying, not Data Preprocessing)
def resize_image(image, size=(400, 400)):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    fig.set_size_inches(4, 4)
    fig.tight_layout(pad=0)

    # Convert the Matplotlib figure to a numpy array
    fig.canvas.draw()
    resized_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    resized_image = resized_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return resized_image

# Model Main Function
def model():
    text_container = st.container()

    # Model selection
    model_list = {
        'VGG19 Model': os.path.join('models', 'vgg19_model.h5'),
        'RestNet50 Model': os.path.join('models', 'restnet50_model.h5'),
        'SVM Waste Model': os.path.join('models', 'svm_waste_model.joblib')
    }
    selected_model_name = st.sidebar.selectbox("Select a model", list(model_list.keys()))
    selected_model_path = model_list[selected_model_name]

    # Load the selected model and store it in session state
    if 'model_path' not in st.session_state or st.session_state.model_path != selected_model_path:
        st.session_state.model = load_selected_model(selected_model_path)
        st.session_state.model_path = selected_model_path
        st.session_state.model_type = 'keras' if selected_model_path.endswith('.h5') else 'svm'

    # changing the header title
    st.session_state.maintitle = "Image Classifier: " + selected_model_path[7:]

    # Define Popup Window
    modal = Modal(key="file_uploader", title="Upload Files")
    upload_btn = st.sidebar.button("Upload File")
    if st.sidebar.toggle("Camera"):
        picture = st.sidebar.camera_input("Take a picture")
        if picture:
            with st.spinner("Processing Image"):
                image = cv2.imdecode(np.frombuffer(picture.read(), np.uint8), cv2.IMREAD_COLOR)
                st.session_state.image_data = image
                st.rerun()  

    if upload_btn:
        modal.open()

    if modal.is_open():
        with modal.container():
            uploaded_image = st.file_uploader("Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "bmp"])
            if st.button("Ok"):
                if uploaded_image is not None:
                    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        st.session_state.image_data = image
                        modal.close()
                    else:
                        st.error("Error: Failed to process the uploaded image.")
                else:
                    modal.close()

    # Predict the Image using the current loaded model
    if "image_data" in st.session_state:
        image = st.session_state.image_data

        if image is None:
            text_container.error("Error: Failed to load the image.")
        else:
            with st.spinner("Loading Response..."):
                # Multi Thread method to run the model in another thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(predict_image, image, st.session_state.model, st.session_state.model_type)
                    predicted_class, predictions, confidence = future.result()

                # Interpret predictions
                if st.session_state.model_type == 'keras':
                    with text_container:
                        # Resize the image before displaying
                        resized_image = resize_image(image)

                        # Display the resized image
                        st.image(resized_image, caption='Processed Image', use_column_width=False, width=400)

                        # Display predicted class and confidence
                        st.write("Predicted Class: ", predicted_class)
                        st.write("Prediction: ", predictions)
                        st.write("Confidence: ", confidence)
                elif st.session_state.model_type == 'svm':
                    with text_container:
                        # Resize the image before displaying
                        resized_image = resize_image(image)

                        # Display the resized image
                        st.image(resized_image, caption='Processed Image', use_column_width=False, width=400)

                        # Display predicted class for SVM
                        st.write("Predicted Class: ", predicted_class)

    return None

if __name__ == "__main__":
    if "maintitle" not in st.session_state:
        st.session_state.maintitle = "Image Classifier"
    title = st.header(st.session_state.maintitle, divider='rainbow')

    # Main logic to switch between pages
    if "page" not in st.session_state:
        st.session_state.page = "model"

    if st.session_state.page == "model":
        model()
