import streamlit as st
from streamlit_modal import Modal
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import tensorflow as tf
import concurrent.futures
import joblib  # Assuming you are using joblib to load SVM models
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, ClientSettings
import av
from skimage.feature import hog

# Define a function to load the selected model
def load_selected_model(model_path):
    if model_path.endswith('.h5'):
        return load_model(model_path)
    elif model_path.endswith('.joblib'):
        return joblib.load(model_path)
    else:
        raise ValueError("Unsupported model format")

# Define a function to predict image
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

# Define a class to capture video frames
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        st.session_state.image_data = img
        return img

# Define the main model function
def model():
    text_container = st.container()

    # Model selection
    model_list = {
        'VGG19 Model': os.path.join('models', 'vgg19_model.h5'),
        'Waste Recognition Model': os.path.join('models', 'restnet50_model.h5'),
        'SVM Waste Model': os.path.join('models', 'svm_waste_model.joblib')
    }
    selected_model_name = st.sidebar.selectbox("Select a model", list(model_list.keys()))
    selected_model_path = model_list[selected_model_name]

    # Load the selected model and store it in session state
    if 'model_path' not in st.session_state or st.session_state.model_path != selected_model_path:
        st.session_state.model = load_selected_model(selected_model_path)
        st.session_state.model_path = selected_model_path
        st.session_state.model_type = 'keras' if selected_model_path.endswith('.h5') else 'svm'

    st.session_state.maintitle = "Image Classifier: " + selected_model_path[7:]
    modal = Modal(key="file_uploader", title="Upload Files")

    upload_btn = st.sidebar.button("Upload File")
    live_picture = st.sidebar.button("Camera")

    if live_picture:
        webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            client_settings=ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            ),
            video_transformer_factory=VideoTransformer,
        )

    if upload_btn:
        modal.open()

    if modal.is_open():
        with modal.container():
            uploaded_image = st.file_uploader("Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "bmp"])
            if st.button("Ok"):
                if uploaded_image is not None:
                    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
                    st.session_state.image_data = image  # Store image data in session state
                    modal.close()
                else:
                    modal.close()

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
                        # Display the image with a fixed size
                        st.image(image, caption='Uploaded Image', use_column_width=False, width=400)

                        # Display predicted class and confidence
                        st.write("Predicted Class: ", predicted_class)
                        st.write("Prediction: ", predictions)
                        st.write("Confidence: ", confidence)
                elif st.session_state.model_type == 'svm':
                    with text_container:
                        # Display the image with a fixed size
                        st.image(image, caption='Uploaded Image', use_column_width=False, width=400)

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
