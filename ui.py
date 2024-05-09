import streamlit as st
from streamlit_modal import Modal
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import tensorflow as tf

def introduction():
    st.write("This is a waste classifier. It classifies wastes in three types:")
    st.write("1. Recyclable")
    st.write("2. Non Recyclable")
    st.write("3. Compostable")

    if st.button("Next"):
        st.session_state.page = "model"

    return None

def model():
    text_container = st.container()

    modal = Modal(key="file_uploader", title="Upload Files")

    upload_btn = st.sidebar.button("Upload File")

    if upload_btn:
        modal.open()

    if modal.is_open():
        with modal.container():
            uploaded_image = st.file_uploader("Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "bmp"])
            if st.button("Ok"):
                if uploaded_image is not None:
                    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
                    st.session_state.image_data = image  # Store image data in session state
                    modal.close()
                else:
                    modal.close()

    if "image_data" in st.session_state:
        image = st.session_state.image_data
               
        
        if image is None:
            text_container.error("Error: Failed to load the image.")
        else:
            resize = tf.image.resize(image, (256, 256))
            if resize is None:
                text_container.error("Error: Failed to resize the image.")
            else:
                with st.spinner("Loading Response..."):
                    # Loading the Model and perform the prediction
                    model = load_model(os.path.join('models', 'deviceClassifier.h5')) 
                    predictions = model.predict(np.expand_dims(resize/255, 0))

                    # Interpret predictions
                    class_names = ['cellphone', 'desktop', 'laptop']
                    predicted_class_index = np.argmax(predictions)
                    predicted_class = class_names[predicted_class_index]
                    confidence = predictions[0][predicted_class_index]

                    with text_container:
                        # Display the image
                        st.image(image, caption='Uploaded Image', use_column_width=True)

                        # Display predicted class and confidence
                        st.write("Predicted Class: ", predicted_class)
                        st.write("Confidence: ", confidence)

    return None


if __name__ == "__main__":
    # Display static title
    st.title("Waste Recognition")
    
    # Main logic to switch between pages
    if "page" not in st.session_state:
        st.session_state.page = "introduction"

    if st.session_state.page == "introduction":
        introduction()

    elif st.session_state.page == "model":
        model()

#TODO Centering the Upload File button
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
