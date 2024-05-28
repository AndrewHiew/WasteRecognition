from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# Load the model
model = load_model(os.path.join('models', 'demoModel.h5'))

# Load and preprocess the image
image = cv2.imread('test.jpg')
if image is None:
    print("Error: Failed to load the image.")
else:
    print("Image shape:", image.shape)
    resize = tf.image.resize(image, (256, 256))
    if resize is None:
        print("Error: Failed to resize the image.")
    else:
        predictions = model.predict(np.expand_dims(resize/255, 0))
        print(predictions)

        class_names = ['cellphone', 'desktop', 'laptop']
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index]

        print("Predicted Class:", predicted_class)
        print("Confidence:", confidence)
