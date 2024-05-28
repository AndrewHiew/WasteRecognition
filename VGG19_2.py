# steps to image prediction modelling 

# 1. pre-preprocessing - preparing images
# 2. feature extraction - extract relevent features from the pre-processed images (input as classifier)
# 3. classification - machine learning model for prediction 


import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
import imghdr
import os
import cv2
from PIL import Image

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
data_dir= 'data'
image_extensions = ['jpeg','jpg','bmp','png']
#exclude mac DS_STRORE 
files = [f for f in os.listdir(data_dir) if f != '.DS_Store']
#print(files)



#1. Load Data

data = tf.keras.utils.image_dataset_from_directory('data')
#images represented as numpy array
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
batch[0].min()

#1.2 array to load all images
for image_class in files :
    for image in [f for f in os.listdir(os.path.join(data_dir,image_class)) if f != '.DS_Store']:
        #print(image)
        image_path = os.path.join(data_dir,image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip=imghdr.what(image_path)
            if tip  not in image_extensions:
                print("Image is not in the list of extensions{}".format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("Error with image{}".format(image_path))
            os.remove(image_path)

#keras built-in dataset
image_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    batch_size=32,  # Adjust batch size as needed
    
    image_size=(224, 224),  # Set the target image size
    validation_split=0.2,  # Split data into training and validation sets
    subset="training",  # Specify if it's the training or validation set
    seed=123  # Set seed for reproducibility
)

#validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    batch_size=32,  # Adjust batch size as needed
    image_size=(224, 224),  # Set the target image size
    validation_split=0.2,  # Split data into training and validation sets
    subset="validation",  # Specify if it's the training or validation set
    seed=123  # Set seed for reproducibility
)

class_names = image_dataset.class_names
num_classes = len(class_names)


# #2.Process Data 
train_dataset = image_dataset.map(lambda x, y: (preprocess_input(x), y))
val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y))

#3. VGG19 Model 

#load the VGG19 Model 
vgg_model= VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
vgg_model.trainable = False

# sequential model
model = Sequential([
    vgg_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#5. Train Model 
history = model.fit(train_dataset, validat
                    ion_data=val_dataset, epochs=10)  
test_loss, test_acc = model.evaluate(val_dataset)
print("Test Accuracy:", test_acc)

#6. Graph 
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#7. Prediction 

model.save('vgg19_model.h5')
print("model saved: vgg19_model.h5")
