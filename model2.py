import cv2
import imghdr
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, SparseCategoricalAccuracy

gpus = tf.config.experimental.list_logical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'datasets'

# Assuming these are your class labels
class_names = sorted(os.listdir(data_dir))

image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print("Image not in ext list{}".format(image_path))
                os.remove(image_path)

        except Exception as e:
            print("Issue with image {}".format(image_path))
            os.remove(image_path)

data = tf.keras.utils.image_dataset_from_directory('datasets')

# Data Preprocess
scaled_data = data.map(lambda x,y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

# Split Data
train_size = 7
val_size = 2
test_size = 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# Build Deep Learning Model
model = Sequential()

model.add(Conv2D(16, (3,3), 1 , activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1 , activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1 , activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))  # Change here

model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])  # Change here
print(model.summary())


# Train the Model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Evaluate Performance
pre = Precision()
recall = Recall()
acc = SparseCategoricalAccuracy()  # Change here


print(f'Precision:{pre.result().numpy()}, Recall:{recall.result().numpy()}, Accuracy:{acc.result().numpy()}')
model.save(os.path.join('models', 'demoModel.h5'))
