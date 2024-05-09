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

"""
Data Prepration
"""

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

""" 
Data Preprocess
"""
scaled_data = data.map(lambda x,y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

# Display the first four images from the batch (For Checking)
# plt.figure(figsize=(20, 20))
# for i in range(4):
#     plt.subplot(1, 4, i + 1)
#     plt.imshow(batch[0][i].astype('uint8'))  # Convert to uint8 for proper display
#     plt.title(class_names[batch[1][i]])
#     plt.axis("off")
# plt.show()

# Split Data
train_size = 7
val_size = 2
test_size = 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# Build Deep Learning Model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# Train the Model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

import matplotlib.pyplot as plt

# Get training history
training_loss = hist.history['loss']
training_accuracy = hist.history['accuracy']

# Get validation history
validation_loss = hist.history['val_loss']
validation_accuracy = hist.history['val_accuracy']

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# Evaluate Performance
pre = Precision()
recall = Recall()
acc = SparseCategoricalAccuracy()

# Initialize variables to store cumulative values
total_precision = 0
total_recall = 0
total_accuracy = 0
num_batches = 0

# Evaluate performance for each batch in the test dataset
for images, labels in test:
    predictions = model.predict(images)
    
    # Convert predictions to labels by taking the argmax along the class axis
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Update metrics with true labels and predicted labels
    pre.update_state(labels, predicted_labels)
    recall.update_state(labels, predicted_labels)
    acc.update_state(labels, predictions)
    
    total_precision += pre.result().numpy()
    total_recall += recall.result().numpy()
    total_accuracy += acc.result().numpy()
    num_batches += 1

# Compute average precision, recall, and accuracy across all batches
average_precision = total_precision / num_batches
average_recall = total_recall / num_batches
average_accuracy = total_accuracy / num_batches

print(f'Average Precision: {average_precision}')
print(f'Average Recall: {average_recall}')
print(f'Average Accuracy: {average_accuracy}')

model.save(os.path.join('models', 'demoModel.h5'))