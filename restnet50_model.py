import cv2
import imghdr
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import Precision, Recall, SparseCategoricalAccuracy

# Enable memory growth for GPUs
gpus = tf.config.experimental.list_logical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'waste_datasets'

"""
Data Preparation
"""

# Get class names
class_names = sorted(os.listdir(data_dir))

# Ensure image extensions are correct
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print(f"Image not in ext list: {image_path}")
                os.remove(image_path)
        except Exception as e:
            print(f"Issue with image {image_path}: {e}")
            os.remove(image_path)

# Load dataset
data = tf.keras.utils.image_dataset_from_directory(data_dir, label_mode='int', image_size=(256, 256))

# Data Preprocessing
scaled_data = data.map(lambda x, y: (x / 255.0, y))

# Visualize some images and labels
scaled_iterator = scaled_data.as_numpy_iterator()
batch = scaled_iterator.next()
plt.figure(figsize=(20, 20))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(batch[0][i])
    plt.title(class_names[batch[1][i]])
    plt.axis("off")
plt.show()

# Convert dataset to numpy arrays for stratified splitting
images, labels = [], []
for img, lbl in scaled_data:
    images.append(img.numpy())
    labels.append(lbl.numpy())
images = np.concatenate(images, axis=0)
labels = np.concatenate(labels, axis=0)

# Stratified split
train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.3, stratify=labels)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.33, stratify=temp_labels)

# Convert back to tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# Verify dataset sizes
print(f"Train set size: {len(train_ds)}")
print(f"Validation set size: {len(val_ds)}")
print(f"Test set size: {len(test_ds)}")

print("Press enter to continue")
input()

"""
Model Training
"""

# Building the Model using ResNet50
base_model = ResNet50(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Add custom layers on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train the Model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train_ds, epochs=40, validation_data=val_ds, callbacks=[tensorboard_callback])

# Get training history
training_loss = hist.history['loss']
training_accuracy = hist.history['accuracy']
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

"""
Model Evaluation
"""

# Define metrics for each class
class_metrics = {class_name: {'precision': Precision(), 'recall': Recall(), 'accuracy': SparseCategoricalAccuracy()} for class_name in class_names}

# Evaluate performance for each batch in the test dataset
for images, labels in test_ds:
    predictions = model.predict(images)
    
    # Convert predictions to labels by taking the argmax along the class axis
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Update metrics for each class
    for class_index, class_name in enumerate(class_names):
        class_labels = (labels == class_index)
        class_preds = (predicted_labels == class_index)
        class_metrics[class_name]['precision'].update_state(class_labels, class_preds)
        class_metrics[class_name]['recall'].update_state(class_labels, class_preds)
        class_metrics[class_name]['accuracy'].update_state(class_labels, predictions)

# Print metrics for each class
for class_name, metrics in class_metrics.items():
    precision = metrics['precision'].result().numpy()
    recall = metrics['recall'].result().numpy()
    accuracy = metrics['accuracy'].result().numpy()
    print(f'Class: {class_name} - Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}')

model.save(os.path.join('models', 'demoModel3.h5'))