import cv2
import numpy as np
import os
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from skimage import exposure

# Data preparation
data_dir = 'waste_datasets'
class_names = sorted(os.listdir(data_dir))

X = []
y = []

# Define the target size for resizing images
target_size = (256, 256)

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize or pad the image to ensure all images have the same dimensions
        resized_image = cv2.resize(image, target_size)
        
        # Extract HOG features from the resized image
        hog_features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        
        # Ensure all HOG features have the same length by padding if necessary
        desired_length = 404
        current_length = len(hog_features)
        if current_length < desired_length:
            pad_width = desired_length - current_length
            hog_features = np.pad(hog_features, (0, pad_width), 'constant', constant_values=(0))  # Pad features
        
        X.append(hog_features)
        y.append(class_names.index(class_name))

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Evaluate classifier
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Overall Accuracy:", accuracy)

# Print classification report for detailed accuracy of each class
print(classification_report(y_test, y_pred, target_names=class_names))

# Saving the model
# Define the file path where you want to save the model
model_file_path = "models/svm_waste_model.joblib"

# Save the trained SVM classifier model
dump(svm_classifier, model_file_path)
print("Model saved successfully at:", model_file_path)
