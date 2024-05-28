import cv2
from skimage.feature import hog
from joblib import load

# Define the file path where the model is saved
model_file_path = "models/svm_model.joblib"

# Load the saved SVM classifier model
svm_classifier = load(model_file_path)

# Load the new image
image_path = "test7.jpg"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize or pad the image to ensure it has the same dimensions as the training images
target_size = (256, 256)
resized_image = cv2.resize(gray_image, target_size)

# Extract HOG features from the resized image
hog_features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

# Make predictions using the loaded SVM classifier
predicted_class_index = svm_classifier.predict([hog_features])[0]

# Get the predicted class name from the index
class_names = ['cellphone', 'desktop', 'laptop']
predicted_class = class_names[predicted_class_index]

print("Predicted class:", predicted_class)
