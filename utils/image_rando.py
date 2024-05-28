import os
import random
import shutil

# Source directory containing images
source_dir = "keyboard"
destination_dir = "temp2"

num_images = 100

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
selected_images = random.sample(all_images, num_images)

counter = 0
for image in selected_images:
    source_path = os.path.join(source_dir, image)
    destination_path = os.path.join(destination_dir, image)
    shutil.copyfile(source_path, destination_path)
    counter = counter + 1

print(f"{counter} images copied to {destination_dir}")
