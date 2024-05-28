import os

def rename_images(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter only image files
    image_files = [file for file in files if file.endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]
    
    # Sort the image files
    image_files.sort()
    
    # Rename images sequentially from images1 to images999
    total_images = len(image_files)
    num_digits = len(str(total_images))
    for index, old_name in enumerate(image_files, start=1):
        new_name = f'images{index:0{num_digits}}{os.path.splitext(old_name)[1]}'
        new_path = os.path.join(directory, new_name)
        os.rename(os.path.join(directory, old_name), new_path)
        print(f'Renamed {old_name} to {new_name}')
    
    # Rename images again from image1 to image999
    for index, old_name in enumerate(image_files, start=1):
        new_name = f'image{index:0{num_digits}}{os.path.splitext(old_name)[1]}'
        new_path = os.path.join(directory, new_name)
        os.rename(os.path.join(directory, old_name), new_path)
        print(f'Renamed {old_name} to {new_name}')

if __name__ == "__main__":
    directory = input("Enter the directory containing the images: ")
    rename_images(directory)
