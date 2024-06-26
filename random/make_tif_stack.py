import os
import numpy as np
import imageio
import tifffile

# Function to read image files from a folder
def read_images_from_folder(folder_path):
    images = []
    file_list = sorted(os.listdir(folder_path))
    for filename in file_list:
        if filename.endswith(".png"):
            img = imageio.imread(os.path.join(folder_path, filename))
            images.append(img)
    return images

# Function to recombine images into a TIFF stack
def create_tiff_stack(images, output_path):
    tifffile.imwrite(output_path, np.stack(images, axis=0))

# Path to the folder containing image files
input_folder = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'test_data_3')

# Output path for the recombined TIFF stack
output_tiff_path = os.path.join(input_folder, 'tiff_stack.TIFF')

# Read images from the folder
image_list = read_images_from_folder(input_folder)

# Recombine images into a TIFF stack
create_tiff_stack(image_list, output_tiff_path)

print("TIFF stack created successfully.")
