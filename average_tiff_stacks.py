import os
import numpy as np
import tifffile
import re

num_stacks = 3

# Natural sorting key function that converts numeric parts to integers
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', filename)]

# Function to read the first num_stacks TIFF files from a folder with natural sorting
def read_tiff_stack(folder_path, num_stacks):
    tiff_stack = []
    # Apply natural sort on filenames
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".TIFF")], key=natural_sort_key)
    limited_file_list = file_list[:num_stacks]  # Load only the first num_stacks files
    for filename in limited_file_list:
        print(filename)
        img = tifffile.imread(os.path.join(folder_path, filename)).astype(np.uint16)
        tiff_stack.append(img)
    return np.array(tiff_stack)

# Function to compute the average of TIFF stacks
def compute_average(tiff_stacks):
    # Compute the mean and ensure it's in 16-bit range [0, 65535]
    avg_img = np.mean(tiff_stacks, axis=0)
    avg_img = np.clip(avg_img, 0, 65535)  # Clip values to ensure they are within the valid range
    return avg_img.astype(np.uint16)

# Function to write the average TIFF stack to a file
def write_average_tiff(average_image, output_path):
    tifffile.imwrite(output_path, average_image)

# Path to the folder containing TIFF stacks
input_folder = r"Z:\members\Wang\Data\Mouse\Embryo\20230602\LogScale\Mouse_embyo_10hour"

# Output path for the average TIFF stack
output_tiff_path = r"C:\Users\rausc\Documents\EMBL\data\Mouse_embryo_10hour-average\Mouse_embyo_10hour-average-3.TIFF"

# Read the first num_stacks TIFF stacks from the folder
tiff_stacks = read_tiff_stack(input_folder, num_stacks)

# Compute the average of the TIFF stacks
average_image = compute_average(tiff_stacks)

# Write the average TIFF stack to a file
write_average_tiff(average_image, output_tiff_path)

print("Average TIFF stack created successfully.")
