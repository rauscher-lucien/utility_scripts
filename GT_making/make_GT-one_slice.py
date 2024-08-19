import os
import numpy as np
import tifffile
import re

# Natural sorting key function that converts numeric parts to integers
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', filename)]

# Function to read the first num_stacks TIFF files from a folder with natural sorting
def read_tiff_stack(folder_path, num_stacks, slice_index):
    tiff_slices = []
    # Apply natural sort on filenames
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".TIFF")], key=natural_sort_key)
    limited_file_list = file_list[:num_stacks]  # Load only the first num_stacks files
    for filename in limited_file_list:
        print(f"Reading {filename}, slice {slice_index}")
        img = tifffile.imread(os.path.join(folder_path, filename)).astype(np.uint16)
        if img.ndim > 2:
            img_slice = img[slice_index]  # Extract the specific slice
        else:
            raise ValueError(f"The image {filename} does not have enough dimensions to extract a slice.")
        tiff_slices.append(img_slice)
    return np.array(tiff_slices)

# Function to compute the average of TIFF slices
def compute_average(tiff_slices):
    # Compute the mean and ensure it's in 16-bit range [0, 65535]
    avg_img = np.mean(tiff_slices, axis=0)
    avg_img = np.clip(avg_img, 0, 65535)  # Clip values to ensure they are within the valid range
    return avg_img.astype(np.uint16)

# Function to write the average TIFF slice to a file
def write_average_tiff(average_image, output_path):
    tifffile.imwrite(output_path, average_image)

# Function to construct the output path based on the input folder, number of stacks, and slice index
def construct_output_path(output_folder, input_folder, num_stacks, slice_index):
    folder_name = os.path.basename(os.path.normpath(input_folder))
    output_filename = f"{folder_name}-average-{num_stacks}-slice-{slice_index}.TIFF"
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, output_filename)

if __name__ == '__main__':
    # Path to the folder containing TIFF stacks
    input_folder = r"C:\Users\rausc\Documents\EMBL\data\big_data\Drosophila20210316LogScale01L_Good_Sample_02_t_"
    
    # Number of stacks to read and average
    num_stacks = 100

    # Path to the output folder
    output_folder = r"C:\Users\rausc\Documents\EMBL\data\droso-results"

    # Slice index to average
    slice_index = 90

    # Construct the output path based on the specified output folder, input folder, number of stacks, and slice index
    output_tiff_path = construct_output_path(output_folder, input_folder, num_stacks, slice_index)

    # Read the first num_stacks TIFF stacks from the folder and extract the specified slice
    tiff_slices = read_tiff_stack(input_folder, num_stacks, slice_index)

    # Compute the average of the TIFF slices
    average_image = compute_average(tiff_slices)

    # Write the average TIFF slice to a file
    write_average_tiff(average_image, output_tiff_path)

    print("Average TIFF slice created successfully.")
    print(f"Output saved to: {output_tiff_path}")
