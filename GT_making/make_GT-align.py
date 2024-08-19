import os
import numpy as np
import tifffile
import re
from skimage import transform
from skimage.registration import phase_cross_correlation

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

# Function to align a stack of TIFF images using the first image as the reference
def align_tiff_stack(tiff_stack):
    reference_image = tiff_stack[0]
    aligned_stack = [reference_image]  # Start with the reference image
    
    for i in range(1, len(tiff_stack)):
        shift, error, diffphase = phase_cross_correlation(reference_image, tiff_stack[i], upsample_factor=10)
        print(f"Image {i} shift: {shift}")
        aligned_image = transform.warp(tiff_stack[i], transform.AffineTransform(translation=-shift))
        aligned_stack.append(aligned_image.astype(np.uint16))
    
    return np.array(aligned_stack)

# Function to compute the average of TIFF stacks
def compute_average(tiff_stacks):
    # Compute the mean and ensure it's in 16-bit range [0, 65535]
    avg_img = np.mean(tiff_stacks, axis=0)
    avg_img = np.clip(avg_img, 0, 65535)  # Clip values to ensure they are within the valid range
    return avg_img.astype(np.uint16)

# Function to write the average TIFF stack to a file
def write_average_tiff(average_image, output_path):
    tifffile.imwrite(output_path, average_image)

# Function to construct the output path based on the input folder and number of stacks
def construct_output_path(output_folder, input_folder, num_stacks):
    folder_name = os.path.basename(os.path.normpath(input_folder))
    output_filename = f"{folder_name}-average-{num_stacks}.TIFF"
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, output_filename)

if __name__ == '__main__':
    # Path to the folder containing TIFF stacks
    input_folder = r"C:\Users\rausc\Documents\EMBL\data\big_data\MouseEmbryo20230602LogScaleMouse_embyo_10hour"
    
    # Number of stacks to read and average
    num_stacks = 10

    # Path to the output folder
    output_folder = r"C:\Users\rausc\Documents\EMBL\data\test"

    # Construct the output path based on the specified output folder, input folder, and number of stacks
    output_tiff_path = construct_output_path(output_folder, input_folder, num_stacks)

    # Read the first num_stacks TIFF stacks from the folder
    tiff_stacks = read_tiff_stack(input_folder, num_stacks)

    # Align the TIFF stacks
    aligned_stacks = align_tiff_stack(tiff_stacks)

    # Compute the average of the aligned TIFF stacks
    average_image = compute_average(aligned_stacks)

    # Write the average TIFF stack to a file
    write_average_tiff(average_image, output_tiff_path)

    print("Average TIFF stack created successfully.")
    print(f"Output saved to: {output_tiff_path}")
