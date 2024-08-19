import os
import numpy as np
import tifffile
import re
from skimage import transform
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte, img_as_uint

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
def align_slices(tiff_stack, slice_index):
    reference_slice = tiff_stack[0][slice_index]
    aligned_slices = [reference_slice]  # Start with the reference slice
    
    for i in range(1, len(tiff_stack)):
        slice_to_align = tiff_stack[i][slice_index]
        shift, error, diffphase = phase_cross_correlation(reference_slice, slice_to_align, upsample_factor=10)
        print(f"Image {i} shift: {shift}")
        aligned_slice = transform.warp(slice_to_align, transform.AffineTransform(translation=-shift), mode='edge')
        aligned_slice = img_as_uint(aligned_slice)  # Convert to uint16 after warping
        aligned_slices.append(aligned_slice)
    
    return np.array(aligned_slices)

# Function to compute the average of slices
def compute_average(slices):
    # Compute the mean and ensure it's in 16-bit range [0, 65535]
    avg_img = np.mean(slices, axis=0)
    avg_img = np.clip(avg_img, 0, 65535)  # Clip values to ensure they are within the valid range
    return avg_img.astype(np.uint16)

# Function to plot the original and aligned slices successively
def plot_slices(original_slices, aligned_slices, slice_index):
    for i in range(len(original_slices)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_slices[i][slice_index], cmap='gray')
        axes[0].set_title(f'Original Slice {i}')
        axes[1].imshow(aligned_slices[i], cmap='gray')
        axes[1].set_title(f'Aligned Slice {i}')
        plt.tight_layout()
        plt.show()
        plt.pause(1)  # Pause for 1 second between plots
        plt.close(fig)  # Close the figure to move to the next one

if __name__ == '__main__':
    # Path to the folder containing TIFF stacks
    input_folder = r"C:\Users\rausc\Documents\EMBL\data\big_data\MouseEmbryo20230602LogScaleMouse_embyo_10hour"
    
    # Number of stacks to read
    num_stacks = 50

    # Slice index to process
    slice_index = 250  # Change this to the desired slice index

    # Path to the output folder
    output_folder = r"C:\Users\rausc\Documents\EMBL\data\test"

    # Read the first num_stacks TIFF stacks from the folder
    tiff_stacks = read_tiff_stack(input_folder, num_stacks)

    # Align the slices
    aligned_slices = align_slices(tiff_stacks, slice_index)

    # Compute the average of the aligned slices
    average_image = compute_average(aligned_slices)

    # Plot the original and aligned slices successively
    plot_slices(tiff_stacks, aligned_slices, slice_index)

    # Construct the output path based on the specified output folder, input folder, and number of stacks
    output_filename = f"slice_{slice_index}_average.TIFF"
    os.makedirs(output_folder, exist_ok=True)
    output_tiff_path = os.path.join(output_folder, output_filename)

    # Write the average slice to a file
    tifffile.imwrite(output_tiff_path, average_image)

    print("Average slice created successfully.")
    print(f"Output saved to: {output_tiff_path}")


