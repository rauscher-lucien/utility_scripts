import tifffile as tiff
import numpy as np
import os

def crop_tiff_stack(input_file, output_file):
    # Load the TIFF file
    with tiff.TiffFile(input_file) as tif:
        images = tif.asarray()

    # Check if images is indeed a stack
    if images.ndim != 3:
        raise ValueError("Input file does not contain a stack with depth in the third dimension.")

    # Calculate the current depth
    current_depth = images.shape[0]
    print(f"Current depth: {current_depth}")

    # Determine the number of slices to remove to make the depth divisible by 8
    new_depth = current_depth - (current_depth % 8)
    print(f"New depth (cropped to be divisible by 8): {new_depth}")

    # Crop the stack
    cropped_images = images[:new_depth]

    # Save the new cropped TIFF stack
    tiff.imwrite(output_file, cropped_images, photometric='minisblack')

    print(f"Cropped stack saved to {output_file}")

# Example usage
input_filename = r"C:\Users\rausc\Documents\EMBL\data\big_data\OCT-data-1\average_tiff_stack.TIFF"
output_filename = r"C:\Users\rausc\Documents\EMBL\data\big_data\OCT-data-1\average_tiff_stack.TIFF"
crop_tiff_stack(input_filename, output_filename)

