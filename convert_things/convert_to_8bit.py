import os
import numpy as np
import tifffile

def rescale_tiff_stack(input_file, output_file):
    # Load the TIFF stack
    with tifffile.TiffFile(input_file) as tif:
        images = tif.asarray()

    # Convert to float to prevent clipping issues during rescale
    images = images.astype(np.float32)

    # Rescale the pixel values to be between 0 and 255
    images_rescaled = (images - images.min()) / (images.max() - images.min()) * 255

    # Convert to 8-bit unsigned integers
    images_8bit = images_rescaled.astype(np.uint8)

    # Save the rescaled 8-bit stack
    tifffile.imwrite(output_file, images_8bit, photometric='minisblack')

input_file = r"C:\Users\rausc\Documents\EMBL\data\big_data_small\OCT-data-1\Good_Sample_03_t_1.TIFF"
output_file = r"C:\Users\rausc\Documents\EMBL\data\big_data_small\OCT-data-1\bit_stack.TIFF"

rescale_tiff_stack(input_file, output_file)

print("Rescaling completed!")
