import tifffile as tiff
from sklearn.preprocessing import PowerTransformer
import numpy as np
import os

def power_transform_stack(input_path, output_path, method='yeo-johnson'):
    """
    Reads a TIFF stack, applies a Yeo-Johnson power transformation, and saves the result.

    Args:
        input_path (str): Path to the input TIFF stack.
        output_path (str): Path to save the transformed TIFF stack.
        method (str): Method of power transformation ('yeo-johnson' or 'box-cox').
    """
    # Load the TIFF stack
    with tiff.TiffFile(input_path) as tif:
        images = tif.asarray()

    # Check if the data is in an appropriate format (expecting 3D array: slices, height, width)
    if images.ndim != 3:
        raise ValueError("Input TIFF must be a 3D stack (slices, height, width).")

    # Initialize the power transformer
    transformer = PowerTransformer(method=method, standardize=True)

    # Prepare images for transformation (reshape to 2D)
    num_images, height, width = images.shape
    images_reshaped = images.reshape(num_images, height * width).T

    # Apply the power transformation
    images_transformed = transformer.fit_transform(images_reshaped)
    images_transformed = images_transformed.T.reshape(num_images, height, width)

    # Save the transformed images to a new TIFF file
    with tiff.TiffWriter(output_path, bigtiff=True) as tif_writer:
        for i in range(num_images):
            tif_writer.save(images_transformed[i].astype(np.float32))

if __name__ == "__main__":
    # Hardcoded paths for the input and output TIFF files
    input_path = r"C:\Users\rausc\Documents\EMBL\data\Nematostella_B\Nematostella_B_V0.TIFF"
    output_path = r"C:\Users\rausc\Documents\EMBL\data\Nema_B-power_transformed\Nematostella_B_V0-power_transformed.TIFF"
    
    # Call the function with hardcoded paths
    power_transform_stack(input_path, output_path)
