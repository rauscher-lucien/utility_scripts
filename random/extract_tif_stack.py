import os
import tifffile
from PIL import Image
import numpy as np

def save_tiff_stack_as_png_normalized(tiff_stack_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read TIFF stack
    with tifffile.TiffFile(tiff_stack_path) as tif:
        # Iterate through all pages (images) in the TIFF stack
        for i, page in enumerate(tif.pages):
            # Convert the image to PIL Image
            image = Image.fromarray(page.asarray())
            # Convert to numpy array and normalize to range 0-1
            image_array = np.array(image)
            image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
            # Generate output file path
            output_path = os.path.join(output_folder, f'image_{i}.png')
            # Save the normalized image as PNG
            image_normalized = Image.fromarray((image_array * 255).astype(np.uint8))
            image_normalized.save(output_path)

# Example usage:
tiff_stack_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\nema_avg_40.TIFF"
output_folder = r"C:\Users\rausc\Documents\EMBL\data\nema_avg_40-single_slices"
save_tiff_stack_as_png_normalized(tiff_stack_path, output_folder)
