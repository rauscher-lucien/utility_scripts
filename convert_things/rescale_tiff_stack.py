import os
import numpy as np
import tifffile

def rescale_tiff_stack(input_file, output_file):
    # Load the TIFF stack
    with tifffile.TiffFile(input_file) as tif:
        images = tif.asarray()
    
    # Convert to float to prevent clipping issues during rescale
    images = images.astype(np.float32)
    
    # Rescale the pixel values to be between 0 and 1
    images_rescaled = (images - images.min()) / (images.max() - images.min())

    print(images.max())
    print(images.min())
    
    # Save the rescaled stack
    tifffile.imwrite(output_file, images_rescaled, photometric='minisblack')


input_file = r"C:\Users\rausc\Documents\EMBL\data\nema_avg\nema_avg_40.TIFF"
output_file = r"C:\Users\rausc\Documents\EMBL\data\Nema_B-analysis\rescaled_nema_avg_40.TIFF"

rescale_tiff_stack(input_file, output_file)

print("Rescaling completed!")
