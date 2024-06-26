import os
import numpy as np
import tifffile

def offset_slices_x_direction(input_folder, output_folder, ofn, pixel_offset_x):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each file in the input directory
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, ofn)
            
            # Read the TIFF stack
            stack = tifffile.imread(input_path)
            
            if stack.ndim == 3:  # Check if it's a 3D stack
                # Offset every second slice in the x-direction
                for i in range(1, stack.shape[0], 2):  # Start at index 1 and step by 2
                    stack[i] = np.roll(stack[i], pixel_offset_x, axis=1)  # Roll along x-axis
            
                # Save the modified stack
                tifffile.imwrite(output_path, stack, photometric='minisblack')
                print(f"Processed and saved modified stack: {output_path}")
            else:
                print(f"Skipping {filename}, as it does not appear to be a 3D stack.")

# Example usage
input_folder = r"C:\Users\rausc\Documents\EMBL\data\droso_good_avg"
output_folder = r"C:\Users\rausc\Documents\EMBL\data\droso_good_avg_offset"
offset_filename = 'droso_good_avg_40-offset-2.TIFF'
pixel_offset_x = -2  # Offset every second slice by 10 pixels in the x direction
offset_slices_x_direction(input_folder, output_folder, offset_filename, pixel_offset_x)
