import os
import numpy as np
import tifffile

def offset_slices_x_direction(input_file, output_folder, pixel_offset_x):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the input file name and extension
    input_filename = os.path.basename(input_file)
    input_name, input_ext = os.path.splitext(input_filename)
    
    # Construct the output file name
    offset_filename = f"{input_name}-offset-{pixel_offset_x}{input_ext}"
    output_path = os.path.join(output_folder, offset_filename)
    
    # Read the TIFF stack
    stack = tifffile.imread(input_file)
    
    if stack.ndim == 3:  # Check if it's a 3D stack
        # Offset every second slice in the x-direction
        for i in range(1, stack.shape[0], 2):  # Start at index 1 and step by 2
            stack[i] = np.roll(stack[i], pixel_offset_x, axis=1)  # Roll along x-axis
        
        # Save the modified stack
        tifffile.imwrite(output_path, stack, photometric='minisblack')
        print(f"Processed and saved modified stack: {output_path}")
    else:
        print(f"Skipping {input_filename}, as it does not appear to be a 3D stack.")

# Example usage
input_file = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Drosophila20210316LogScale01L_Good_Sample_02_t_-average-100.TIFF"
output_folder = r"C:\Users\rausc\Documents\EMBL\data\droso-results"
pixel_offset_x = -2  # Offset every second slice by -2 pixels in the x direction
offset_slices_x_direction(input_file, output_folder, pixel_offset_x)

