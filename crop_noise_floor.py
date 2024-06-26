import tifffile
import numpy as np

def reduce_noise_floor(tiff_path, output_path, noise_floor_threshold=0.35):
    # Load the TIFF file
    stack = tifffile.imread(tiff_path)

    # Convert the stack to float if it's not already to avoid data type overflow
    stack = stack.astype(np.float32)
    
    # Apply the threshold
    # Set values below the noise floor threshold to zero
    stack[stack <= noise_floor_threshold] = 0
    
    # Save the modified TIFF stack
    tifffile.imwrite(output_path, stack, photometric='minisblack')

    print(f"Modified TIFF stack saved to {output_path}")

# Example usage
tiff_path = r"C:\Users\rausc\Documents\EMBL\data\Nema_B-analysis\rescaled_nema_avg_40.TIFF"  # Path to the original TIFF file
output_path = r"C:\Users\rausc\Documents\EMBL\data\Nema_B-analysis\rescaled_nema_avg_40-reduced_background.TIFF"  # Path where the modified TIFF file will be saved
reduce_noise_floor(tiff_path, output_path)
