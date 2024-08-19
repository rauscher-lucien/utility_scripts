import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import gaussian

def gaussian_filter_3d_image_stack(image_stack, sigma, test_index=None):
    if test_index is not None:
        if test_index < 0 or test_index >= image_stack.shape[0]:
            raise IndexError(f"Test index {test_index} out of range.")
        
        # Convert to float, assuming the original data range is 0-65535 for 16-bit
        test_slice = image_stack[test_index].astype(np.float64) / 65535.0
        
        # Apply Gaussian filter to the test slice
        filtered_test_slice = gaussian(test_slice, sigma=sigma, mode='reflect')
        
        # Display the test slice result
        plt.imshow(filtered_test_slice, cmap='gray')
        plt.title(f"Filtered Test Slice {test_index}")
        plt.colorbar()
        plt.show()

    # Initialize an empty array for the filtered stack, making sure to work with float64 for processing
    filtered_stack = np.zeros_like(image_stack, dtype=np.float64)

    # Apply Gaussian filtering slice by slice
    for i, slice in enumerate(image_stack):
        print(f"Processing slice {i}")
        # Convert to float, assuming the original data range is 0-65535 for 16-bit
        slice_float = slice.astype(np.float64) / 65535.0
        
        # Apply Gaussian filter to the slice
        filtered_slice = gaussian(slice_float, sigma=sigma, mode='reflect')
        filtered_stack[i, :, :] = filtered_slice

    # Convert the result back to 16-bit if necessary, ensuring the data is properly scaled back to the 0-65535 range
    filtered_stack_16bit = np.clip(filtered_stack * 65535, 0, 65535).astype('uint16')
    return filtered_stack_16bit

def save_filtered_stack(filtered_stack, input_path, sigma):
    if filtered_stack is None:
        return
    
    # Generate the output filename based on the input filename
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_filtered_gaussian_sigma_{sigma}{ext}"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    # Save the filtered 3D stack
    tifffile.imwrite(output_path, filtered_stack)
    print(f"Filtered image stack saved at {output_path}")

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF"
    sigma = 2.0  # Standard deviation for Gaussian kernel
    test_index = 100  # Example index to test a specific slice before processing the entire volume

    # Step 1: Read the 3D image stack
    image_stack = tifffile.imread(data_path)

    # Step 2: Apply Gaussian filtering to the image stack with testing
    filtered_stack = gaussian_filter_3d_image_stack(image_stack, sigma, test_index=test_index)

    # Step 3: Save the filtered 3D stack
    save_filtered_stack(filtered_stack, data_path, sigma)


