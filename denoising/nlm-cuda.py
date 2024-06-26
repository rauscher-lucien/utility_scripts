import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os

def cuda_nlm_filter_3d_image_stack(image_stack, h, search_window, block_size, test_index=None):
    # Initialize CUDA device
    cuda = cv2.cuda

    # Test a specific slice before processing the entire stack
    if test_index is not None:
        if test_index < 0 or test_index >= image_stack.shape[0]:
            raise IndexError(f"Test index {test_index} out of range.")
        
        # Convert to float32, assuming the original data range is 0-65535 for 16-bit
        test_slice = image_stack[test_index].astype(np.float32) / 65535.0
        
        # Upload the slice to GPU
        gpu_test_slice = cuda.GpuMat()
        gpu_test_slice.upload(test_slice)

        # Apply CUDA NLM filter to the test slice
        filtered_gpu_test_slice = cuda.fastNlMeansDenoising(gpu_test_slice, h, search_window, block_size)
        
        # Download the filtered slice from GPU
        filtered_test_slice = filtered_gpu_test_slice.download()
        
        # Display the test slice result
        plt.imshow(filtered_test_slice, cmap='gray')
        plt.title(f"Filtered Test Slice {test_index}")
        plt.colorbar()
        plt.show()

        # Ask user for confirmation to proceed
        proceed = input("Proceed with filtering the entire volume? (yes/no): ")
        if proceed.lower() != 'yes':
            print("Filtering stopped by the user.")
            return None

    # Initialize an empty array for the filtered stack, making sure to work with float32 for processing
    filtered_stack = np.zeros_like(image_stack, dtype=np.float32)

    # Apply CUDA NLM filtering slice by slice
    for i, slice in enumerate(image_stack):
        print(f"Processing slice {i}")
        # Convert to float32, assuming the original data range is 0-65535 for 16-bit
        slice_float = slice.astype(np.float32) / 65535.0
        
        # Upload the slice to GPU
        gpu_slice = cuda.GpuMat()
        gpu_slice.upload(slice_float)
        
        # Apply CUDA NLM filter to the slice
        filtered_gpu_slice = cuda.fastNlMeansDenoising(gpu_slice, h, search_window, block_size)
        
        # Download the filtered slice from GPU
        filtered_slice = filtered_gpu_slice.download()
        filtered_stack[i, :, :] = filtered_slice

    # Convert the result back to 16-bit if necessary, ensuring the data is properly scaled back to the 0-65535 range
    filtered_stack_16bit = np.clip(filtered_stack * 65535, 0, 65535).astype('uint16')
    return filtered_stack_16bit

def save_filtered_stack(filtered_stack, input_path):
    if filtered_stack is None:
        return
    
    # Generate the output filename based on the input filename
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_filtered_nlm_cuda{ext}"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    # Save the filtered 3D stack
    tifffile.imwrite(output_path, filtered_stack)
    print(f"Filtered image stack saved at {output_path}")

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF"
    h = 10  # Filtering strength
    search_window = 21  # Size of the search window
    block_size = 7  # Size of the block to compare
    test_index = 250  # Example index to test a specific slice before processing the entire volume

    # Step 1: Read the 3D image stack
    image_stack = tifffile.imread(data_path)

    # Step 2: Apply CUDA NLM filtering to the image stack with testing
    filtered_stack = cuda_nlm_filter_3d_image_stack(image_stack, h, search_window, block_size, test_index=test_index)

    # Step 3: Save the filtered 3D stack
    save_filtered_stack(filtered_stack, data_path)

