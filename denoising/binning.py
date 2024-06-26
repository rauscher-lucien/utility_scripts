import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def adjust_dimensions(slice, bin_size):
    # Adjust dimensions to be divisible by bin_size
    new_height = (slice.shape[0] // bin_size) * bin_size
    new_width = (slice.shape[1] // bin_size) * bin_size
    return slice[:new_height, :new_width]

def bin_slice(slice, bin_size):
    # Make sure dimensions are adjusted to be divisible by bin_size
    slice = adjust_dimensions(slice, bin_size)
    # Reduce the resolution of the slice by binning
    reshaped = slice.reshape((slice.shape[0] // bin_size, bin_size,
                              slice.shape[1] // bin_size, bin_size))
    binned_slice = reshaped.mean(axis=(1, 3))
    return binned_slice

def bin_3d_image_stack(image_stack, bin_size, test_index=None):
    # Test a specific slice before processing the entire stack
    if test_index is not None:
        if test_index < 0 or test_index >= image_stack.shape[0]:
            raise IndexError(f"Test index {test_index} out of range.")

        # Apply binning to the test slice
        test_slice = image_stack[test_index]
        binned_test_slice = bin_slice(test_slice, bin_size)

        # Display the test slice result
        plt.imshow(binned_test_slice, cmap='gray')
        plt.title(f"Binned Test Slice {test_index}")
        plt.colorbar()
        plt.show()

        # Ask user for confirmation to proceed
        proceed = input("Proceed with binning the entire volume? (yes/no): ")
        if proceed.lower() != 'yes':
            print("Binning stopped by the user.")
            return None

    # Initialize an empty array for the binned stack
    adjusted_slice = adjust_dimensions(image_stack[0], bin_size)
    binned_stack_shape = (
        image_stack.shape[0],
        adjusted_slice.shape[0] // bin_size,
        adjusted_slice.shape[1] // bin_size
    )
    binned_stack = np.zeros(binned_stack_shape, dtype=image_stack.dtype)

    # Apply binning slice by slice
    for i, slice in enumerate(image_stack):
        print(f"Processing slice {i}")
        binned_stack[i, :, :] = bin_slice(slice, bin_size)

    return binned_stack

def save_binned_stack(binned_stack, input_path, bin_size):
    if binned_stack is None:
        return

    # Generate the output filename based on the input filename and bin size
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_binned_{bin_size}x{bin_size}{ext}"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    # Save the binned 3D stack
    tifffile.imwrite(output_path, binned_stack)
    print(f"Binned image stack saved at {output_path}")

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF"
    bin_size = 4  # Bin size (e.g., 2x2)
    test_index = 250  # Example index to test a specific slice before processing the entire volume

    # Step 1: Read the 3D image stack
    image_stack = tifffile.imread(data_path)

    # Step 2: Apply binning to the image stack with testing
    binned_stack = bin_3d_image_stack(image_stack, bin_size, test_index=test_index)

    # Step 3: Save the binned 3D stack
    save_binned_stack(binned_stack, data_path, bin_size)
