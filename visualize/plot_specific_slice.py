import os
import matplotlib.pyplot as plt
import tifffile
import numpy as np

def plot_and_save_slice_from_tiff(stack_path, slice_index):
    # Read the TIFF stack
    stack = tifffile.imread(stack_path)
    
    # Check if the slice index is valid
    if slice_index < 0 or slice_index >= stack.shape[0]:
        print(f"Slice index {slice_index} is out of range for the given stack. Max index: {stack.shape[0]-1}")
        return

    # Get the directory and base name of the input stack path
    stack_dir = os.path.dirname(stack_path)
    stack_base = os.path.basename(stack_path)
    stack_name, _ = os.path.splitext(stack_base)
    
    # Construct the output path
    output_path = os.path.join(stack_dir, f"{stack_name}_slice_{slice_index}.png")

    # Normalize the specific slice to the range 0-1
    slice_data = stack[slice_index].astype(float)
    slice_data -= slice_data.min()  # Subtract the minimum value
    slice_data /= slice_data.max()  # Divide by the maximum value to scale to 0-1

    # Plot the specific slice
    fig, ax = plt.subplots(figsize=(6, 6))  # Use subplots for better control
    im = ax.imshow(slice_data, cmap='gray', interpolation='nearest')
    ax.set_title(f'Slice {slice_index}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Create a colorbar with customized size and location
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.6, aspect=10)  # Adjust these parameters

    # Save the plot to the specified output path
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Slice {slice_index} has been saved to {output_path}")

# Example usage
stack_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_nlm_h1.4_ps4_pd20.TIFF"
slice_index = 80  # Specify the desired slice index here
plot_and_save_slice_from_tiff(stack_path, slice_index)




