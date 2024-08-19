import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def normalize_slice(slice_data):
    slice_data = slice_data.astype(float)
    slice_data -= slice_data.min()  # Subtract the minimum value
    slice_data /= slice_data.max()  # Divide by the maximum value to scale to 0-1
    return slice_data

def plot_slice(slice_data, show_axis=True):
    fig, ax = plt.subplots(figsize=(6, 6))  # Use subplots for better control
    im = ax.imshow(slice_data, cmap='gray', interpolation='nearest')
    
    if show_axis:
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
    else:
        ax.axis('off')

    # Create a colorbar with customized size and location
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.6, aspect=10)  # Adjust these parameters
    return fig

def save_plot(fig, output_path):
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Plot saved to {output_path}")

def plot_and_save_slice_from_tiff(stack_path, slice_index, show_axis=True):
    stack = read_tiff_stack(stack_path)
    
    if slice_index < 0 or slice_index >= stack.shape[0]:
        print(f"Slice index {slice_index} is out of range for the given stack. Max index: {stack.shape[0]-1}")
        return

    stack_dir = os.path.dirname(stack_path)
    stack_base = os.path.basename(stack_path)
    stack_name, _ = os.path.splitext(stack_base)
    
    output_path = os.path.join(stack_dir, f"{stack_name}_slice_{slice_index}.png")

    slice_data = normalize_slice(stack[slice_index])
    fig = plot_slice(slice_data, show_axis)
    save_plot(fig, output_path)

if __name__ == "__main__":
    stack_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-single\2D-N2N-single_volume_output_stack-mouse-project-test_1_mouse_model_nameUNet3_UNet_base8_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch28540_slice_250.png"
    slice_index = 250  # Specify the desired slice index here
    show_axis = False  # Set to False to hide the axis
    plot_and_save_slice_from_tiff(stack_path, slice_index, show_axis)

