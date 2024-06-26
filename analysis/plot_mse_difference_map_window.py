import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def crop_to_multiple_of_16(img_stack):
    h, w = img_stack.shape[1:3]
    new_h = h - (h % 16)
    new_w = w - (w % 16)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img_stack[:, top:top+new_h, left:left+new_w]

def plot_mse_difference_map(ground_truth_path, denoised_image_path, slice_index, species_name, x_range, y_range, save_plot=True):
    # Load the ground truth and denoised images
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_image_stack = read_tiff_stack(denoised_image_path).squeeze()

    # Ensure the stacks have the same number of slices based on the smaller stack
    min_slices = min(ground_truth_stack.shape[0], denoised_image_stack.shape[0])
    ground_truth_stack = ground_truth_stack[:min_slices].squeeze()
    denoised_image_stack = denoised_image_stack[:min_slices].squeeze()

    # Crop stacks to have dimensions that are multiples of 16
    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_image_stack = crop_to_multiple_of_16(denoised_image_stack)

    # Ensure the cropped stacks have the same dimensions
    assert cropped_ground_truth_stack.shape == cropped_denoised_image_stack.shape, "Cropped stacks must have the same dimensions."

    # Ensure the slice index is within the valid range
    if slice_index < 0 or slice_index >= cropped_ground_truth_stack.shape[0]:
        raise IndexError(f"Slice index {slice_index} is out of range. Must be between 0 and {cropped_ground_truth_stack.shape[0] - 1}.")

    # Get the specified slice
    ground_truth_slice = cropped_ground_truth_stack[slice_index].astype(np.float32)
    denoised_image_slice = cropped_denoised_image_stack[slice_index].astype(np.float32)

    # Crop the slices to the specified ranges
    ground_truth_slice_cropped = ground_truth_slice[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    denoised_image_slice_cropped = denoised_image_slice[y_range[0]:y_range[1], x_range[0]:x_range[1]]

    # Compute MSE difference map
    mse_difference_map_3 = np.zeros_like(ground_truth_slice_cropped)
    mse_difference_map = (denoised_image_slice_cropped - ground_truth_slice_cropped) ** 2
    mse_difference_map_2 = denoised_image_slice_cropped - ground_truth_slice_cropped
    for i in range(0, ground_truth_slice_cropped.shape[0]):
        for j in range(0, ground_truth_slice_cropped.shape[1]):
            a = denoised_image_slice_cropped[i, j]
            b = ground_truth_slice_cropped[i, j]
            c = a-b
            mse_difference_map_3[i, j] = c
            print(1)

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the ground truth slice
    im1 = axes[0].imshow(ground_truth_slice_cropped, cmap='gray', vmin=0, vmax=65535)
    axes[0].set_title(f'Ground Truth (Slice {slice_index})')
    axes[0].axis('on')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot the denoised image slice
    im2 = axes[1].imshow(denoised_image_slice_cropped, cmap='gray', vmin=0, vmax=65535)
    axes[1].set_title(f'Denoised Image (Slice {slice_index})')
    axes[1].axis('on')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot the MSE difference map
    im3 = axes[2].imshow(mse_difference_map, cmap='hot', vmin=0, vmax=np.max(mse_difference_map))
    axes[2].set_title(f'MSE Difference Map (Slice {slice_index})')
    axes[2].axis('on')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f'MSE Difference Analysis for {species_name} - Slice {slice_index}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure as one plot
    if save_plot:
        plot_path = os.path.join(os.path.dirname(denoised_image_path), 
                                 f'mse_difference_analysis_{species_name}_slice_{slice_index}_x{x_range[0]}-{x_range[1]}_y{y_range[0]}-{y_range[1]}.png')
        plt.savefig(plot_path, format='png', dpi=300)
        plt.close()
        print(f"MSE difference analysis plot saved at {plot_path}")
    else:
        plt.show()

# Example usage
ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema_avg\nema_avg_40.TIFF"
denoised_image_path = r"Z:\members\Rauscher\projects\one_adj_slice\Nema_B-test_2\results\Nematostella_B\output_stack-Nema_B-test_2-Nematostella_B-epoch499.TIFF"
slice_index = 80  # Example slice index
species_name = "Nematostella"
x_range = (330, 340)  # Example x range
y_range = (120, 130)  # Example y range
plot_mse_difference_map(ground_truth_path, denoised_image_path, slice_index, species_name, x_range, y_range)

