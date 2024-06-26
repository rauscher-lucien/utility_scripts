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

def plot_noise_pattern(ground_truth_path, noisy_image_path, slice_index, species_name, save_plot=True, normalize=False):
    # Load the ground truth and noisy images
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    noisy_image_stack = read_tiff_stack(noisy_image_path).squeeze()

    # Ensure the stacks have the same number of slices based on the smaller stack
    min_slices = min(ground_truth_stack.shape[0], noisy_image_stack.shape[0])
    ground_truth_stack = ground_truth_stack[:min_slices].squeeze()
    noisy_image_stack = noisy_image_stack[:min_slices].squeeze()

    # Crop stacks to have dimensions that are multiples of 16
    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_noisy_image_stack = crop_to_multiple_of_16(noisy_image_stack)

    # Ensure the cropped stacks have the same dimensions
    assert cropped_ground_truth_stack.shape == cropped_noisy_image_stack.shape, "Cropped stacks must have the same dimensions."

    # Ensure the slice index is within the valid range
    if slice_index < 0 or slice_index >= cropped_ground_truth_stack.shape[0]:
        raise IndexError(f"Slice index {slice_index} is out of range. Must be between 0 and {cropped_ground_truth_stack.shape[0] - 1}.")

    # Get the specified slice
    ground_truth = cropped_ground_truth_stack[slice_index].astype(np.float32)
    noisy_image = cropped_noisy_image_stack[slice_index].astype(np.float32)

    # Compute noise by subtracting the ground truth from the noisy image
    noise = noisy_image - ground_truth

    # Normalize the noise to the range 0-1 if the option is enabled
    if normalize:
        noise = noise / 65535.0

    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the noise in real space
    im = ax.imshow(noise, cmap='gray', vmin=0, vmax=1 if normalize else noise.max())
    ax.set_title(f'Noise Pattern (Slice {slice_index})')
    ax.axis('on')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'Noise Analysis for {species_name} - Slice {slice_index}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure as one plot
    if save_plot:
        plot_path = os.path.join(os.path.dirname(ground_truth_path), f'noise_analysis_{species_name}_slice_{slice_index}.png')
        plt.savefig(plot_path, format='png', dpi=300)
        plt.close()
        print(f"Noise analysis plot saved at {plot_path}")
    else:
        plt.show()

# Example usage
clean_image_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\output_stack-droso_good-test_1-droso_good-epoch503.TIFF"
noisy_image_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF"
slice_index = 95  # Example slice index
species_name = "droso-noise-network"
plot_noise_pattern(clean_image_path, noisy_image_path, slice_index, species_name, normalize=True)

