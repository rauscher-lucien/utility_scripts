import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def normalize_noise(noise):
    noise_min = noise.min()
    noise_max = noise.max()
    return (noise - noise_min) / (noise_max - noise_min)

def clip_noise(noise, clip_range):
    return np.clip(noise, clip_range[0], clip_range[1])

def plot_noise_pattern(ground_truth_path, noisy_image_path, slice_index, species_name, clip_range=None, x_range=None, y_range=None, save_plot=True):
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

    # Normalize the noise to the 0-1 range
    normalized_noise = normalize_noise(noise)

    # Clip the noise if clip_range is provided
    if clip_range:
        normalized_noise = clip_noise(normalized_noise, clip_range)

    # Apply x and y range if specified
    if x_range:
        normalized_noise = normalized_noise[:, x_range[0]:x_range[1]]
    if y_range:
        normalized_noise = normalized_noise[y_range[0]:y_range[1], :]

    # Plot only the normalized noise pattern
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(normalized_noise, cmap='gray', vmin=clip_range[0] if clip_range else 0, vmax=clip_range[1] if clip_range else 1)
    ax.set_title(f'Normalized Noise Pattern (Slice {slice_index}) - {species_name}')
    ax.axis('on')

    # Create a colorbar that matches the height of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Intensity')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure as one plot
    if save_plot:
        clip_str = f"_clip_{clip_range[0]}_{clip_range[1]}" if clip_range else ""
        x_str = f"_xrange_{x_range[0]}_{x_range[1]}" if x_range else ""
        y_str = f"_yrange_{y_range[0]}_{y_range[1]}" if y_range else ""
        plot_path = os.path.join(os.path.dirname(ground_truth_path), f'noise_analysis_{species_name}_slice_{slice_index}_normed{clip_str}{x_str}{y_str}.png')
        plt.savefig(plot_path, format='png', dpi=300)
        plt.close()
        print(f"Noise analysis plot saved at {plot_path}")
    else:
        plt.show()

# Example usage
clean_image_path = r"Z:\members\Rauscher\projects\one_adj_slice\mouse_embryo-test_1\results\mouse_embryo\output_stack-mouse_embryo-test_1-mouse_embryo-epoch534.TIFF"
noisy_image_path = r"Z:\members\Rauscher\data\big_data_small\mouse_embryo\Mouse_embyo_10hour_V0.TIFF"
slice_index = 290  # Example slice index
species_name = "Mouse Embryo"
clip_range = (0.6, 1.0)  # Example clip range
x_range = (200, 250)  # Example x range
y_range = (50, 100)  # Example y range
plot_noise_pattern(clean_image_path, noisy_image_path, slice_index, species_name, clip_range, x_range, y_range)






