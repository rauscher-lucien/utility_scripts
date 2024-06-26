import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import mpmath

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

def crop_to_range(img_stack, z_range, x_range, y_range):
    if z_range is not None:
        img_stack = img_stack[z_range[0]:z_range[1], :, :]
    if x_range is not None:
        img_stack = img_stack[:, x_range[0]:x_range[1], :]
    if y_range is not None:
        img_stack = img_stack[:, :, y_range[0]:y_range[1]]
    return img_stack

def compute_noise(noisy, clean):
    old_min = 0.0
    old_max = 65536.0
    new_min = 0.0
    new_max = 1.0

    # noisy = new_min + ((noisy - old_min) / (old_max - old_min)) * (new_max - new_min)
    # clean = new_min + ((clean - old_min) / (old_max - old_min)) * (new_max - new_min) 

    # Take the exponent of both noisy and clean
    # noisy = np.exp(noisy)
    # clean = np.exp(clean)

    # Compute the noise by subtracting the clean data from the noisy data
    noise = noisy - clean

    return noise

def plot_noise_distribution(ground_truth_path, noisy_image_path, species_name, z_range, x_range, y_range):
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

    # Convert to float64 to prevent overflow in exponentiation
    cropped_ground_truth_stack = cropped_ground_truth_stack.astype(np.float64)
    cropped_noisy_image_stack = cropped_noisy_image_stack.astype(np.float64)

    # Crop to the specified range
    cropped_ground_truth_stack = crop_to_range(cropped_ground_truth_stack, z_range, x_range, y_range)
    cropped_noisy_image_stack = crop_to_range(cropped_noisy_image_stack, z_range, x_range, y_range)

    # Compute noise
    noise = compute_noise(cropped_noisy_image_stack, cropped_ground_truth_stack)

    # Flatten the noise array
    noise_flat = noise.flatten()

    # Plot the noise distribution
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot histogram normally
    ax.hist(noise_flat, bins=256, density=True, alpha=0.6, color='b', range=(-20000, 20000))

    ax.set_title(f'Noise Distribution for {species_name}')
    ax.set_xlabel('Noise Value')
    ax.set_ylabel('Density')
    ax.grid(True)

    # Save the figure as one plot
    plot_path = os.path.join(os.path.dirname(ground_truth_path), f'noise_distribution_{species_name}.png')
    plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Noise distribution plot saved at {plot_path}")

# Example usage
clean_image_path = r"C:\Users\rausc\Documents\EMBL\data\test_3\output_stack-mouse_embryo-test_1-mouse_embryo-epoch534.TIFF"
noisy_image_path = r"C:\Users\rausc\Documents\EMBL\data\test_3\Mouse_embyo_10hour_V0.TIFF"
species_name = "ME-noise-background-test-1"

# Define the x and y range for cropping
z_range = (227, 358)
x_range = (0, 70)
y_range = (0, 70)

# Plot with specified x and y ranges for the volumes
plot_noise_distribution(clean_image_path, noisy_image_path, species_name, z_range, x_range, y_range)
