import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import mpmath

def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray().squeeze()
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
    if y_range is not None:
        img_stack = img_stack[:, y_range[0]:y_range[1], :]
    if x_range is not None:
        img_stack = img_stack[:, :, x_range[0]:x_range[1]]
    return img_stack

def compute_noise(noisy, clean, scale=False, exp_scale=False, intensity_range=None):
    old_min = 0.0
    old_max = 65536.0
    new_min = 0.0
    new_max = 1.0

    if intensity_range is not None:
        mask = (noisy >= intensity_range[0]) & (noisy <= intensity_range[1]) & (clean >= intensity_range[0]) & (clean <= intensity_range[1])
        noisy = noisy[mask]
        clean = clean[mask]

    if scale:
        noisy = new_min + ((noisy - old_min) / (old_max - old_min)) * (new_max - new_min)
        clean = new_min + ((clean - old_min) / (old_max - old_min)) * (new_max - new_min)

    if exp_scale:
        noisy = np.exp(noisy)
        clean = np.exp(clean)

    noise = noisy - clean

    return noise

def plot_noise_distribution(ground_truth_path, noisy_image_path, species_name, z_range, x_range, y_range, scale=False, exp_scale=False, intensity_range=None):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    noisy_image_stack = read_tiff_stack(noisy_image_path).squeeze()

    min_slices = min(ground_truth_stack.shape[0], noisy_image_stack.shape[0])
    ground_truth_stack = ground_truth_stack[:min_slices].squeeze()
    noisy_image_stack = noisy_image_stack[:min_slices].squeeze()

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_noisy_image_stack = crop_to_multiple_of_16(noisy_image_stack)

    assert cropped_ground_truth_stack.shape == cropped_noisy_image_stack.shape, "Cropped stacks must have the same dimensions."

    cropped_ground_truth_stack = cropped_ground_truth_stack.astype(np.float64)
    cropped_noisy_image_stack = cropped_noisy_image_stack.astype(np.float64)

    cropped_ground_truth_stack = crop_to_range(cropped_ground_truth_stack, z_range, x_range, y_range)
    cropped_noisy_image_stack = crop_to_range(cropped_noisy_image_stack, z_range, x_range, y_range)

    noise = compute_noise(cropped_noisy_image_stack, cropped_ground_truth_stack, scale=scale, exp_scale=exp_scale, intensity_range=intensity_range)

    noise_flat = noise.flatten()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(noise_flat, bins=256, density=True, alpha=0.6, color='b', range=(-10000, 10000))

    ax.set_title(f'Noise Distribution for {species_name}')
    ax.set_xlabel('Noise Value')
    ax.set_ylabel('Density')
    ax.grid(True)

    plot_path = os.path.join(os.path.dirname(ground_truth_path), f'noise_distribution_{species_name}.png')
    plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Noise distribution plot saved at {plot_path}")

# Example usage
clean_image_path = r"C:\Users\rausc\Documents\EMBL\data\test_2\droso_good_avg_40-offset-2.TIFF"
noisy_image_path = r"C:\Users\rausc\Documents\EMBL\data\test_2\Good_Sample_02_t_1.TIFF"
species_name = "Droso-noise-GT-inside-1"

z_range = (61, 141)
x_range = (72, 164)
y_range = (48, 136)
scale = False
exp_scale = False
intensity_range = None

plot_noise_distribution(clean_image_path, noisy_image_path, species_name, z_range, x_range, y_range, scale, exp_scale, intensity_range)
