import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import scipy.stats as stats
import pandas as pd

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

def scale_back_np(img):
    img = img.astype(np.float64)
    img = img / 65536
    img = img * 100 - 50
    img = np.power(10, img)
    return img

def rescale_noise(noise):
    noise = np.log10(noise)
    return noise

def compute_noise(noisy, clean, intensity_range=None):
    if intensity_range is not None:
        mask = (noisy >= intensity_range[0]) & (noisy <= intensity_range[1]) & (clean >= intensity_range[0]) & (clean <= intensity_range[1])
        noisy = noisy[mask]
        clean = clean[mask]

    noisy = scale_back_np(noisy)
    clean = scale_back_np(clean)

    noisy = np.log(noisy)
    clean = np.log(clean)

    diff = noisy - clean

    true_noise = (np.power(10, diff) - 1) * np.power(10, clean)

    return true_noise

def characterize_noise_distribution(noise):
    noise_flat = noise.flatten()

    mean = np.mean(noise_flat)
    median = np.median(noise_flat)
    std_dev = np.std(noise_flat)
    variance = np.var(noise_flat)
    skewness = stats.skew(noise_flat)
    kurtosis = stats.kurtosis(noise_flat)

    stats_dict = {
        'mean': mean,
        'median': median,
        'std_dev': std_dev,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

    return stats_dict

def save_noise_to_csv(noise, output_path):
    noise_flat = noise.flatten()
    noise_df = pd.DataFrame(noise_flat, columns=["noise"])
    noise_df.to_csv(output_path, index=False)
    print(f"Noise data saved to {output_path}")

def plot_noise_distribution(ground_truth_path, noisy_image_path, species_name, z_range, x_range, y_range, intensity_range=None, bins=1000, log_scale=True, save_csv=False):
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

    noise = compute_noise(cropped_noisy_image_stack, cropped_ground_truth_stack, intensity_range=intensity_range)

    noise_flat = noise.flatten()

    noise_stats = characterize_noise_distribution(noise)
    print("Noise Distribution Statistics:", noise_stats)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    if log_scale:
        bin_edges = np.logspace(np.log10(1e-50), np.log10(1e50), bins)
    else:
        bin_edges = np.linspace(-50, 50, bins)

    ax.hist(noise_flat, bins=bin_edges, density=True, alpha=0.6, color='b')

    ax.set_title(f'Noise Distribution for {species_name}')
    ax.set_xlabel('Noise Value')
    ax.set_ylabel('Count')
    ax.grid(True)

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    plot_path = os.path.join(os.path.dirname(ground_truth_path), f'noise_distribution_{species_name}.png')
    plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Noise distribution plot saved at {plot_path}")

    if save_csv:
        csv_path = os.path.join(os.path.dirname(ground_truth_path), f'noise_data_{species_name}.csv')
        save_noise_to_csv(noise, csv_path)

# Example usage
clean_image_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-Nema_B-test_3-Nematostella_B-epoch501.TIFF"
noisy_image_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF"
species_name = "nema-noise-network-test"

# droso
# (39, 149)
# (70, 180)
# (66, 127)

# mouse 
# (202, 347)
# (174, 313)
# (56, 109)

z_range = None
x_range = None
y_range = None
intensity_range = None
bins = 1000

log_scale = False
save_csv = True

plot_noise_distribution(clean_image_path, noisy_image_path, species_name, z_range, x_range, y_range, intensity_range=intensity_range, bins=bins, log_scale=log_scale, save_csv=save_csv)
