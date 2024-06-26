import os
import numpy as np
import tifffile
import pandas as pd
import pywt

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

def wavelet_analysis(noise_image, wavelet='db1', level=2):
    coeffs = pywt.wavedec2(noise_image, wavelet=wavelet, level=level)
    cA, *details = coeffs

    metrics = []
    for i, (cH, cV, cD) in enumerate(details):
        mean_H = np.mean(cH)
        std_H = np.std(cH)
        speckle_index_H = std_H / mean_H if mean_H != 0 else float('inf')
        
        mean_V = np.mean(cV)
        std_V = np.std(cV)
        speckle_index_V = std_V / mean_V if mean_V != 0 else float('inf')
        
        mean_D = np.mean(cD)
        std_D = np.std(cD)
        speckle_index_D = std_D / mean_D if mean_D != 0 else float('inf')
        
        metrics.append({
            'Level': i+1,
            'Mean_H': mean_H,
            'Std_H': std_H,
            'Speckle_Index_H': speckle_index_H,
            'Mean_V': mean_V,
            'Std_V': std_V,
            'Speckle_Index_V': speckle_index_V,
            'Mean_D': mean_D,
            'Std_D': std_D,
            'Speckle_Index_D': speckle_index_D
        })
    
    return metrics

def analyze_noise_with_wavelet(ground_truth_path, noisy_image_path, species_name, x_range=None, y_range=None, save_values=True):
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

    # Apply x and y range if specified
    if x_range:
        cropped_ground_truth_stack = cropped_ground_truth_stack[:, :, x_range[0]:x_range[1]]
        cropped_noisy_image_stack = cropped_noisy_image_stack[:, :, x_range[0]:x_range[1]]
    if y_range:
        cropped_ground_truth_stack = cropped_ground_truth_stack[:, y_range[0]:y_range[1], :]
        cropped_noisy_image_stack = cropped_noisy_image_stack[:, y_range[0]:y_range[1], :]

    # Compute noise by subtracting the ground truth from the noisy image
    noise = cropped_noisy_image_stack.astype(np.float32) - cropped_ground_truth_stack.astype(np.float32)

    # Perform wavelet analysis on the noise
    wavelet_metrics = wavelet_analysis(noise)

    # Convert wavelet metrics to DataFrame and save
    if save_values:
        wavelet_df = pd.DataFrame(wavelet_metrics)
        wavelet_values_path = os.path.join(os.path.dirname(ground_truth_path), f'wavelet_metrics_{species_name}.csv')
        wavelet_df.to_csv(wavelet_values_path, index=False)
        print(f"Wavelet metrics saved at {wavelet_values_path}")

# Example usage
clean_image_path = r"Z:\members\Rauscher\projects\one_adj_slice\mouse_embryo-test_1\results\mouse_embryo\output_stack-mouse_embryo-test_1-mouse_embryo-epoch534.TIFF"
noisy_image_path = r"Z:\members\Rauscher\data\big_data_small\mouse_embryo\Mouse_embyo_10hour_V0.TIFF"
species_name = "Mouse Embryo"
x_range = None  # Example x range
y_range = None  # Example y range
analyze_noise_with_wavelet(clean_image_path, noisy_image_path, species_name, x_range, y_range)
