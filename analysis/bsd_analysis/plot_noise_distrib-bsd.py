import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pandas as pd

def read_tiff_image(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        image = tiff.asarray()
    return image

def crop_to_multiple_of_16(image):
    h, w = image.shape
    new_h = h - (h % 16)
    new_w = w - (w % 16)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return image[top:top+new_h, left:left+new_w]

def save_noise_values(noise_values, output_path):
    df = pd.DataFrame(noise_values, columns=['Noise'])
    df.to_csv(output_path, index=False)

def plot_noise_distribution(ground_truth_path, noisy_image_path, species_name, 
                            save_plot=True, save_values=False):
    # Load the ground truth and noisy images
    ground_truth_image = read_tiff_image(ground_truth_path)
    noisy_image = read_tiff_image(noisy_image_path)

    # Crop images to have dimensions that are multiples of 16
    cropped_ground_truth_image = crop_to_multiple_of_16(ground_truth_image)
    cropped_noisy_image = crop_to_multiple_of_16(noisy_image)

    # Ensure the cropped images have the same dimensions
    assert cropped_ground_truth_image.shape == cropped_noisy_image.shape, "Cropped images must have the same dimensions."

    # Convert images to float64 for noise calculation
    ground_truth_float = cropped_ground_truth_image.astype(np.float64)
    noisy_image_float = cropped_noisy_image.astype(np.float64)

    # Compute noise by subtracting the ground truth from the noisy image
    noise = noisy_image_float - ground_truth_float

    # Flatten the noise array
    noise_flat = noise.flatten()

    # Save the noise values to a CSV file if save_values is True
    if save_values:
        values_path = os.path.join(os.path.dirname(ground_truth_path), f'noise_values_{species_name}.csv')
        save_noise_values(noise_flat, values_path)
        print(f"Noise values saved at {values_path}")

    # Plot the noise distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(noise_flat, bins=200, density=True, alpha=0.6, color='b')

    ax.set_title(f'Noise Distribution for {species_name}')
    ax.set_xlabel('Noise Value')
    ax.set_ylabel('Density')
    ax.grid(True)

    # Save the figure as one plot
    if save_plot:
        plot_path = os.path.join(os.path.dirname(ground_truth_path), f'noise_distribution_{species_name}.png')
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Noise distribution plot saved at {plot_path}")
    else:
        plt.show()

# Example usage
clean_image_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\12003_16bit_gray.tif"
noisy_image_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\poisson_noised_12003_16bit_gray.tif"
species_name = "BSD-poisson"
save_values = True
plot_noise_distribution(clean_image_path, noisy_image_path, species_name, save_plot=True, save_values=save_values)

