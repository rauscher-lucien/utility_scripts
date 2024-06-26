import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

def read_jpg_image(filepath):
    with Image.open(filepath) as img:
        if img.mode != 'L':
            raise ValueError("The image is not an 8-bit grayscale image.")
        image = np.array(img)
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
    ground_truth_image = read_jpg_image(ground_truth_path)
    noisy_image = read_jpg_image(noisy_image_path)

    # Crop images to have dimensions that are multiples of 16
    cropped_ground_truth_image = crop_to_multiple_of_16(ground_truth_image)
    cropped_noisy_image = crop_to_multiple_of_16(noisy_image)

    # Ensure the cropped images have the same dimensions
    assert cropped_ground_truth_image.shape == cropped_noisy_image.shape, "Cropped images must have the same dimensions."

    # Convert images to float64 for noise calculation
    ground_truth_float = cropped_ground_truth_image.astype(np.float64)
    noisy_image_float = cropped_noisy_image.astype(np.float64)

    # optional log
    ground_truth_float = np.log(ground_truth_float)
    noisy_image_float = np.log(noisy_image_float)


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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    bins = np.arange(-2, 2, 0.05)  # Bins centered on integer values
    ax.hist(noise_flat, bins=bins, density=True, alpha=0.75, color='b', edgecolor='black')

    # Set the x-axis limit to [-255, 255], which is the range of possible differences between two 8-bit images
    ax.set_xlim([-2, 2])
    
    # Set titles and labels with improved formatting
    ax.set_title(f'Noise Distribution for {species_name}', fontsize=15, fontweight='bold')
    ax.set_xlabel('Noise Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Save the figure as one plot
    if save_plot:
        plot_path = os.path.join(os.path.dirname(ground_truth_path), f'noise_distribution_{species_name}.png')
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Noise distribution plot saved at {plot_path}")
    else:
        plt.show()

# Example usage
clean_image_path = r"Z:\members\Rauscher\projects\N2N-BSD300\BSD-8bit-test_3\results\BSD1\output.jpg"
noisy_image_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\gaussian-8bit\double_noisy\noised_12003_8bit_gray.jpg"
species_name = "BSD-gaussian-log-1"
save_values = False
plot_noise_distribution(clean_image_path, noisy_image_path, species_name, save_plot=True, save_values=save_values)


