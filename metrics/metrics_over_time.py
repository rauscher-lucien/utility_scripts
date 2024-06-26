import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(noisy_img_path, denoised_img_path):
    # Load images
    noisy_img = np.array(Image.open(noisy_img_path).convert("L"))  # Convert to grayscale
    denoised_img = np.array(Image.open(denoised_img_path).convert("L"))

    # Compute Mean Squared Error (MSE)
    mse = np.mean((noisy_img - denoised_img) ** 2)

    # Compute Mean Squared Value of the original noisy image
    original_mse = np.mean(noisy_img ** 2)

    # Calculate SNR in dB
    snr = 10 * np.log10(original_mse / mse)

    # Compute contrast of the original noisy image
    noisy_contrast = np.std(noisy_img)

    # Compute contrast of the difference between the original noisy image and the denoised image
    diff_contrast = np.std(noisy_img - denoised_img)

    # Calculate CNR
    cnr = diff_contrast / noisy_contrast

    # Calculate SSIM (Structural Similarity Index)
    ssim_index, _ = ssim(noisy_img, denoised_img, full=True)

    # Calculate EPI (Edge Preservation Index)
    epi = ssim_index / (1 + mse)

    return snr, cnr, epi

def compute_metrics_over_time(noisy_folder, denoised_folder):
    snr_values = []
    cnr_values = []
    epi_values = []

    # Get list of file names in the noisy folder
    noisy_files = sorted(os.listdir(noisy_folder))
    denoised_files = sorted(os.listdir(denoised_folder))

    for noisy_file, denoised_file in zip(noisy_files, denoised_files):
        noisy_img_path = os.path.join(noisy_folder, noisy_file)
        denoised_img_path = os.path.join(denoised_folder, denoised_file)

        snr, cnr, epi = calculate_metrics(noisy_img_path, denoised_img_path)

        snr_values.append(snr)
        cnr_values.append(cnr)
        epi_values.append(epi)

    return snr_values, cnr_values, epi_values

def plot_metrics_evolution(snr_values, cnr_values, epi_values):
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, label='SNR')
    plt.plot(cnr_values, label='CNR')
    plt.plot(epi_values, label='EPI')
    plt.xlabel('Time')
    plt.ylabel('Metric Value')
    plt.title('Evolution of Metrics over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
noisy_folder = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'test_noisy')
denoised_folder = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'test_denoised')

snr_values, cnr_values, epi_values = compute_metrics_over_time(noisy_folder, denoised_folder)
plot_metrics_evolution(snr_values, cnr_values, epi_values)
