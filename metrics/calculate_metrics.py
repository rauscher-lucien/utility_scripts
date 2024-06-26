import os
from PIL import Image
import numpy as np
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

# Example usage:
noisy_image_path = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'test_data_2', 'input.png')
denoised_image_path = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'test_data_2', 'output.png')

# Calculate metrics
snr_value, cnr_value, epi_value = calculate_metrics(noisy_image_path, denoised_image_path)
print("SNR:", snr_value)
print("CNR:", cnr_value)
print("EPI:", epi_value)
