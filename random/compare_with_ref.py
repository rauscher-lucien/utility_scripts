import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def compare_with_reference(reference_img_path, denoised_folder):
    # Load reference denoised image
    reference_img = np.array(Image.open(reference_img_path).convert("L"))

    mse_values = []

    # Get list of file names in the denoised folder
    denoised_files = sorted(os.listdir(denoised_folder))

    for denoised_file in denoised_files:
        denoised_img_path = os.path.join(denoised_folder, denoised_file)
        # Load denoised image
        denoised_img = np.array(Image.open(denoised_img_path).convert("L"))

        # Calculate MSE
        mse_val = mse(reference_img, denoised_img)
        mse_values.append(mse_val)

    return mse_values

def plot_mse_comparison(mse_values):
    plt.figure(figsize=(10, 6))
    plt.plot(mse_values, label='MSE with Reference Image')
    plt.xlabel('Image Index')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison with Reference Denoised Image')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
reference_denoised_image_path = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'test_data_2', 'output.png')
other_denoised_folder = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'test_denoised')

mse_values = compare_with_reference(reference_denoised_image_path, other_denoised_folder)
plot_mse_comparison(mse_values)
