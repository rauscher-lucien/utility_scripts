import numpy as np
from PIL import Image
import os

def add_gaussian_noise(img, mean=0.0, std=50.0):
    """
    Add Gaussian noise to an image.
    Args:
        img (numpy.ndarray): Input image array.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    Returns:
        numpy.ndarray: Noisy image array.
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    noise = np.random.normal(mean, std, img.size)
    noisy_img_array = np.array(img) + noise.reshape(img.shape)
    return noisy_img_array.clip(0, 255).astype(np.uint8)

def process_image(input_path, output_path, mean=0.0, std=50.0):
    # Load image
    img = np.array(Image.open(input_path).convert('L'))  # Convert to grayscale for simplicity
    
    # Add noise
    noisy_img_array = add_gaussian_noise(img, mean, std)

    # Convert back to PIL image and save
    noisy_img = Image.fromarray(noisy_img_array)
    noisy_img.save(output_path)
    print(f"Noisy image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    input_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\12003_16bit_gray.tif"
    output_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\noisy\12003_noisy_2.jpg"
    process_image(input_path, output_path)
