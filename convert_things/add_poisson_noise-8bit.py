import os
import numpy as np
from PIL import Image

def add_poisson_noise_to_jpg(file_path, lam):
    """
    Adds Poisson noise to an 8-bit grayscale JPG image with a given lambda.

    Parameters:
    - file_path: Path to the original JPG file.
    - lam: Lambda scaling factor for the Poisson noise, controlling the noise level.
    """
    # Check if the file is an 8-bit grayscale JPG
    if not file_path.lower().endswith('.jpg'):
        raise ValueError("The file is not a JPG image.")
    
    with Image.open(file_path) as img:
        if img.mode != 'L':
            raise ValueError("The image is not an 8-bit grayscale image.")
        images = np.array(img).astype(np.float32)
    
    # Since the original images are 8-bit, the maximum value is 255
    max_val = 255

    # Scale images to represent photon counts by lambda
    scaled_images = images / max_val * lam
    
    # Generate Poisson noise based on the scaled images
    noised_images = np.random.poisson(scaled_images)
    
    # Scale back to the original range
    noised_images = noised_images / lam * max_val
    
    # Normalize both the noisy and original images using the same min and max values
    min_val = np.min(noised_images)
    max_noised_val = np.max(noised_images)
    
    # Normalize the noisy image back to the 8-bit range
    noised_images = (noised_images - min_val) / (max_noised_val - min_val) * max_val
    noised_images = np.clip(noised_images, 0, max_val).astype(np.uint8)  # Ensure it's back to uint8

    # Normalize the original image back to the 8-bit range using the same min and max values
    normalized_images = (images - min_val) / (max_noised_val - min_val) * max_val
    normalized_images = np.clip(normalized_images, 0, max_val).astype(np.uint8)  # Ensure it's back to uint8

    # Construct the paths for the new JPG files
    dir_path, file_name = os.path.split(file_path)
    noised_file_name = "poisson_noised2_" + file_name
    normalized_file_name = "normalized2_" + file_name
    noised_file_path = os.path.join(dir_path, noised_file_name)
    normalized_file_path = os.path.join(dir_path, normalized_file_name)

    # Save the noised and normalized images as new JPG files
    Image.fromarray(noised_images).save(noised_file_path, format='JPEG')
    Image.fromarray(normalized_images).save(normalized_file_path, format='JPEG')

    print(f"Saved Poisson noised JPG image to: {noised_file_path}")
    print(f"Saved normalized JPG image to: {normalized_file_path}")

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\12003_8bit_gray.jpg"
    add_poisson_noise_to_jpg(file_path, lam=10)



