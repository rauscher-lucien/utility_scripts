import os
import numpy as np
from PIL import Image

def add_gaussian_noise_to_jpg(file_path, noise_param, noise_shift=0):
    """
    Adds Gaussian noise to an 8-bit grayscale JPG image with a given standard deviation or a range for it,
    and allows for shifting the peak of the Gaussian distribution.

    Parameters:
    - file_path: Path to the original JPG file.
    - noise_param: Standard deviation of the Gaussian noise as a fraction of the max value,
                   or a tuple specifying a range from which to randomly select the std deviation.
    - noise_shift: Shift of the Gaussian noise mean from 0. It's a fraction of the max value,
                   representing the peak shift of the noise distribution.
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
    
    # Determine the standard deviation for the noise
    if isinstance(noise_param, (list, tuple)) and len(noise_param) == 2:
        std_dev = np.random.uniform(noise_param[0], noise_param[1]) * max_val
    else:
        std_dev = noise_param * max_val
    
    # Determine the shift for the Gaussian noise distribution
    mean_shift = noise_shift * max_val

    noise = np.random.normal(mean_shift, std_dev, images.shape)  # Generate Gaussian noise

    # Add noise
    noised_images = images + noise
    
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
    noised_file_name = "noised_" + file_name
    normalized_file_name = "normalized_" + file_name
    noised_file_path = os.path.join(dir_path, noised_file_name)
    normalized_file_path = os.path.join(dir_path, normalized_file_name)

    # Save the noised and normalized images as new JPG files
    Image.fromarray(noised_images).save(noised_file_path, format='JPEG')
    Image.fromarray(normalized_images).save(normalized_file_path, format='JPEG')

    print(f"Saved noised JPG image to: {noised_file_path}")
    print(f"Saved normalized JPG image to: {normalized_file_path}")

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\12003_8bit_gray.jpg"
    add_gaussian_noise_to_jpg(file_path, noise_param=0.1, noise_shift=0.0)
