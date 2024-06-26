import os
import numpy as np
import tifffile

def add_gaussian_noise_to_stack(file_path, noise_param, noise_shift=0):
    """
    Adds Gaussian noise to a TIFF stack with a given standard deviation or a range for it,
    and allows for shifting the peak of the Gaussian distribution.

    Parameters:
    - file_path: Path to the original TIFF file.
    - noise_param: Standard deviation of the Gaussian noise as a fraction of the max value,
                   or a tuple specifying a range from which to randomly select the std deviation.
    - noise_shift: Shift of the Gaussian noise mean from 0. It's a fraction of the max value,
                   representing the peak shift of the noise distribution.
    """
    # Read the original TIFF stack
    with tifffile.TiffFile(file_path) as tiff:
        images = tiff.asarray().astype(np.float32)
    
    # Since the original images are 16-bit, the maximum value is 65535
    max_val = 65535
    
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
    
    # Normalize the noisy image back to the 16-bit range
    noised_images = (noised_images - min_val) / (max_noised_val - min_val) * max_val
    noised_images = np.clip(noised_images, 0, max_val).astype(np.uint16)  # Ensure it's back to uint16

    # Normalize the original image back to the 16-bit range using the same min and max values
    normalized_images = (images - min_val) / (max_noised_val - min_val) * max_val
    normalized_images = np.clip(normalized_images, 0, max_val).astype(np.uint16)  # Ensure it's back to uint16

    # Construct the paths for the new TIFF files
    dir_path, file_name = os.path.split(file_path)
    noised_file_name = "noised2_" + file_name
    normalized_file_name = "normalized2_" + file_name
    noised_file_path = os.path.join(dir_path, noised_file_name)
    normalized_file_path = os.path.join(dir_path, normalized_file_name)

    # Save the noised stack as a new TIFF file
    tifffile.imwrite(noised_file_path, noised_images, photometric='minisblack')
    
    # Save the normalized original stack as a new TIFF file
    tifffile.imwrite(normalized_file_path, normalized_images, photometric='minisblack')

    print(f"Saved noised TIFF stack to: {noised_file_path}")
    print(f"Saved normalized TIFF stack to: {normalized_file_path}")

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\12003_16bit_gray.tif"
    add_gaussian_noise_to_stack(file_path, noise_param=0.1, noise_shift=0.0)

