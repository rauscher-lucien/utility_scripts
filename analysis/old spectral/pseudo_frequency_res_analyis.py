import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os

def read_tiff_stack(filepath):
    # Reads a TIFF file and returns it as a numpy array.
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def crop_to_multiple_of_16(img_stack):
    # Crops an image stack so that the height and width are multiples of 16.
    h, w = img_stack.shape[1:3]
    new_h = h - (h % 16)
    new_w = w - (w % 16)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img_stack[:, top:top+new_h, left:left+new_w]

def calculate_residuals(noisy_stack, denoised_stack):
    # Calculates the residuals between the noisy and denoised stacks.
    cropped_noisy_stack = crop_to_multiple_of_16(noisy_stack[2:-2])
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)
    return cropped_noisy_stack - cropped_denoised_stack

def frequency_analysis(residuals, output_dir):
    # Perform a Fourier Transform on the residuals and analyze the frequency domain.
    f_transform = np.fft.fftshift(np.fft.fftn(residuals))
    magnitude_spectrum = np.log(np.abs(f_transform) + 1)
    
    # Plot and save the average magnitude spectrum
    avg_magnitude_spectrum = np.mean(magnitude_spectrum, axis=0)  # Assuming residuals is of shape (N, H, W)
    plt.imshow(avg_magnitude_spectrum, cmap='hot')
    plt.colorbar()
    plt.title("Average Magnitude Spectrum of Residuals")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    spectrum_filename = 'average_magnitude_spectrum.png'
    plt.savefig(os.path.join(output_dir, spectrum_filename))
    print(f"Average magnitude spectrum saved to {os.path.join(output_dir, spectrum_filename)}")
    plt.close()

if __name__ == "__main__":
    path_to_noisy_image = r"Z:\members\Rauscher\data\big_data_small\good_sample_unidentified\good_sample_unidentified.tif"
    path_to_denoised_image = r"C:\Users\rausc\Documents\EMBL\projects\FastDVDNet\good_sample-unidentified-test_1\results\inference_300-good_sample_unidentified\output_stack-good_sample-unidentified-test_1-inference_300-good_sample_unidentified.TIFF"
    output_directory = r"C:\Users\rausc\Documents\EMBL\projects\FastDVDNet\good_sample-unidentified-test_1\results\inference_300-good_sample_unidentified"

    noisy_stack = read_tiff_stack(path_to_noisy_image)
    denoised_stack = read_tiff_stack(path_to_denoised_image)

    if noisy_stack.shape[0] < 5:
        raise ValueError("The noisy stack is too small to crop!")

    residuals = calculate_residuals(noisy_stack, denoised_stack)
    frequency_analysis(residuals, output_directory)
