import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def nlm_filter_test_slice(noisy_stack, ground_truth_stack, h, patch_size, patch_distance, test_index):
    if test_index < 0 or test_index >= noisy_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = noisy_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth_stack[test_index].astype(np.float64) / 65535.0
    
    # Estimate the noise standard deviation from the test slice
    sigma_est = np.mean(estimate_sigma(test_slice))
    
    # Apply NLM filter to the test slice
    filtered_test_slice = denoise_nl_means(test_slice, h=h*sigma_est, fast_mode=True,
                                           patch_size=patch_size, patch_distance=patch_distance)
    
    # Calculate PSNR and SSIM
    test_psnr = psnr(ground_truth_slice, filtered_test_slice)
    test_ssim = ssim(ground_truth_slice, filtered_test_slice, data_range=filtered_test_slice.max() - filtered_test_slice.min())
    
    print(f"PSNR: {test_psnr}, SSIM: {test_ssim}")

    # Display the test slice result
    plt.imshow(filtered_test_slice, cmap='gray')
    plt.title(f"Filtered Test Slice {test_index}\nPSNR: {test_psnr:.2f}, SSIM: {test_ssim:.4f}")
    plt.colorbar()
    plt.show()

    return test_psnr, test_ssim

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B-average-100.TIFF"
    h = 1.9  # Parameter for NLM filter, higher means more smoothing
    patch_size = 8  # Size of patches used for NLM filter
    patch_distance = 14  # Maximum distance to search patches for NLM filter
    test_index = 60  # Example index to test a specific slice

    # Step 1: Read the 3D image stack and the ground truth stack
    noisy_stack = tifffile.imread(data_path)
    ground_truth_stack = tifffile.imread(ground_truth_path)

    # Step 2: Apply NLM filtering to the test slice and compute PSNR and SSIM
    test_psnr, test_ssim = nlm_filter_test_slice(noisy_stack, ground_truth_stack, h, patch_size, patch_distance, test_index)

    print(f"Test slice {test_index} PSNR: {test_psnr:.2f}, SSIM: {test_ssim:.4f}")
