import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr

def nlm_filter_3d_image_stack(image_stack, ground_truth, h_range, patch_size_range, patch_distance_range, test_index):
    if test_index < 0 or test_index >= image_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = image_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth[test_index].astype(np.float64) / 65535.0
    
    best_psnr = -np.inf
    best_params = None
    
    for h in h_range:
        for patch_size in patch_size_range:
            for patch_distance in patch_distance_range:
                print(f"Testing h={h}, patch_size={patch_size}, patch_distance={patch_distance}")
                
                # Estimate the noise standard deviation from the test slice
                sigma_est = np.mean(estimate_sigma(test_slice))
                
                # Apply NLM filter to the test slice
                filtered_test_slice = denoise_nl_means(test_slice, h=h*sigma_est, fast_mode=True,
                                                       patch_size=patch_size, patch_distance=patch_distance)
                
                # Calculate PSNR
                current_psnr = psnr(ground_truth_slice, filtered_test_slice)
                print(f"PSNR: {current_psnr}")

                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    best_params = (h, patch_size, patch_distance)
    
    print(f"Best PSNR: {best_psnr} with parameters h={best_params[0]}, patch_size={best_params[1]}, patch_distance={best_params[2]}")
    return best_params

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\droso_good_avg_40-offset-2.TIFF"
    test_index = 80  # Example index to test a specific slice
    
    # Ranges and step sizes for parameters
    h_range = np.arange(1.4, 1.5, 0.1)
    patch_size_range = np.arange(4, 8, 1)
    patch_distance_range = np.arange(10, 30, 5)
    
    # Step 1: Read the 3D image stack and the ground truth
    image_stack = tifffile.imread(data_path)
    ground_truth = tifffile.imread(ground_truth_path)
    
    # Step 2: Find the best parameters for NLM filtering
    best_params = nlm_filter_3d_image_stack(image_stack, ground_truth, h_range, patch_size_range, patch_distance_range, test_index)
    
    # Use the best parameters to filter the entire stack (if needed)
    # filtered_stack = nlm_filter_3d_image_stack(image_stack, *best_params)
    # save_filtered_stack(filtered_stack, data_path)
