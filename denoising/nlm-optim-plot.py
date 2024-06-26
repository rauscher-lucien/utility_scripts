import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr

def nlm_filter_3d_image_stack(image_stack, ground_truth, h, patch_size_range, patch_distance_range, test_index):
    if test_index < 0 or test_index >= image_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = image_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth[test_index].astype(np.float64) / 65535.0
    
    best_psnr = -np.inf
    best_params = None
    
    patch_size_values, patch_distance_values = np.meshgrid(patch_size_range, patch_distance_range)
    psnr_values = np.zeros_like(patch_size_values, dtype=np.float64)
    
    for i, patch_size in enumerate(patch_size_range):
        for j, patch_distance in enumerate(patch_distance_range):
            print(f"Testing patch_size={patch_size}, patch_distance={patch_distance}")
            
            # Estimate the noise standard deviation from the test slice
            sigma_est = np.mean(estimate_sigma(test_slice))
            
            # Apply NLM filter to the test slice
            filtered_test_slice = denoise_nl_means(test_slice, h=h*sigma_est, fast_mode=True,
                                                   patch_size=patch_size, patch_distance=patch_distance)
            
            # Calculate PSNR
            current_psnr = psnr(ground_truth_slice, filtered_test_slice)
            psnr_values[j, i] = current_psnr
            print(f"PSNR: {current_psnr}")

            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_params = (patch_size, patch_distance)
    
    print(f"Best PSNR: {best_psnr} with parameters patch_size={best_params[0]}, patch_distance={best_params[1]}")
    
    # Plot PSNR values as a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(patch_size_values, patch_distance_values, psnr_values, cmap='viridis')
    ax.set_title('PSNR for different combinations of patch_size and patch_distance')
    ax.set_xlabel('Patch Size')
    ax.set_ylabel('Patch Distance')
    ax.set_zlabel('PSNR')
    
    # Save the plot
    base_name = os.path.basename(data_path)
    name, ext = os.path.splitext(base_name)
    plot_filename = f"{name}_test_slice_{test_index}_PSNR_plot.png"
    plot_path = os.path.join(os.path.dirname(data_path), plot_filename)
    plt.savefig(plot_path)
    print(f"PSNR plot saved at {plot_path}")
    
    plt.show()
    
    return best_params

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\droso_good_avg_40-offset-2.TIFF"
    test_index = 80  # Example index to test a specific slice
    
    # Constant h value
    h = 1.4
    
    # Ranges and step sizes for parameters
    patch_size_range = np.arange(2, 21, 2)
    patch_distance_range = np.arange(2, 21, 2)
    
    # Step 1: Read the 3D image stack and the ground truth
    image_stack = tifffile.imread(data_path)
    ground_truth = tifffile.imread(ground_truth_path)
    
    # Step 2: Find the best parameters for NLM filtering
    best_params = nlm_filter_3d_image_stack(image_stack, ground_truth, h, patch_size_range, patch_distance_range, test_index)
