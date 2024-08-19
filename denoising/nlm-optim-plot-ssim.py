import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import structural_similarity as ssim

def nlm_filter_3d_image_stack_ssim(image_stack, ground_truth, h, patch_size_range, patch_distance_range, test_index):
    if test_index < 0 or test_index >= image_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = image_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth[test_index].astype(np.float64) / 65535.0
    
    best_ssim = -np.inf
    best_params = None
    
    patch_size_values, patch_distance_values = np.meshgrid(patch_size_range, patch_distance_range)
    ssim_values = np.zeros_like(patch_size_values, dtype=np.float64)
    
    for i, patch_size in enumerate(patch_size_range):
        for j, patch_distance in enumerate(patch_distance_range):
            print(f"Testing patch_size={patch_size}, patch_distance={patch_distance}")
            
            # Estimate the noise standard deviation from the test slice
            sigma_est = np.mean(estimate_sigma(test_slice))
            
            # Apply NLM filter to the test slice
            filtered_test_slice = denoise_nl_means(test_slice, h=h*sigma_est, fast_mode=True,
                                                   patch_size=patch_size, patch_distance=patch_distance)
            
            # Calculate SSIM
            current_ssim = ssim(ground_truth_slice, filtered_test_slice, data_range=filtered_test_slice.max() - filtered_test_slice.min())
            ssim_values[j, i] = current_ssim
            print(f"SSIM: {current_ssim}")

            if current_ssim > best_ssim:
                best_ssim = current_ssim
                best_params = (patch_size, patch_distance)
    
    print(f"Best SSIM: {best_ssim} with parameters patch_size={best_params[0]}, patch_distance={best_params[1]}")
    
    # Plot SSIM values as a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(patch_size_values, patch_distance_values, ssim_values, cmap='viridis')
    ax.set_title('SSIM for different combinations of patch_size and patch_distance')
    ax.set_xlabel('Patch Size')
    ax.set_ylabel('Patch Distance')
    ax.set_zlabel('SSIM')
    
    # Save the plot
    base_name = os.path.basename(data_path)
    name, ext = os.path.splitext(base_name)
    plot_filename = f"{name}_test_slice_{test_index}_SSIM_plot.png"
    plot_path = os.path.join(os.path.dirname(data_path), plot_filename)
    plt.savefig(plot_path)
    print(f"SSIM plot saved at {plot_path}")
    
    plt.show()
    
    return best_params

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B-average-100.TIFF"
    test_index = 60  # Example index to test a specific slice
    
    # Constant h value
    h = 1.4
    
    # Ranges and step sizes for parameters
    patch_size_range = np.arange(2, 25, 2)
    patch_distance_range = np.arange(2, 25, 2)
    
    # Step 1: Read the 3D image stack and the ground truth
    image_stack = tifffile.imread(data_path)
    ground_truth = tifffile.imread(ground_truth_path)
    
    # Step 2: Find the best parameters for NLM filtering
    best_params = nlm_filter_3d_image_stack_ssim(image_stack, ground_truth, h, patch_size_range, patch_distance_range, test_index)