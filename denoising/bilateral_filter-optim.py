import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.restoration import denoise_bilateral
from skimage.metrics import peak_signal_noise_ratio as psnr

def bilateral_filter_3d_image_stack(image_stack, ground_truth, sigma_color_range, sigma_spatial_range, test_index):
    if test_index < 0 or test_index >= image_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = image_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth[test_index].astype(np.float64) / 65535.0
    
    best_psnr = -np.inf
    best_params = None
    
    sigma_color_values, sigma_spatial_values = np.meshgrid(sigma_color_range, sigma_spatial_range)
    psnr_values = np.zeros_like(sigma_color_values)
    
    for i, sigma_color in enumerate(sigma_color_range):
        for j, sigma_spatial in enumerate(sigma_spatial_range):
            print(f"Testing sigma_color={sigma_color}, sigma_spatial={sigma_spatial}")
            
            # Apply Bilateral filter to the test slice
            filtered_test_slice = denoise_bilateral(test_slice, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
            
            # Calculate PSNR
            current_psnr = psnr(ground_truth_slice, filtered_test_slice)
            psnr_values[j, i] = current_psnr
            print(f"PSNR: {current_psnr}")

            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_params = (sigma_color, sigma_spatial)
    
    print(f"Best PSNR: {best_psnr} with sigma_color={best_params[0]}, sigma_spatial={best_params[1]}")
    
    # Plot PSNR values as a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(sigma_color_values, sigma_spatial_values, psnr_values, cmap='viridis')
    ax.set_title('PSNR for different combinations of sigma_color and sigma_spatial')
    ax.set_xlabel('Sigma Color')
    ax.set_ylabel('Sigma Spatial')
    ax.set_zlabel('PSNR')
    
    # Save the plot
    base_name = os.path.basename(data_path)
    name, ext = os.path.splitext(base_name)
    plot_filename = f"{name}_bilateral_filter__test_slice_{test_index}_PSNR_plot.png"
    plot_path = os.path.join(os.path.dirname(data_path), plot_filename)
    plt.savefig(plot_path)
    print(f"PSNR plot saved at {plot_path}")
    
    plt.show()
    
    return best_params

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour-average-10.TIFF"
    test_index = 250  # Example index to test a specific slice
    
    # Ranges for sigma_color and sigma_spatial parameters
    sigma_color_range = np.arange(0.06, 0.10, 0.01)
    sigma_spatial_range = np.arange(2, 6, 1.0)
    
    # Step 1: Read the 3D image stack and the ground truth
    image_stack = tifffile.imread(data_path)
    ground_truth = tifffile.imread(ground_truth_path)
    
    # Step 2: Find the best parameters for Bilateral filtering and plot PSNR
    best_params = bilateral_filter_3d_image_stack(image_stack, ground_truth, sigma_color_range, sigma_spatial_range, test_index)
