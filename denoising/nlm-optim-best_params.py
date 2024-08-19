import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def nlm_filter_3d_image_stack(noisy_stack, ground_truth, h, patch_size_range, patch_distance_range, test_index):
    if test_index < 0 or test_index >= noisy_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = noisy_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth[test_index].astype(np.float64) / 65535.0
    
    psnr_values = []
    ssim_values = []
    best_psnr = -np.inf
    best_ssim = -np.inf
    best_params_psnr = None
    best_params_ssim = None
    
    patch_size_values, patch_distance_values = np.meshgrid(patch_size_range, patch_distance_range)
    
    for patch_size in patch_size_range:
        for patch_distance in patch_distance_range:
            print(f"Testing patch_size={patch_size}, patch_distance={patch_distance}")
            
            # Estimate the noise standard deviation from the test slice
            sigma_est = np.mean(estimate_sigma(test_slice))
            
            # Apply NLM filter to the test slice
            filtered_test_slice = denoise_nl_means(test_slice, h=h*sigma_est, fast_mode=True,
                                                   patch_size=patch_size, patch_distance=patch_distance)
            
            # Calculate PSNR and SSIM
            current_psnr = psnr(ground_truth_slice, filtered_test_slice)
            current_ssim = ssim(ground_truth_slice, filtered_test_slice, data_range=filtered_test_slice.max() - filtered_test_slice.min())
            print(f"PSNR: {current_psnr}, SSIM: {current_ssim}")

            psnr_values.append((patch_size, patch_distance, current_psnr))
            ssim_values.append((patch_size, patch_distance, current_ssim))

            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_params_psnr = (patch_size, patch_distance)
        
            if current_ssim > best_ssim:
                best_ssim = current_ssim
                best_params_ssim = (patch_size, patch_distance)
    
    print(f"Best PSNR: {best_psnr} with parameters patch_size={best_params_psnr[0]}, patch_distance={best_params_psnr[1]}")
    print(f"Best SSIM: {best_ssim} with parameters patch_size={best_params_ssim[0]}, patch_distance={best_params_ssim[1]}")
    
    return best_params_psnr, best_params_ssim, psnr_values, ssim_values

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\general-results"
    sample_paths = [
        (r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B-average-100.TIFF",
         r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF", 60),
        (r"C:\Users\rausc\Documents\EMBL\data\droso-results\Drosophila20210316LogScale01L_Good_Sample_02_t_-average-100-offset--2.TIFF",
         r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF", 100),
        (r"C:\Users\rausc\Documents\EMBL\data\mouse-results\MouseEmbryo20230602LogScaleMouse_embyo_10hour-average-20.TIFF",
         r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF", 250)
        # Add more (ground_truth_path, noisy_file_path, test_index) tuples as needed
    ]
    
    h = 1.4
    patch_size_range = np.arange(2, 25, 2)
    patch_distance_range = np.arange(2, 25, 2)
    custom_labels = [
        "nema",
        "droso",
        "mouse"
        # Add more custom labels as needed
    ]
    
    if len(custom_labels) != len(sample_paths):
        raise ValueError("The number of custom labels must match the number of samples.")
    
    all_psnr_scores = []
    all_ssim_scores = []
    best_params_psnr_values = []
    best_params_ssim_values = []

    for i, (gt_path, noisy_path, test_index) in enumerate(sample_paths):
        ground_truth = read_tiff_stack(gt_path)
        noisy_stack = read_tiff_stack(noisy_path)
        
        best_params_psnr, best_params_ssim, psnr_values, ssim_values = nlm_filter_3d_image_stack(noisy_stack, ground_truth, h, patch_size_range, patch_distance_range, test_index)
        
        all_psnr_scores.append(psnr_values)
        all_ssim_scores.append(ssim_values)
        best_params_psnr_values.append(best_params_psnr)
        best_params_ssim_values.append(best_params_ssim)
    
    # Aggregate PSNR and SSIM values over all samples
    psnr_dict = {}
    ssim_dict = {}
    for scores in all_psnr_scores:
        for patch_size, patch_distance, value in scores:
            psnr_dict[(patch_size, patch_distance)] = psnr_dict.get((patch_size, patch_distance), []) + [value]
    
    for scores in all_ssim_scores:
        for patch_size, patch_distance, value in scores:
            ssim_dict[(patch_size, patch_distance)] = ssim_dict.get((patch_size, patch_distance), []) + [value]

    avg_psnr_scores = {k: np.mean(v) for k, v in psnr_dict.items()}
    avg_ssim_scores = {k: np.mean(v) for k, v in ssim_dict.items()}
    
    # Normalize scores to 0-1 range
    min_psnr = min(avg_psnr_scores.values())
    max_psnr = max(avg_psnr_scores.values())
    min_ssim = min(avg_ssim_scores.values())
    max_ssim = max(avg_ssim_scores.values())
    
    normalized_psnr_scores = {k: (v - min_psnr) / (max_psnr - min_psnr) for k, v in avg_psnr_scores.items()}
    normalized_ssim_scores = {k: (v - min_ssim) / (max_ssim - min_ssim) for k, v in avg_ssim_scores.items()}
    
    # Compute combined scores and find best parameters
    combined_scores = {k: (normalized_psnr_scores[k] + normalized_ssim_scores[k]) / 2 for k in normalized_psnr_scores}
    best_params_overall = max(combined_scores, key=combined_scores.get)

    print(f"Best overall parameters: Patch Size={best_params_overall[0]}, Patch Distance={best_params_overall[1]}")
    
    # Prepare data for 3D plotting
    patch_size_values, patch_distance_values = np.meshgrid(patch_size_range, patch_distance_range)
    psnr_z = np.array([avg_psnr_scores.get((ps, pd), 0) for ps, pd in zip(patch_size_values.ravel(), patch_distance_values.ravel())]).reshape(patch_size_values.shape)
    ssim_z = np.array([avg_ssim_scores.get((ps, pd), 0) for ps, pd in zip(patch_size_values.ravel(), patch_distance_values.ravel())]).reshape(patch_size_values.shape)

    # Plot average PSNR and SSIM over all samples with separate 3D plots
    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(patch_size_values, patch_distance_values, psnr_z, cmap='viridis', alpha=0.8)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_xlabel('Patch Size')
    ax1.set_ylabel('Patch Distance')
    ax1.set_zlabel('Average PSNR')
    ax1.set_title('Average PSNR over all samples')

    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(patch_size_values, patch_distance_values, ssim_z, cmap='plasma', alpha=0.8)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    ax2.set_xlabel('Patch Size')
    ax2.set_ylabel('Patch Distance')
    ax2.set_zlabel('Average SSIM')
    ax2.set_title('Average SSIM over all samples')

    plt.suptitle('Average PSNR and SSIM over all samples')
    plt.tight_layout()
    plot_filename = 'nlm_psnr_ssim_comparison_3d.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.show()

    print(f"Plot saved to {os.path.join(output_dir, plot_filename)}")
