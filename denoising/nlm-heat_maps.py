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

def nlm_filter_3d_image_stack(noisy_stack, ground_truth, h_range, patch_size_range, patch_distance_range, test_index):
    if test_index < 0 or test_index >= noisy_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = noisy_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth[test_index].astype(np.float64) / 65535.0
    
    psnr_values = []
    ssim_values = []
    
    for h in h_range:
        for patch_size in patch_size_range:
            for patch_distance in patch_distance_range:
                print(f"Testing h={h}, patch_size={patch_size}, patch_distance={patch_distance}")
                
                # Estimate the noise standard deviation from the test slice
                sigma_est = np.mean(estimate_sigma(test_slice))
                
                # Apply NLM filter to the test slice
                filtered_test_slice = denoise_nl_means(test_slice, h=h*sigma_est, fast_mode=True,
                                                       patch_size=patch_size, patch_distance=patch_distance)
                
                # Calculate PSNR and SSIM
                current_psnr = psnr(ground_truth_slice, filtered_test_slice)
                current_ssim = ssim(ground_truth_slice, filtered_test_slice, data_range=filtered_test_slice.max() - filtered_test_slice.min())
                print(f"PSNR: {current_psnr}, SSIM: {current_ssim}")

                psnr_values.append((h, patch_size, patch_distance, current_psnr))
                ssim_values.append((h, patch_size, patch_distance, current_ssim))
    
    return psnr_values, ssim_values

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
    
    h_range = np.arange(0.5, 2.1, 0.5)  # Range of h values to test
    patch_size_range = np.arange(2, 25, 2)  # Range of patch sizes to test
    patch_distance_range = np.arange(2, 25, 2)  # Range of patch distances to test
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

    for i, (gt_path, noisy_path, test_index) in enumerate(sample_paths):
        ground_truth = read_tiff_stack(gt_path)
        noisy_stack = read_tiff_stack(noisy_path)
        
        psnr_values, ssim_values = nlm_filter_3d_image_stack(noisy_stack, ground_truth, h_range, patch_size_range, patch_distance_range, test_index)
        
        all_psnr_scores.append(psnr_values)
        all_ssim_scores.append(ssim_values)
    
    # Aggregate PSNR and SSIM values over all samples
    psnr_dict = {}
    ssim_dict = {}
    for scores in all_psnr_scores:
        for h, patch_size, patch_distance, value in scores:
            psnr_dict[(h, patch_size, patch_distance)] = psnr_dict.get((h, patch_size, patch_distance), []) + [value]
    
    for scores in all_ssim_scores:
        for h, patch_size, patch_distance, value in scores:
            ssim_dict[(h, patch_size, patch_distance)] = ssim_dict.get((h, patch_size, patch_distance), []) + [value]

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

    print(f"Best overall parameters: h={best_params_overall[0]}, Patch Size={best_params_overall[1]}, Patch Distance={best_params_overall[2]}")

    # Prepare data for 2D heatmap plotting
    for h in h_range:
        psnr_heatmap = np.zeros((len(patch_size_range), len(patch_distance_range)))
        ssim_heatmap = np.zeros((len(patch_size_range), len(patch_distance_range)))
        
        for i, patch_size in enumerate(patch_size_range):
            for j, patch_distance in enumerate(patch_distance_range):
                psnr_heatmap[i, j] = avg_psnr_scores.get((h, patch_size, patch_distance), 0)
                ssim_heatmap[i, j] = avg_ssim_scores.get((h, patch_size, patch_distance), 0)
        
        # Plot PSNR heatmap
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(psnr_heatmap, extent=[patch_distance_range[0], patch_distance_range[-1], patch_size_range[0], patch_size_range[-1]], aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='PSNR')
        plt.title(f'PSNR Heatmap for h={h}')
        plt.xlabel('Patch Distance')
        plt.ylabel('Patch Size')
        
        # Plot SSIM heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(ssim_heatmap, extent=[patch_distance_range[0], patch_distance_range[-1], patch_size_range[0], patch_size_range[-1]], aspect='auto', origin='lower', cmap='plasma')
        plt.colorbar(label='SSIM')
        plt.title(f'SSIM Heatmap for h={h}')
        plt.xlabel('Patch Distance')
        plt.ylabel('Patch Size')
        
        plt.suptitle(f'PSNR and SSIM Heatmaps for h={h}')
        plt.tight_layout()
        plot_filename = f'nlm_psnr_ssim_heatmap_h_{h:.1f}.png'
        plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
        plt.show()

        print(f"Plot saved to {os.path.join(output_dir, plot_filename)}")
