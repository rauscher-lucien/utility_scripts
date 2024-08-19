import os
import numpy as np
import tifffile
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
        (r"C:\Users\rausc\Documents\EMBL\data\mouse-results\MouseEmbryo20230602LogScaleMouse_embyo_10hour-average-20.TIFF",
         r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF", 250)
        # Add more (ground_truth_path, noisy_file_path, test_index) tuples as needed
    ]
    
    h_range = np.arange(0.5, 2.0, 0.1)  # Range of h values to test
    patch_size_range = np.arange(2, 25, 2)  # Range of patch sizes to test
    patch_distance_range = np.arange(2, 25, 2)  # Range of patch distances to test
    custom_labels = [
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
