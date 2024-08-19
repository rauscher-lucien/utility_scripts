import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def nlm_filter_3d_image_stack(noisy_stack, ground_truth, h_range, patch_size, patch_distance, test_index):
    if test_index < 0 or test_index >= noisy_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = noisy_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth[test_index].astype(np.float64) / 65535.0
    
    psnr_values = []
    ssim_values = []
    
    for h in h_range:
        print(f"Testing h={h}")
        
        # Estimate the noise standard deviation from the test slice
        sigma_est = np.mean(estimate_sigma(test_slice))
        
        # Apply NLM filter to the test slice
        filtered_test_slice = denoise_nl_means(test_slice, h=h*sigma_est, fast_mode=True,
                                               patch_size=patch_size, patch_distance=patch_distance)
        
        # Calculate PSNR and SSIM
        current_psnr = psnr(ground_truth_slice, filtered_test_slice)
        current_ssim = ssim(ground_truth_slice, filtered_test_slice, data_range=filtered_test_slice.max() - filtered_test_slice.min())
        print(f"PSNR: {current_psnr}, SSIM: {current_ssim}")

        psnr_values.append(current_psnr)
        ssim_values.append(current_ssim)
    
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
    
    h_range = np.arange(0.5, 2.0, 0.1)  # Range of h values to test
    patch_size = 8  # Fixed patch size
    patch_distance = 20  # Fixed patch distance
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
        
        psnr_values, ssim_values = nlm_filter_3d_image_stack(noisy_stack, ground_truth, h_range, patch_size, patch_distance, test_index)
        
        all_psnr_scores.append(psnr_values)
        all_ssim_scores.append(ssim_values)
    
    # Compute average PSNR and SSIM over all samples for each h value
    avg_psnr_scores = np.mean(all_psnr_scores, axis=0)
    avg_ssim_scores = np.mean(all_ssim_scores, axis=0)
    
    # Normalize scores to 0-1 range
    min_psnr = np.min(avg_psnr_scores)
    max_psnr = np.max(avg_psnr_scores)
    min_ssim = np.min(avg_ssim_scores)
    max_ssim = np.max(avg_ssim_scores)
    
    normalized_psnr_scores = (avg_psnr_scores - min_psnr) / (max_psnr - min_psnr)
    normalized_ssim_scores = (avg_ssim_scores - min_ssim) / (max_ssim - min_ssim)
    
    # Compute combined scores and find best h value
    combined_scores = (normalized_psnr_scores + normalized_ssim_scores) / 2
    best_h_overall = h_range[np.argmax(combined_scores)]

    print(f"Best overall h value: {best_h_overall}")
    
    # Plot average PSNR and SSIM over all samples with separate y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('h')
    ax1.set_ylabel('Average PSNR', color='tab:blue')
    ax1.plot(h_range, avg_psnr_scores, label='Average PSNR', marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Average SSIM', color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(h_range, avg_ssim_scores, label='Average SSIM', marker='s', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Average PSNR and SSIM over all samples')
    plt.grid(True)

    plot_filename = 'nlm_h_psnr_ssim_comparison.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.show()

    print(f"Plot saved to {os.path.join(output_dir, plot_filename)}")
