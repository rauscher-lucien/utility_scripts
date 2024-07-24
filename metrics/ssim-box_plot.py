import numpy as np
import tifffile
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def crop_to_multiple_of_16(img_stack):
    h, w = img_stack.shape[1:3]
    new_h = h - (h % 16)
    new_w = w - (w % 16)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img_stack[:, top:top+new_h, left:left+new_w]

def calculate_ssim_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[0:-1]  # Adjust the slice as necessary

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    ssim_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score = ssim(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=65535)
        ssim_scores.append(score)

    return ssim_scores

def plot_ssim_scores_boxplot_with_half_box_and_scatter(all_ssim_scores, labels, output_dir):
    plt.figure(figsize=(10, 15))
    
    positions = np.arange(len(all_ssim_scores))
    
    # Create the half box plot
    for i, ssim_scores in enumerate(all_ssim_scores):
        box = plt.boxplot(ssim_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter
    for i, ssim_scores in enumerate(all_ssim_scores):
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(ssim_scores))
        plt.scatter(jittered_x, ssim_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('SSIM Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('SSIM Score (dB)')
    plt.grid(True)
    
    plot_filename = 'ssim_scores-mouse-compare_all_methods-1.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"SSIM scores box plot with scatter saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\mouse-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour-average-10.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0_filtered_gaussian_2.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0_filtered_nlm_h1.4_ps4_pd20.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0_filtered.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\mouse-results\output_stack-mouse_embryo-test_1-mouse_embryo-epoch534.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\mouse-results\output_stack-big_data_small-no_nema-no_droso-test_1-mouse_embryo-epoch547.TIFF"


        # Add more file paths as needed
    ]
    
    custom_labels = [
        "noisy",
        "gauss",
        "nlm",
        "bm3d",
        "single",
        "general"
        # Add more custom labels as needed
    ]
    
    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_ssim_scores = []
    
    for denoised_stack_path in denoised_files:
        psnr_scores = calculate_ssim_for_stacks(ground_truth_path, denoised_stack_path)
        all_ssim_scores.append(psnr_scores)

    plot_ssim_scores_boxplot_with_half_box_and_scatter(all_ssim_scores, custom_labels, output_dir)