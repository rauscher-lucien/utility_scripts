import numpy as np
import tifffile
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
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

def calculate_psnr_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[0:-1]  # Adjust the slice as necessary

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    psnr_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score = psnr(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=65535)
        psnr_scores.append(score)

    return psnr_scores

def plot_psnr_scores_boxplot_with_half_box_and_scatter(all_psnr_scores, labels, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 15))
    
    # Define the y-axis break limits
    break_points = [28, 34]

    positions = np.arange(len(all_psnr_scores))
    
    # Create the half box plot for ax1 (upper part)
    for i, psnr_scores in enumerate(all_psnr_scores):
        box = ax1.boxplot(psnr_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter for ax1 (upper part)
    for i, psnr_scores in enumerate(all_psnr_scores):
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(psnr_scores))
        ax1.scatter(jittered_x, psnr_scores, alpha=0.5, color='red')

    # Create the half box plot for ax2 (lower part)
    for i, psnr_scores in enumerate(all_psnr_scores):
        box = ax2.boxplot(psnr_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter for ax2 (lower part)
    for i, psnr_scores in enumerate(all_psnr_scores):
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(psnr_scores))
        ax2.scatter(jittered_x, psnr_scores, alpha=0.5, color='red')

    # Set y-axis limits for both axes
    ax1.set_ylim(break_points[1], max(max(all_psnr_scores)))
    ax2.set_ylim(min(min(all_psnr_scores)), break_points[0])

    # Hide the spines between ax1 and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    # Add diagonal lines to indicate the break
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.xticks(ticks=positions, labels=labels)
    fig.suptitle('PSNR Scores Box Plot with Scatter for Different Denoised Images')
    ax2.set_ylabel('PSNR Score (dB)')
    ax2.grid(True)
    
    plot_filename = 'psnr_scores-nema-compare_all_methods-break-1.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"PSNR scores box plot with scatter saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\nema_avg_40.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_gaussian_2.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_nlm_h1.4_ps4_pd20.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-Nema_B-test_3-Nematostella_B-epoch501.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-big_data_small-no_nema-no_droso-test_1-Nematostella_B-epoch547.TIFF"

        # Add more file paths as needed
    ]
    
    custom_labels = [
        "noisy",
        "gauss",
        "NLM",
        "BM3D",
        "single",
        "general"
        # Add more custom labels as needed
    ]
    
    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_psnr_scores = []
    
    for denoised_stack_path in denoised_files:
        psnr_scores = calculate_psnr_for_stacks(ground_truth_path, denoised_stack_path)
        all_psnr_scores.append(psnr_scores)

    plot_psnr_scores_boxplot_with_half_box_and_scatter(all_psnr_scores, custom_labels, output_dir)
