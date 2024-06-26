import numpy as np
import tifffile
import os
from skimage.metrics import normalized_root_mse as nrmse
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

def calculate_nrmse_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[0:-1]  # Adjust the slice as necessary

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    nrmse_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score = nrmse(cropped_ground_truth_stack[i], cropped_denoised_stack[i])
        nrmse_scores.append(score)

    return nrmse_scores

def plot_nrmse_scores_boxplot_with_half_box_and_scatter(all_nrmse_scores, labels, output_dir):
    plt.figure(figsize=(10, 15))  # Increased the height to make the plot longer vertically
    
    positions = np.arange(len(all_nrmse_scores))
    
    # Create the half box plot
    for i, nrmse_scores in enumerate(all_nrmse_scores):
        box = plt.boxplot(nrmse_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter
    for i, nrmse_scores in enumerate(all_nrmse_scores):
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(nrmse_scores))
        plt.scatter(jittered_x, nrmse_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('NRMSE Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('NRMSE Score')
    plt.grid(True)
    
    plot_filename = 'nrmse_scores-nema-gen.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"NRMSE scores box plot with scatter saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\test_1"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\test_1\nema_avg_40.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\test_1\Nematostella_B_V0.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\test_1\Nematostella_B_V0_filtered.TIFF",
        r"Z:\members\Rauscher\projects\one_adj_slice\big_data_small-no_nema-no_droso-test_1\results\Nematostella_B\output_stack-big_data_small-no_nema-no_droso-test_1-Nematostella_B-epoch547.TIFF",
        # Add more file paths as needed
    ]
    
    custom_labels = [
        "noisy",
        "filtered",
        "network",
        # Add more custom labels as needed
    ]
    
    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_nrmse_scores = []
    
    for denoised_stack_path in denoised_files:
        nrmse_scores = calculate_nrmse_for_stacks(ground_truth_path, denoised_stack_path)
        all_nrmse_scores.append(nrmse_scores)

    plot_nrmse_scores_boxplot_with_half_box_and_scatter(all_nrmse_scores, custom_labels, output_dir)
