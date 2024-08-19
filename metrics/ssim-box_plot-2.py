import numpy as np
import tifffile
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def crop_to_multiple(img_stack, multiple):
    h, w = img_stack.shape[1:3]
    new_h = h - (h % multiple)
    new_w = w - (w % multiple)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img_stack[:, top:top+new_h, left:left+new_w]

def calculate_ssim_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    # Adjust ground truth stack depth if necessary
    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        if ground_truth_stack.shape[0] > denoised_stack.shape[0]:
            ground_truth_stack = ground_truth_stack[:denoised_stack.shape[0]]
        else:
            denoised_stack = denoised_stack[:ground_truth_stack.shape[0]]

    cropped_ground_truth_stack_16 = crop_to_multiple(ground_truth_stack, 16)
    cropped_denoised_stack_16 = crop_to_multiple(denoised_stack, 16)

    if cropped_ground_truth_stack_16.shape == cropped_denoised_stack_16.shape:
        cropped_ground_truth_stack = cropped_ground_truth_stack_16
        cropped_denoised_stack = cropped_denoised_stack_16
    else:
        cropped_ground_truth_stack = crop_to_multiple(ground_truth_stack, 32)
        cropped_denoised_stack = crop_to_multiple(denoised_stack, 32)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    ssim_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score, _ = ssim(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=65535, full=True)
        ssim_scores.append(score)

    return ssim_scores

def filter_outliers(ssim_scores, sensitivity=1.5):
    q1 = np.percentile(ssim_scores, 25)
    q3 = np.percentile(ssim_scores, 75)
    iqr = q3 - q1
    lower_bound = q1 - sensitivity * iqr
    upper_bound = q3 + sensitivity * iqr
    filtered_scores = [score for score in ssim_scores if lower_bound <= score <= upper_bound]
    return filtered_scores

def plot_ssim_scores_boxplot_with_half_box_and_scatter(all_ssim_scores, labels, output_dir, plot_filename, sensitivity=1.5, dpi=100, font_size=12, line_thickness=1.5):
    plt.figure(figsize=(15, 10))

    # Define the custom blue color
    blue_color = (0/255, 101/255, 189/255)
    black_color = 'black'
    
    positions = np.arange(len(all_ssim_scores))
    
    # Create the half box plot
    for i, ssim_scores in enumerate(all_ssim_scores):
        filtered_scores = filter_outliers(ssim_scores, sensitivity)
        box = plt.boxplot(filtered_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False, 
                          boxprops=dict(linewidth=line_thickness, facecolor=blue_color),
                          medianprops=dict(linewidth=line_thickness, color=black_color),
                          whiskerprops=dict(linewidth=line_thickness, color=black_color),
                          capprops=dict(linewidth=line_thickness, color=black_color),
                          flierprops=dict(markeredgewidth=line_thickness, color=blue_color))

    # Overlay the scatter plot with jitter
    for i, ssim_scores in enumerate(all_ssim_scores):
        filtered_scores = filter_outliers(ssim_scores, sensitivity)
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(filtered_scores))
        plt.scatter(jittered_x, filtered_scores, alpha=0.5, color=blue_color, s=60, zorder=2)

    plt.xticks(ticks=positions, labels=labels, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel('SSIM Score', fontsize=font_size)
    plt.grid(True, linewidth=line_thickness, zorder=1)
    
    # Set the thickness of the plot spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(line_thickness)

    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"SSIM scores box plot with scatter saved to {plot_path}")
    
    # Save mean SSIM and standard deviation to a text file
    text_filename = plot_filename.replace('.png', '.txt')
    text_path = os.path.join(output_dir, text_filename)
    
    with open(text_path, 'w') as f:
        for label, ssim_scores in zip(labels, all_ssim_scores):
            mean_ssim = np.mean(ssim_scores)
            std_ssim = np.std(ssim_scores)
            f.write(f"Label: {label}\n")
            f.write(f"Mean SSIM: {mean_ssim}\n")
            f.write(f"Standard Deviation: {std_ssim}\n")
            f.write("\n")
    
    print(f"Mean SSIM and standard deviation details saved to {text_path}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\mouse-results"
    plot_filename = 'ssim_scores-comp-single-1.png'
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\MouseEmbryo20230602LogScaleMouse_embyo_10hour-average-20.TIFF"
    denoised_files = [
        r"\\?\UNC\tier2.embl.de\prevedel\members\Rauscher\final_projects\2D-N2N-single_volume\test_3_nema_model_nameUNet4_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000\results\mouse\2D-N2N-single_volume_output_stack-mouse-project-test_3_nema_model_nameUNet4_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch20230.TIFF",
        r"\\?\UNC\tier2.embl.de\prevedel\members\Rauscher\final_projects\2D-N2N-single_volume\test_1_droso_model_nameUNet4_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000\results\mouse\2D-N2N-single_volume_output_stack-mouse-project-test_1_droso_model_nameUNet4_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch97550.TIFF",
        r"\\?\UNC\tier2.embl.de\prevedel\members\Rauscher\final_projects\2D-N2N-single_volume\test_1_mouse_model_nameUNet4_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000\results\mouse\2D-N2N-single_volume_output_stack-mouse-project-test_1_mouse_model_nameUNet4_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch36420.TIFF"
        # Add more file paths as needed
    ]
    
    custom_labels = [
        "mouse trained on nema",
        "mouse trained on droso",
        "mouse trained on mouse"
        # Add more custom labels as needed
    ]

    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_ssim_scores = []
    
    for denoised_stack_path in denoised_files:
        ssim_scores = calculate_ssim_for_stacks(ground_truth_path, denoised_stack_path)
        all_ssim_scores.append(ssim_scores)

    sensitivity = 1.5  # Adjust this value to change the outlier sensitivity
    dpi = 300  # Adjust the DPI value as needed
    font_size = 28  # Adjust the font size as needed
    line_thickness = 3  # Adjust the line thickness as needed
    plot_ssim_scores_boxplot_with_half_box_and_scatter(all_ssim_scores, custom_labels, output_dir, plot_filename, sensitivity, dpi, font_size, line_thickness)

