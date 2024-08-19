import numpy as np
import tifffile
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import random

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

def calculate_ssim_for_stacks(ground_truth_paths, denoised_stack_paths, sample_names, method_name):
    ssim_scores = {}

    for sample_idx, (ground_truth_path, denoised_stack_path) in enumerate(zip(ground_truth_paths, denoised_stack_paths)):
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

        assert cropped_ground_truth_stack_16.shape == cropped_denoised_stack_16.shape, "Cropped stacks must have the same dimensions."

        sample_name = sample_names[sample_idx]
        ssim_scores[(sample_name, method_name)] = []

        for i in range(cropped_ground_truth_stack_16.shape[0]):
            score, _ = ssim(cropped_ground_truth_stack_16[i], cropped_denoised_stack_16[i], data_range=65535, full=True)
            ssim_scores[(sample_name, method_name)].append(score)

    return ssim_scores

def plot_ssim_scores_boxplot_with_half_box_and_scatter(all_ssim_scores, labels, output_dir, plot_filename, sample_colors, sample_names, dpi=100, font_size=12, line_thickness=1.5):
    plt.figure(figsize=(15, 10))

    # Define the custom blue color
    blue_color = (0/255, 101/255, 189/255)  # Blue color
    black_color = 'black'
    
    positions = np.arange(len(labels))
    
    # Create the half box plot
    for i, label in enumerate(labels):
        ssim_scores = []
        for sample_name in sample_names:
            ssim_scores.extend(all_ssim_scores[(sample_name, label)])

        box = plt.boxplot(ssim_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False, 
                          boxprops=dict(linewidth=line_thickness, facecolor=blue_color),  # Set the facecolor to blue
                          medianprops=dict(linewidth=line_thickness, color=black_color),
                          whiskerprops=dict(linewidth=line_thickness, color=black_color),
                          capprops=dict(linewidth=line_thickness, color=black_color),
                          flierprops=dict(markeredgewidth=line_thickness, color=black_color))

    # Collect all SSIM scores with their colors
    all_points = []
    legend_handles = []

    for sample_idx, sample_name in enumerate(sample_names):
        for i, label in enumerate(labels):
            y_values = all_ssim_scores[(sample_name, label)]
            jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(y_values))
            for x, y in zip(jittered_x, y_values):
                all_points.append((x, y, sample_colors[sample_idx]))

        # Create a dummy plot for the legend
        handle = plt.scatter([], [], color=sample_colors[sample_idx], label=sample_name, s=50)
        legend_handles.append(handle)

    # Shuffle the points to randomize the plotting order
    random.shuffle(all_points)

    # Plot the shuffled points
    for x, y, color in all_points:
        plt.scatter(x, y, alpha=0.5, color=color, s=20)

    plt.xticks(ticks=positions, labels=labels, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel('SSIM Score', fontsize=font_size)
    plt.grid(True, linewidth=line_thickness)
    
    # Set the thickness of the plot spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(line_thickness)

    # Add the legend manually with the created handles
    plt.legend(handles=legend_handles, loc='lower right', fontsize=font_size)

    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"SSIM scores box plot with scatter saved to {plot_path}")
    
    # Save mean SSIM and standard deviation to a text file
    text_filename = plot_filename.replace('.png', '.txt')
    text_path = os.path.join(output_dir, text_filename)
    
    with open(text_path, 'w') as f:
        for label in labels:
            for sample_name in sample_names:
                ssim_scores = all_ssim_scores[(sample_name, label)]
                mean_ssim = np.mean(ssim_scores)
                std_ssim = np.std(ssim_scores)
                f.write(f"Label: {label}, Sample: {sample_name}\n")
                f.write(f"Mean SSIM: {mean_ssim}\n")
                f.write(f"Standard Deviation: {std_ssim}\n")
                f.write("\n")
    
    print(f"Mean SSIM and standard deviation details saved to {text_path}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\general-results"
    plot_filename = 'ssim_scores-single.png'

    # Define the ground truth files (same three files used for each group)
    ground_truth_paths = [
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B-average-100.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\Drosophila20210316LogScale01L_Good_Sample_02_t_-average-100-offset--2.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\mouse-results\MouseEmbryo20230602LogScaleMouse_embyo_10hour-average-20.TIFF"
    ]

    # Define groups of TIFF stacks and corresponding method labels
    denoised_files_grouped = [
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF"
        ], "noisy"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_bm3d_sigma_0.09.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1_filtered_bm3d_sigma_0.09.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0_filtered_bm3d_sigma_0.09.TIFF"
        ], "BM3D"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-N2V\output_stack-test_1_nema_model_nameUNet4_UNet_base16_num_epoch100000_batch_size8_lr1e-05_patience5000-nema-epoch13720-model_nameUNet4-UNet_base16-num_epoch100000-batch_size8-lr1e-05-patience5000.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-N2V\output_stack-test_1_droso_model_nameUNet4_UNet_base16_num_epoch100000_batch_size8_lr1e-05_patience5000-droso-epoch12020-model_nameUNet4-UNet_base16-num_epoch100000-batch_size8-lr1e-05-patience5000.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-N2V\output_stack-test_1_mouse_model_nameUNet4_UNet_base16_num_epoch100000_batch_size8_lr1e-05_patience5000-mouse-epoch4090-model_nameUNet4-UNet_base16-num_epoch100000-batch_size8-lr1e-05-patience5000.TIFF"
        ], "N2V"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet4_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch20230.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-single\2D-N2N-single_volume_output_stack-droso-project-test_1_droso_model_nameUNet4_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch97550.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-single\2D-N2N-single_volume_output_stack-mouse-project-test_1_mouse_model_nameUNet4_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch36420.TIFF"
        ], "2D-N2N"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-single\3D-N2N-single_volume_output_stack-nema-project-test_3_nema_model_nameUNet4_UNet_base32_stack_depth32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch98730.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-3D-N2N-single\3D-N2N-single_volume_output_stack-droso-project-test_1_droso_model_nameUNet4_UNet_base32_stack_depth32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch99330.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-3D-N2N-single\3D-N2N-single_volume_output_stack-mouse-project-test_3_mouse_model_nameUNet4_UNet_base32_stack_depth32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch63940.TIFF"
        ], "3D-N2N")
    ]

    custom_labels = [
        "noisy",
        "BM3D",
        "N2V",
        "2D-N2N",
        "3D-N2N"
    ]

    # Define the colors for each sample in the RGB format
    sample_colors = [
        (227/255, 114/255, 37/255),  # TUM orange
        (162/255, 173/255, 0/255),   # Orange for the second sample
        (218/255, 215/255, 203/255)  # Green for the third sample
    ]

    # Define the names of each sample for the legend
    sample_names = [
        "Nematostella",
        "Drosophila Embryo",
        "Mouse Embryo"
    ]

    if len(custom_labels) != len(denoised_files_grouped):
        raise ValueError("The number of custom labels must match the number of groups.")

    # Initialize a dictionary to store all SSIM scores in a structured manner
    all_ssim_scores = {}

    for denoised_group, method_name in denoised_files_grouped:
        ssim_scores = calculate_ssim_for_stacks(ground_truth_paths, denoised_group, sample_names, method_name)
        all_ssim_scores.update(ssim_scores)  # Add the calculated SSIM scores to the main dictionary

    # Now you can pass `all_ssim_scores` to the plotting function or do further analysis with it

    dpi = 300  # Adjust the DPI value as needed
    font_size = 24  # Adjust the font size as needed
    line_thickness = 2  # Adjust the line thickness as needed

    # Plot the SSIM scores
    plot_ssim_scores_boxplot_with_half_box_and_scatter(all_ssim_scores, custom_labels, output_dir, plot_filename, sample_colors, sample_names, dpi=dpi, font_size=font_size, line_thickness=line_thickness)
