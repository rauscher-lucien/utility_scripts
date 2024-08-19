import numpy as np
import tifffile
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
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

def calculate_psnr_for_stacks(ground_truth_paths, denoised_stack_paths, sample_names, method_name):
    psnr_scores = {}

    for sample_idx, (ground_truth_path, denoised_stack_path) in enumerate(zip(ground_truth_paths, denoised_stack_paths)):
        ground_truth_stack = read_tiff_stack(ground_truth_path)
        denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

        # Adjust ground truth stack depth if necessary
        if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
            if ground_truth_stack.shape[0] > denoised_stack.shape[0]:
                ground_truth_stack = ground_truth_stack[:denoised_stack.shape[0]]
            else:
                denoised_stack = denoised_stack[:ground_truth_stack.shape[0]]

        # First attempt to crop to multiple of 16
        cropped_ground_truth_stack = crop_to_multiple(ground_truth_stack, 16)
        cropped_denoised_stack = crop_to_multiple(denoised_stack, 16)

        # If cropping to 16 doesn't result in the same dimensions, attempt cropping to 32
        if cropped_ground_truth_stack.shape != cropped_denoised_stack.shape:
            cropped_ground_truth_stack = crop_to_multiple(ground_truth_stack, 32)
            cropped_denoised_stack = crop_to_multiple(denoised_stack, 32)

        # Ensure the stacks have the same dimensions after cropping
        assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

        sample_name = sample_names[sample_idx]
        psnr_scores[(sample_name, method_name)] = []

        for i in range(cropped_ground_truth_stack.shape[0]):
            score = psnr(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=65535)
            psnr_scores[(sample_name, method_name)].append(score)

    return psnr_scores




def plot_psnr_scores_boxplot_with_half_box_and_scatter(all_psnr_scores, labels, output_dir, plot_filename, sample_colors, sample_names, dpi=100, font_size=12, line_thickness=1.5):
    plt.figure(figsize=(15, 10))

    # Define the custom blue color
    blue_color = (0/255, 101/255, 189/255)  # Blue color
    black_color = 'black'
    
    positions = np.arange(len(labels))
    
    # Create the half box plot
    for i, label in enumerate(labels):
        psnr_scores = []
        for sample_name in sample_names:
            psnr_scores.extend(all_psnr_scores[(sample_name, label)])

        box = plt.boxplot(psnr_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False, 
                          boxprops=dict(linewidth=line_thickness, facecolor=blue_color),  # Set the facecolor to blue
                          medianprops=dict(linewidth=line_thickness, color=black_color),
                          whiskerprops=dict(linewidth=line_thickness, color=black_color),
                          capprops=dict(linewidth=line_thickness, color=black_color),
                          flierprops=dict(markeredgewidth=line_thickness, color=black_color))

    # Collect all PSNR scores with their colors
    all_points = []
    legend_handles = []

    for sample_idx, sample_name in enumerate(sample_names):
        for i, label in enumerate(labels):
            y_values = all_psnr_scores[(sample_name, label)]
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
    plt.ylabel('PSNR Score (dB)', fontsize=font_size)
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
    print(f"PSNR scores box plot with scatter saved to {plot_path}")
    
    # Save mean PSNR and standard deviation to a text file
    text_filename = plot_filename.replace('.png', '.txt')
    text_path = os.path.join(output_dir, text_filename)
    
    with open(text_path, 'w') as f:
        for label in labels:
            for sample_name in sample_names:
                psnr_scores = all_psnr_scores[(sample_name, label)]
                mean_psnr = np.mean(psnr_scores)
                std_psnr = np.std(psnr_scores)
                f.write(f"Label: {label}, Sample: {sample_name}\n")
                f.write(f"Mean PSNR: {mean_psnr}\n")
                f.write(f"Standard Deviation: {std_psnr}\n")
                f.write("\n")
    
    print(f"Mean PSNR and standard deviation details saved to {text_path}")




if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\general-results"
    plot_filename = 'psnr_scores-specific-comp_datasets.png'

    # Define the ground truth files (same three files used for each group)
    ground_truth_paths = [
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B-average-100.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\Drosophila20210316LogScale01L_Good_Sample_02_t_-average-100-offset--2.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\mouse-results\MouseEmbryo20230602LogScaleMouse_embyo_10hour-average-20.TIFF"
    ]

    # Define groups of TIFF stacks and corresponding method labels
    denoised_files_grouped = [
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_bm3d_sigma_0.09.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1_filtered_bm3d_sigma_0.09.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0_filtered_bm3d_sigma_0.09.TIFF"
        ], "BM3D"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-only_nema\2D-N2N-general_output_stack-nema-project-test_1_big_data_small-only_nema_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch224.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-only_nema\2D-N2N-general_output_stack-droso-project-test_1_big_data_small-only_nema_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch224.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-only_nema\2D-N2N-general_output_stack-mouse-project-test_1_big_data_small-only_nema_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch224.TIFF"
        ], "Nematostella"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-only_droso\2D-N2N-general_output_stack-nema-project-test_1_big_data_small-only_droso_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch418.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-only_droso\2D-N2N-general_output_stack-droso-project-test_1_big_data_small-only_droso_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch418.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-only_droso\2D-N2N-general_output_stack-mouse-project-test_1_big_data_small-only_droso_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch418.TIFF"
        ], "Drosophila"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-only_mouse\2D-N2N-general_output_stack-nema-project-test_1_big_data_small-only_mouse_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch410.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-only_mouse\2D-N2N-general_output_stack-droso-project-test_1_big_data_small-only_mouse_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch410.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-only_mouse\2D-N2N-general_output_stack-mouse-project-test_1_big_data_small-only_mouse_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch410.TIFF"
        ], "Mouse")
    ]

    custom_labels = [
        "BM3D",
        "Nematostella",
        "Drosophila",
        "Mouse"
    ]

    # Define the colors for each sample in the RGB format
    sample_colors = [
        (227/255, 114/255, 37/255),  # tum orange
        (162/255, 173/255, 0/255),  # Orange for the second sample
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

    # Initialize a dictionary to store all PSNR scores in a structured manner
    all_psnr_scores = {}

    for denoised_group, method_name in denoised_files_grouped:
        psnr_scores = calculate_psnr_for_stacks(ground_truth_paths, denoised_group, sample_names, method_name)
        all_psnr_scores.update(psnr_scores)  # Add the calculated PSNR scores to the main dictionary

    # Now you can pass `all_psnr_scores` to the plotting function or do further analysis with it

    dpi = 300  # Adjust the DPI value as needed
    font_size = 24  # Adjust the font size as needed
    line_thickness = 2  # Adjust the line thickness as needed

    # You would modify your plotting function to accommodate this new structure
    plot_psnr_scores_boxplot_with_half_box_and_scatter(all_psnr_scores, custom_labels, output_dir, plot_filename, sample_colors, sample_names, dpi=dpi, font_size=font_size, line_thickness=line_thickness)
