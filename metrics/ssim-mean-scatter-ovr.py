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
        ssim_scores[(sample_name, method_name)] = []

        for i in range(cropped_ground_truth_stack.shape[0]):
            score = ssim(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=65535)
            ssim_scores[(sample_name, method_name)].append(score)

    return ssim_scores


def plot_ssim_scores_with_scatter_mean_std(all_ssim_scores, labels, output_dir, plot_filename, sample_colors, sample_names, dpi=100, font_size=12, line_thickness=1.5):
    plt.figure(figsize=(15, 10))

    positions = np.arange(len(labels))
    
    # Collect all SSIM scores with their colors
    all_points = []
    means = []
    stds = []
    legend_handles = []

    for i, label in enumerate(labels):
        # Collect all SSIM scores for this label across all samples
        y_values = []
        for sample_idx, sample_name in enumerate(sample_names):
            scores = all_ssim_scores[(sample_name, label)]
            y_values.extend(scores)
            jittered_x = np.random.normal(positions[i], 0.04, size=len(scores))
            for x, y in zip(jittered_x, scores):
                all_points.append((x, y, sample_colors[sample_idx]))

        # Calculate mean and standard deviation for the group
        mean_y = np.mean(y_values)
        std_y = np.std(y_values)
        means.append(mean_y)
        stds.append(std_y)

    # Shuffle the points to randomize the plotting order
    random.shuffle(all_points)

    # Plot the shuffled points as a scatter plot
    for x, y, color in all_points:
        plt.scatter(x, y, alpha=0.6, color=color, s=60)

    # Plot the mean points with error bars for the standard deviation
    plt.errorbar(
        positions, 
        means, 
        yerr=stds, 
        fmt='D',  # Diamond marker
        markerfacecolor=(0/255, 101/255, 189/255),  # Blue inside the diamond
        markeredgewidth=line_thickness,  # Set the marker edge thickness to match line_thickness
        markeredgecolor='black',  # Black border around the diamond
        ecolor='black',  # Error bars color
        capsize=10, 
        markersize=20, 
        elinewidth=line_thickness,  # Error bar line thickness
        capthick=line_thickness,  # Thickness of the cap lines at the end of whiskers
        zorder=4
    )

    # Create dummy plots for the legend
    for sample_idx, sample_name in enumerate(sample_names):
        handle = plt.scatter([], [], color=sample_colors[sample_idx], label=sample_name, s=50)
        legend_handles.append(handle)

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
    print(f"SSIM scores scatter plot with mean and std saved to {plot_path}")
    
    # Save mean SSIM and standard deviation to a text file
    text_filename = plot_filename.replace('.png', '.txt')
    text_path = os.path.join(output_dir, text_filename)
    
    with open(text_path, 'w') as f:
        for label, mean_y, std_y in zip(labels, means, stds):
            f.write(f"Label: {label}\n")
            f.write(f"Mean SSIM: {mean_y}\n")
            f.write(f"Standard Deviation: {std_y}\n")
            f.write("\n")
    
    print(f"Mean SSIM and standard deviation details saved to {text_path}")


if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\general-results"
    plot_filename = 'ssim_scores-general-comp-1.png'

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
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-nema-project-test_3_big_data_small_2_model_nameUNet3_UNet_base4_num_epoch1000_batch_size8_lr1e-05_patience50-epoch608.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_3_big_data_small_2_model_nameUNet3_UNet_base4_num_epoch1000_batch_size8_lr1e-05_patience50-epoch608.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_3_big_data_small_2_model_nameUNet3_UNet_base4_num_epoch1000_batch_size8_lr1e-05_patience50-epoch608.TIFF"
        ], "2D-N2N 3 4"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet3_UNet_base8_num_epoch1000_batch_size8_lr1e-05_patience50-epoch593.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_2_big_data_small_2_model_nameUNet3_UNet_base8_num_epoch1000_batch_size8_lr1e-05_patience50-epoch593.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_2_big_data_small_2_model_nameUNet3_UNet_base8_num_epoch1000_batch_size8_lr1e-05_patience50-epoch593.TIFF"
        ], "2D-N2N 3 8"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet3_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50-epoch350.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_2_big_data_small_2_model_nameUNet3_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50-epoch350.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_2_big_data_small_2_model_nameUNet3_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50-epoch350.TIFF"
        ], "2D-N2N 3 16"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet3_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch246.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_2_big_data_small_2_model_nameUNet3_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch246.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_2_big_data_small_2_model_nameUNet3_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch246.TIFF"
        ], "2D-N2N 3 32"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch387.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_2_big_data_small_2_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch387.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_2_big_data_small_2_model_nameUNet3_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch387.TIFF"
        ], "2D-N2N 3 64"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-nema-project-test_3_big_data_small_2_model_nameUNet4_UNet_base4_num_epoch1000_batch_size8_lr1e-05_patience50-epoch389.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_3_big_data_small_2_model_nameUNet4_UNet_base4_num_epoch1000_batch_size8_lr1e-05_patience50-epoch389.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_3_big_data_small_2_model_nameUNet4_UNet_base4_num_epoch1000_batch_size8_lr1e-05_patience50-epoch389.TIFF"
        ], "2D-N2N 4 4"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet4_UNet_base8_num_epoch1000_batch_size8_lr1e-05_patience50-epoch532.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_2_big_data_small_2_model_nameUNet4_UNet_base8_num_epoch1000_batch_size8_lr1e-05_patience50-epoch532.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_2_big_data_small_2_model_nameUNet4_UNet_base8_num_epoch1000_batch_size8_lr1e-05_patience50-epoch532.TIFF"
        ], "2D-N2N 4 8"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet4_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50-epoch268.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_2_big_data_small_2_model_nameUNet4_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50-epoch268.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_2_big_data_small_2_model_nameUNet4_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50-epoch268.TIFF"
        ], "2D-N2N 4 16"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet4_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch506.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_2_big_data_small_2_model_nameUNet4_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch506.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_2_big_data_small_2_model_nameUNet4_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch506.TIFF"
        ], "2D-N2N 4 32"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet4_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch553.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_2_big_data_small_2_model_nameUNet4_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch553.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_2_big_data_small_2_model_nameUNet4_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch553.TIFF"
        ], "2D-N2N 4 64"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-nema-project-test_3_big_data_small_2_model_nameUNet5_UNet_base4_num_epoch1000_batch_size8_lr1e-05_patience50-epoch613.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_3_big_data_small_2_model_nameUNet5_UNet_base4_num_epoch1000_batch_size8_lr1e-05_patience50-epoch613.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_3_big_data_small_2_model_nameUNet5_UNet_base4_num_epoch1000_batch_size8_lr1e-05_patience50-epoch613.TIFF"
        ], "2D-N2N 5 4"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-nema-project-test_3_big_data_small_2_model_nameUNet5_UNet_base8_num_epoch1000_batch_size8_lr1e-05_patience50-epoch424.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_3_big_data_small_2_model_nameUNet5_UNet_base8_num_epoch1000_batch_size8_lr1e-05_patience50-epoch424.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_3_big_data_small_2_model_nameUNet5_UNet_base8_num_epoch1000_batch_size8_lr1e-05_patience50-epoch424.TIFF"
        ], "2D-N2N 5 8"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet5_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50-epoch499.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_2_big_data_small_2_model_nameUNet5_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50-epoch499.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_3_big_data_small_2_model_nameUNet5_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50-epoch267.TIFF"
        ], "2D-N2N 5 16"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-nema-project-test_3_big_data_small_2_model_nameUNet5_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch230.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_3_big_data_small_2_model_nameUNet5_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch230.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_3_big_data_small_2_model_nameUNet5_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch230.TIFF"
        ], "2D-N2N 5 32"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet5_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch405.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-2D-N2N-general\2D-N2N-general_output_stack-droso-project-test_2_big_data_small_2_model_nameUNet5_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch405.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-2D-N2N-general\2D-N2N-general_output_stack-mouse-project-test_2_big_data_small_2_model_nameUNet5_UNet_base64_num_epoch1000_batch_size8_lr1e-05_patience50-epoch405.TIFF"
        ], "2D-N2N 5 64"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-general\3D-N2N-general_output_stack-nema-project-test_1_big_data_small_2_model_nameUNet3_UNet_base8_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch393.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-3D-N2N-general\3D-N2N-general_output_stack-droso-project-test_1_big_data_small_2_model_nameUNet3_UNet_base8_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch393.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-3D-N2N-general\3D-N2N-general_output_stack-mouse-project-test_1_big_data_small_2_model_nameUNet3_UNet_base8_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch393.TIFF"
        ], "3D-N2N 3 8"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-general\3D-N2N-general_output_stack-nema-project-test_1_big_data_small_2_model_nameUNet3_UNet_base16_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch384.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-3D-N2N-general\3D-N2N-general_output_stack-droso-project-test_1_big_data_small_2_model_nameUNet3_UNet_base16_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch384.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-3D-N2N-general\3D-N2N-general_output_stack-mouse-project-test_1_big_data_small_2_model_nameUNet3_UNet_base16_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch384.TIFF"
        ], "3D-N2N 3 16"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-general\3D-N2N-general_output_stack-nema-project-test_1_big_data_small_2_model_nameUNet3_UNet_base32_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch429.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-3D-N2N-general\3D-N2N-general_output_stack-droso-project-test_1_big_data_small_2_model_nameUNet3_UNet_base32_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch429.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-3D-N2N-general\3D-N2N-general_output_stack-mouse-project-test_1_big_data_small_2_model_nameUNet3_UNet_base32_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch429.TIFF"
        ], "3D-N2N 3 32"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-general\3D-N2N-general_output_stack-nema-project-test_1_big_data_small_2_model_nameUNet3_UNet_base64_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch350.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-3D-N2N-general\3D-N2N-general_output_stack-droso-project-test_1_big_data_small_2_model_nameUNet3_UNet_base64_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch350.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-3D-N2N-general\3D-N2N-general_output_stack-mouse-project-test_1_big_data_small_2_model_nameUNet3_UNet_base64_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch350.TIFF"
        ], "3D-N2N 3 64"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-general\3D-N2N-general_output_stack-nema-project-test_1_big_data_small_2_model_nameUNet4_UNet_base8_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch314.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-3D-N2N-general\3D-N2N-general_output_stack-droso-project-test_1_big_data_small_2_model_nameUNet4_UNet_base8_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch314.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-3D-N2N-general\3D-N2N-general_output_stack-mouse-project-test_1_big_data_small_2_model_nameUNet4_UNet_base8_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch314.TIFF"
        ], "3D-N2N 4 8"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-general\3D-N2N-general_output_stack-nema-project-test_1_big_data_small_2_model_nameUNet4_UNet_base16_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch307.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-3D-N2N-general\3D-N2N-general_output_stack-droso-project-test_1_big_data_small_2_model_nameUNet4_UNet_base16_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch307.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-3D-N2N-general\3D-N2N-general_output_stack-mouse-project-test_1_big_data_small_2_model_nameUNet4_UNet_base16_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch307.TIFF"
        ], "3D-N2N 4 16"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-general\3D-N2N-general_output_stack-nema-project-test_1_big_data_small_2_model_nameUNet4_UNet_base32_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch435.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-3D-N2N-general\3D-N2N-general_output_stack-droso-project-test_1_big_data_small_2_model_nameUNet4_UNet_base32_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch435.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-3D-N2N-general\3D-N2N-general_output_stack-mouse-project-test_1_big_data_small_2_model_nameUNet4_UNet_base32_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch435.TIFF"
        ], "3D-N2N 4 32"),
        ([
            r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-general\3D-N2N-general_output_stack-nema-project-test_1_big_data_small_2_model_nameUNet4_UNet_base64_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch288.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\droso-results-3D-N2N-general\3D-N2N-general_output_stack-droso-project-test_1_big_data_small_2_model_nameUNet4_UNet_base64_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch288.TIFF",
            r"C:\Users\rausc\Documents\EMBL\data\mouse-results-3D-N2N-general\3D-N2N-general_output_stack-mouse-project-test_1_big_data_small_2_model_nameUNet4_UNet_base64_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch288.TIFF"
        ], "3D-N2N 4 64")
    ]

    custom_labels = [
        "BM3D",
        "2D-N2N 3 4",
        "2D-N2N 3 8",
        "2D-N2N 3 16",
        "2D-N2N 3 32",
        "2D-N2N 3 64",
        "2D-N2N 4 4",
        "2D-N2N 4 8",
        "2D-N2N 4 16",
        "2D-N2N 4 32",
        "2D-N2N 4 64",
        "2D-N2N 5 4",
        "2D-N2N 5 8",
        "2D-N2N 5 16",
        "2D-N2N 5 32",
        "2D-N2N 5 64",
        "3D-N2N 3 8",
        "3D-N2N 3 16",
        "3D-N2N 3 32",
        "3D-N2N 3 64",
        "3D-N2N 4 8",
        "3D-N2N 4 16",
        "3D-N2N 4 32",
        "3D-N2N 4 64"
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

    # Initialize a dictionary to store all SSIM scores in a structured manner
    all_ssim_scores = {}

    for denoised_group, method_name in denoised_files_grouped:
        ssim_scores = calculate_ssim_for_stacks(ground_truth_paths, denoised_group, sample_names, method_name)
        all_ssim_scores.update(ssim_scores)  # Add the calculated SSIM scores to the main dictionary

    # Now you can pass `all_ssim_scores` to the plotting function or do further analysis with it

    dpi = 300  # Adjust the DPI value as needed
    font_size = 28  # Adjust the font size as needed
    line_thickness = 4  # Adjust the line thickness as needed

    # Call the modified plotting function to create a scatter plot with mean and std
    plot_ssim_scores_with_scatter_mean_std(all_ssim_scores, custom_labels, output_dir, plot_filename, sample_colors, sample_names, dpi=dpi, font_size=font_size, line_thickness=line_thickness)
