import numpy as np
import tifffile
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
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

def calculate_psnr_for_stacks(ground_truth_path, denoised_stack_path):
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

    psnr_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score = psnr(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=65535)
        psnr_scores.append(score)

    return psnr_scores

def filter_outliers(psnr_scores, sensitivity=1.5):
    q1 = np.percentile(psnr_scores, 25)
    q3 = np.percentile(psnr_scores, 75)
    iqr = q3 - q1
    lower_bound = q1 - sensitivity * iqr
    upper_bound = q3 + sensitivity * iqr
    filtered_scores = [score for score in psnr_scores if lower_bound <= score <= upper_bound]
    return filtered_scores

def plot_psnr_scores_boxplot_with_half_box_and_scatter(all_psnr_scores, labels, colors, output_dir, sensitivity=1.5, dpi=100, font_size=12):
    plt.figure(figsize=(15, 10))
    
    positions = np.arange(len(all_psnr_scores))
    
    # Create the half box plot
    for i, (psnr_scores, color) in enumerate(zip(all_psnr_scores, colors)):
        filtered_scores = filter_outliers(psnr_scores, sensitivity)
        box = plt.boxplot(filtered_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor(color)

    # Overlay the scatter plot with jitter
    for i, (psnr_scores, color) in enumerate(zip(all_psnr_scores, colors)):
        filtered_scores = filter_outliers(psnr_scores, sensitivity)
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(filtered_scores))
        plt.scatter(jittered_x, filtered_scores, alpha=0.5, color=color)

    plt.xticks(ticks=positions, labels=labels, fontsize=font_size)
    plt.title('PSNR Scores Box Plot with Scatter for Different Denoised Images', fontsize=font_size)
    plt.ylabel('PSNR Score (dB)', fontsize=font_size)
    plt.grid(True)
    
    plot_filename = 'psnr_scores-nema-2D-single-3.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"PSNR scores box plot with scatter saved to {plot_path}")
    
    # Save mean PSNR and standard deviation to a text file
    text_filename = plot_filename.replace('.png', '.txt')
    text_path = os.path.join(output_dir, text_filename)
    
    with open(text_path, 'w') as f:
        for label, psnr_scores in zip(labels, all_psnr_scores):
            mean_psnr = np.mean(psnr_scores)
            std_psnr = np.std(psnr_scores)
            f.write(f"Label: {label}\n")
            f.write(f"Mean PSNR: {mean_psnr}\n")
            f.write(f"Standard Deviation: {std_psnr}\n")
            f.write("\n")
    
    print(f"Mean PSNR and standard deviation details saved to {text_path}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B-average-100.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet3_UNet_base4_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch8290.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet3_UNet_base8_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch9000.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet3_UNet_base16_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch14620.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet3_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch70980.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet3_UNet_base64_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch99970.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet4_UNet_base4_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch5680.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet4_UNet_base8_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch27050.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet4_UNet_base16_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch21330.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet4_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch20230.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet4_UNet_base64_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch98420.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet5_UNet_base4_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch14360.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet5_UNet_base8_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch10090.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet5_UNet_base16_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch8030.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet5_UNet_base32_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch2050.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-single\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_3_nema_model_nameUNet5_UNet_base64_num_epoch100000_batch_size8_lr1e-05_patience5000-epoch62710.TIFF"
        # Add more file paths as needed
    ]
    
    custom_labels = [
        "3 4",
        "3 8",
        "3 16",
        "3 32",
        "3 64",
        "4 4",
        "4 8",
        "4 16",
        "4 32",
        "4 64",
        "5 4",
        "5 8",
        "5 16",
        "5 32",
        "5 64"
        # Add more custom labels as needed
    ]
    
    custom_colors = [
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "green",
        "green",
        "green",
        "green",
        "green",
        "red",
        "red",
        "red",
        "red",
        "red"
        # Add more custom colors as needed
    ]
    
    if len(custom_labels) != len(denoised_files) or len(custom_labels) != len(custom_colors):
        raise ValueError("The number of custom labels and custom colors must match the number of denoised files.")
    
    all_psnr_scores = []
    
    for denoised_stack_path in denoised_files:
        psnr_scores = calculate_psnr_for_stacks(ground_truth_path, denoised_stack_path)
        all_psnr_scores.append(psnr_scores)

    sensitivity = 1.5  # Adjust this value to change the outlier sensitivity
    dpi = 300  # Adjust the DPI value as needed
    font_size = 20  # Set the font size for the plot
    plot_psnr_scores_boxplot_with_half_box_and_scatter(all_psnr_scores, custom_labels, custom_colors, output_dir, sensitivity, dpi, font_size)


