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

def plot_ssim_scores_boxplot_with_half_box_and_scatter(all_ssim_scores, labels, output_dir, sensitivity=1.5):
    plt.figure(figsize=(15, 10))
    
    positions = np.arange(len(all_ssim_scores))
    
    # Create the half box plot
    for i, ssim_scores in enumerate(all_ssim_scores):
        filtered_scores = filter_outliers(ssim_scores, sensitivity)
        box = plt.boxplot(filtered_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter
    for i, ssim_scores in enumerate(all_ssim_scores):
        filtered_scores = filter_outliers(ssim_scores, sensitivity)
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(filtered_scores))
        plt.scatter(jittered_x, filtered_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('SSIM Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('SSIM Score')
    plt.grid(True)
    
    plot_filename = 'ssim_scores-nema-2D-single-1.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
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
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B-average-100.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered.TIFF",
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
        "bm3d",
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

    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_ssim_scores = []
    
    for denoised_stack_path in denoised_files:
        ssim_scores = calculate_ssim_for_stacks(ground_truth_path, denoised_stack_path)
        all_ssim_scores.append(ssim_scores)

    sensitivity = 1.5  # Adjust this value to change the outlier sensitivity
    plot_ssim_scores_boxplot_with_half_box_and_scatter(all_ssim_scores, custom_labels, output_dir, sensitivity)
