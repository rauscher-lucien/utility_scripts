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
    plt.figure(figsize=(10, 15))
    
    positions = np.arange(len(all_psnr_scores))
    
    # Create the half box plot
    for i, psnr_scores in enumerate(all_psnr_scores):
        box = plt.boxplot(psnr_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter
    for i, psnr_scores in enumerate(all_psnr_scores):
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(psnr_scores))
        plt.scatter(jittered_x, psnr_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('PSNR Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('PSNR Score (dB)')
    plt.grid(True)
    
    plot_filename = 'psnr_scores-nema-final-single-all-5.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"PSNR scores box plot with scatter saved to {os.path.join(output_dir, plot_filename)}")

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
    
    all_psnr_scores = []
    
    for denoised_stack_path in denoised_files:
        psnr_scores = calculate_psnr_for_stacks(ground_truth_path, denoised_stack_path)
        all_psnr_scores.append(psnr_scores)

    plot_psnr_scores_boxplot_with_half_box_and_scatter(all_psnr_scores, custom_labels, output_dir)
