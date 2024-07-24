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
    
    plot_filename = 'nrmse_scores-nema-final-1.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"NRMSE scores box plot with scatter saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\nema_avg_40.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\final-nema-results\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_1_Nematostella_B_model_nameUNet3_UNet_base16_num_epoch10000_batch_size8_lr1e-05-epoch9470.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\final-nema-results\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_1_Nematostella_B_model_nameUNet3_UNet_base32_num_epoch10000_batch_size8_lr1e-05-epoch9970.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\final-nema-results\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_1_Nematostella_B_model_nameUNet3_UNet_base64_num_epoch10000_batch_size8_lr1e-05-epoch9450.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\final-nema-results\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_1_Nematostella_B_model_nameUNet4_UNet_base16_num_epoch10000_batch_size8_lr1e-05-epoch9440.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\final-nema-results\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_1_Nematostella_B_model_nameUNet4_UNet_base32_num_epoch10000_batch_size8_lr1e-05-epoch9950.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\final-nema-results\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_1_Nematostella_B_model_nameUNet4_UNet_base64_num_epoch10000_batch_size8_lr1e-05-epoch9850.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\final-nema-results\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_1_Nematostella_B_model_nameUNet5_UNet_base16_num_epoch10000_batch_size8_lr1e-05-epoch7200.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\final-nema-results\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_1_Nematostella_B_model_nameUNet5_UNet_base32_num_epoch10000_batch_size8_lr1e-05-epoch5350.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\final-nema-results\2D-N2N-single_volume_output_stack-Nematostella_B-project-test_1_Nematostella_B_model_nameUNet5_UNet_base64_num_epoch10000_batch_size8_lr1e-05-epoch9750.TIFF",


        # Add more file paths as needed
    ]
    
    custom_labels = [
        "3 16",
        "3 32",
        "3 64",
        "4 16",
        "4 32",
        "4 64",
        "5 16",
        "5 32",
        "5 64"
        # Add more custom labels as needed
    ]
    
    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_nrmse_scores = []
    
    for denoised_stack_path in denoised_files:
        nrmse_scores = calculate_nrmse_for_stacks(ground_truth_path, denoised_stack_path)
        all_nrmse_scores.append(nrmse_scores)

    plot_nrmse_scores_boxplot_with_half_box_and_scatter(all_nrmse_scores, custom_labels, output_dir)


    # output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
    # ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\nema_avg_40.TIFF"
    # denoised_files = [
    #     r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_gaussian_2.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_nlm_h1.4_ps4_pd20.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-Nema_B-test_3-Nematostella_B-epoch501.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-big_data_small-no_nema-no_droso-test_1-Nematostella_B-epoch547.TIFF"

    #     # Add more file paths as needed
    # ]
    
    # custom_labels = [
    #     "noisy",
    #     "gauss",
    #     "NLM",
    #     "BM3D",
    #     "single",
    #     "general"
    #     # Add more custom labels as needed
    # ]


    # output_dir = r"C:\Users\rausc\Documents\EMBL\data\droso-results"
    # ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\droso_good_avg_40-offset-2.TIFF"
    # denoised_files = [
    #     r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1_filtered_gaussian_2.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1_filtered_nlm_h1.4_ps4_pd20.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1_filtered.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\droso-results\output_stack-droso_good-test_1-droso_good-epoch503.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\droso-results\output_stack-big_data_small-no_nema-no_droso-test_1-droso_good-epoch547.TIFF"


    #     # Add more file paths as needed
    # ]
    
    # custom_labels = [
    #     "noisy",
    #     "gauss",
    #     "nlm",
    #     "bm3d",
    #     "single",
    #     "general"
    #     # Add more custom labels as needed
    # ]



    # output_dir = r"C:\Users\rausc\Documents\EMBL\data\mouse-results"
    # ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour-average-10.TIFF"
    # denoised_files = [
    #     r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0_filtered_gaussian_2.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0_filtered_nlm_h1.4_ps4_pd20.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0_filtered.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\mouse-results\output_stack-mouse_embryo-test_1-mouse_embryo-epoch534.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\mouse-results\output_stack-big_data_small-no_nema-no_droso-test_1-mouse_embryo-epoch547.TIFF"


    #     # Add more file paths as needed
    # ]
    
    # custom_labels = [
    #     "noisy",
    #     "gauss",
    #     "nlm",
    #     "bm3d",
    #     "single",
    #     "general"
    #     # Add more custom labels as needed
    # ]