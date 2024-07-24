import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import brisque

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

def normalize_image(img):
    img_min, img_max = img.min(), img.max()
    img_normalized = (img - img_min) / (img_max - img_min)
    return img_normalized

def grayscale_to_rgb(img):
    return np.stack((img,) * 3, axis=-1)

def calculate_brisque_for_stacks(denoised_stack_path):
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()
    cropped_denoised_stack = crop_to_multiple(denoised_stack, 16)

    brisque_scores = []
    for i in range(cropped_denoised_stack.shape[0]):
        img = cropped_denoised_stack[i]
        img_normalized = normalize_image(img)
        img_ubyte = img_as_ubyte(img_normalized)
        img_rgb = grayscale_to_rgb(img_ubyte)
        score = brisque.BRISQUE().score(img_rgb)
        brisque_scores.append(score)

    return brisque_scores

def plot_brisque_scores_boxplot_with_half_box_and_scatter(all_brisque_scores, labels, output_dir):
    plt.figure(figsize=(15, 10))
    
    positions = np.arange(len(all_brisque_scores))
    
    # Create the half box plot
    for i, brisque_scores in enumerate(all_brisque_scores):
        box = plt.boxplot(brisque_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter
    for i, brisque_scores in enumerate(all_brisque_scores):
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(brisque_scores))
        plt.scatter(jittered_x, brisque_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('BRISQUE Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('BRISQUE Score')
    plt.grid(True)
    
    plot_filename = 'brisque_scores-nema-2D-single-1.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"BRISQUE scores box plot with scatter saved to {plot_path}")

    # Save mean BRISQUE and standard deviation to a text file
    text_filename = plot_filename.replace('.png', '.txt')
    text_path = os.path.join(output_dir, text_filename)
    
    with open(text_path, 'w') as f:
        for label, brisque_scores in zip(labels, all_brisque_scores):
            mean_brisque = np.mean(brisque_scores)
            std_brisque = np.std(brisque_scores)
            f.write(f"Label: {label}\n")
            f.write(f"Mean BRISQUE: {mean_brisque}\n")
            f.write(f"Standard Deviation: {std_brisque}\n")
            f.write("\n")
    
    print(f"Mean BRISQUE and standard deviation details saved to {text_path}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
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
    
    all_brisque_scores = []
    
    for denoised_stack_path in denoised_files:
        brisque_scores = calculate_brisque_for_stacks(denoised_stack_path)
        all_brisque_scores.append(brisque_scores)

    plot_brisque_scores_boxplot_with_half_box_and_scatter(all_brisque_scores, custom_labels, output_dir)
