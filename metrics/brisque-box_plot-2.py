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

def filter_outliers(scores, sensitivity=1.5):
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    lower_bound = q1 - sensitivity * iqr
    upper_bound = q3 + sensitivity * iqr
    filtered_scores = [score for score in scores if lower_bound <= score <= upper_bound]
    return filtered_scores

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

def plot_brisque_scores_boxplot_with_half_box_and_scatter(all_brisque_scores, labels, output_dir, sensitivity=1.5, dpi=100):
    plt.figure(figsize=(15, 10))
    
    positions = np.arange(len(all_brisque_scores))
    
    # Create the half box plot
    for i, brisque_scores in enumerate(all_brisque_scores):
        filtered_scores = filter_outliers(brisque_scores, sensitivity)
        box = plt.boxplot(filtered_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter
    for i, brisque_scores in enumerate(all_brisque_scores):
        filtered_scores = filter_outliers(brisque_scores, sensitivity)
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(filtered_scores))
        plt.scatter(jittered_x, filtered_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('BRISQUE Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('BRISQUE Score')
    plt.grid(True)
    
    plot_filename = 'brisque_scores-nema-general-1.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
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
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B-average-100.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_bm3d_sigma_0.09.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet4_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch506.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-general\3D-N2N-general_output_stack-nema-project-test_1_big_data_small_2_model_nameUNet4_UNet_base32_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch435.TIFF"
        # Add more file paths as needed
    ]
    
    custom_labels = [
        "BM3D 0.09",
        "2D-N2N general 4 32",
        "3D-N2N general 4 32"
        # Add more custom labels as needed
    ]

    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_brisque_scores = []
    
    for denoised_stack_path in denoised_files:
        brisque_scores = calculate_brisque_for_stacks(denoised_stack_path)
        all_brisque_scores.append(brisque_scores)

    sensitivity = 1.5  # Adjust this value to change the outlier sensitivity
    dpi = 300  # Adjust the DPI value as needed
    plot_brisque_scores_boxplot_with_half_box_and_scatter(all_brisque_scores, custom_labels, output_dir, sensitivity, dpi)

