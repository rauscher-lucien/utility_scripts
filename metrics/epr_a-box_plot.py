import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.color import rgb2gray

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

def edge_detection(image, sigma=1.0):
    if image.ndim == 3:  # Convert to grayscale if the image is RGB
        image = rgb2gray(image)
    edges = canny(image, sigma=sigma)
    return edges

def calculate_epr(ground_truth_path, distorted_stack_path, sigma=1.0):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    distorted_stack = read_tiff_stack(distorted_stack_path).squeeze()

    if ground_truth_stack.shape[0] != distorted_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[0:-1]  # Adjust the slice as necessary

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_distorted_stack = crop_to_multiple_of_16(distorted_stack)

    assert cropped_ground_truth_stack.shape == cropped_distorted_stack.shape, "Cropped stacks must have the same dimensions."

    epr_a_scores = []

    for i in range(cropped_ground_truth_stack.shape[0]):
        reference_edges = edge_detection(cropped_ground_truth_stack[i], sigma=sigma)
        distorted_edges = edge_detection(cropped_distorted_stack[i], sigma=sigma)

        R = np.sum(reference_edges)
        R_and_D = np.sum(reference_edges & distorted_edges)

        epr_a = R_and_D / R if R != 0 else 0
        epr_a_scores.append(epr_a)

    return epr_a_scores

def plot_epr_a_scores_boxplot_with_half_box_and_scatter(all_epr_a_scores, labels, output_dir):
    plt.figure(figsize=(15, 10))
    positions = np.arange(len(all_epr_a_scores))
    
    for i, epr_scores in enumerate(all_epr_a_scores):
        box = plt.boxplot(epr_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    for i, epr_scores in enumerate(all_epr_a_scores):
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(epr_scores))
        plt.scatter(jittered_x, epr_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('EPRa Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('EPRa Score')
    plt.grid(True)
    
    plot_filename = 'epr_a_scores-nema-general-1.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"EPRa scores box plot with scatter saved to {plot_path}")
    
    # Save mean EPRa and standard deviation to a text file
    text_filename = plot_filename.replace('.png', '.txt')
    text_path = os.path.join(output_dir, text_filename)
    
    with open(text_path, 'w') as f:
        for label, epr_scores in zip(labels, all_epr_a_scores):
            mean_epr = np.mean(epr_scores)
            std_epr = np.std(epr_scores)
            f.write(f"Label: {label}\n")
            f.write(f"Mean EPRa: {mean_epr}\n")
            f.write(f"Standard Deviation: {std_epr}\n")
            f.write("\n")
    
    print(f"Mean EPRa and standard deviation details saved to {text_path}")

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
        raise ValueError("The number of custom labels must match the number of distorted files.")
    
    all_epr_a_scores = []
    
    for distorted_stack_path in denoised_files:
        epr_a_scores = calculate_epr(ground_truth_path, distorted_stack_path)
        all_epr_a_scores.append(epr_a_scores)

    plot_epr_a_scores_boxplot_with_half_box_and_scatter(all_epr_a_scores, custom_labels, output_dir)

