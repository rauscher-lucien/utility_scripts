import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
from skimage.filters import sobel
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

def calculate_tenengrad(image):
    if image.ndim == 3:  # Convert to grayscale if the image is RGB
        image = rgb2gray(image)
    gradient_magnitude = sobel(image)
    sharpness = np.sum(gradient_magnitude)
    return sharpness

def calculate_tenengrad_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[0:-1]  # Adjust the slice as necessary

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    tenengrad_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        sharpness = calculate_tenengrad(cropped_denoised_stack[i])
        tenengrad_scores.append(sharpness)

    return tenengrad_scores

def plot_sharpness_scores_boxplot_with_half_box_and_scatter(all_sharpness_scores, labels, output_dir):
    plt.figure(figsize=(10, 15))
    
    positions = np.arange(len(all_sharpness_scores))
    
    # Create the half box plot
    for i, sharpness_scores in enumerate(all_sharpness_scores):
        box = plt.boxplot(sharpness_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter
    for i, sharpness_scores in enumerate(all_sharpness_scores):
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(sharpness_scores))
        plt.scatter(jittered_x, sharpness_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('Tenengrad Sharpness Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('Sharpness Score')
    plt.grid(True)
    
    plot_filename = 'tenengrad_sharpness_scores.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"Tenengrad sharpness scores box plot with scatter saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\nema_avg_40.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_gaussian_2.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_nlm_h1.4_ps4_pd20.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-Nema_B-test_3-Nematostella_B-epoch501.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-big_data_small-no_nema-no_droso-test_1-Nematostella_B-epoch547.TIFF"

        # Add more file paths as needed
    ]
    
    custom_labels = [
        "noisy",
        "gauss",
        "NLM",
        "BM3D",
        "single",
        "general"
        # Add more custom labels as needed
    ]
    
    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of distorted files.")
    
    all_sharpness_scores = []
    
    for denoised_stack_path in denoised_files:
        sharpness_scores = calculate_tenengrad_for_stacks(ground_truth_path, denoised_stack_path)
        all_sharpness_scores.append(sharpness_scores)

    plot_sharpness_scores_boxplot_with_half_box_and_scatter(all_sharpness_scores, custom_labels, output_dir)
