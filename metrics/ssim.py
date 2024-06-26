import os
import numpy as np
import tifffile
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Function to read a TIFF stack from a file
def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

# Function to crop an image stack to dimensions that are a multiple of 16
def crop_to_multiple_of_16(img_stack):
    h, w = img_stack.shape[1:3]
    new_h = h - (h % 16)
    new_w = w - (w % 16)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img_stack[:, top:top + new_h, left: left + new_w]

# Function to normalize an image to a 0-1 range
def normalize_image(image):
    return image.astype(np.float32) / image.max()

# Function to calculate SSIM scores between two TIFF stacks
def calculate_ssim_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    ground_truth_stack = normalize_image(ground_truth_stack)
    denoised_stack = normalize_image(denoised_stack)

    # Adjust the slice count to match both stacks
    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[:denoised_stack.shape[0]]

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    ssim_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score = ssim(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=1.0)
        ssim_scores.append(score)

    return ssim_scores

# Function to plot SSIM scores and include the average SSIM in the title
def plot_ssim_scores(ssim_scores, output_dir):
    average_ssim = np.mean(ssim_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(ssim_scores, marker='o', linestyle='-')
    plt.title(f'SSIM Scores per Image in Stack (Average SSIM: {average_ssim:.2f})')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM Score')
    plt.grid(True)

    plot_filename = 'ssim_scores.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    print(f"SSIM scores plot saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    # Paths for ground truth and denoised image stacks
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema_avg\nema_avg_40.TIFF"
    denoised_stack_path = r"Z:\members\Rauscher\projects\one_adj_slice\Nema_B-test_2\results\Nematostella_B\output_stack-Nema_B-test_2-Nematostella_B-epoch499.TIFF"
    
    # Calculate SSIM scores
    ssim_scores = calculate_ssim_for_stacks(ground_truth_path, denoised_stack_path)
    output_dir = os.path.dirname(denoised_stack_path)
    
    # Plot SSIM scores and save the plot
    plot_ssim_scores(ssim_scores, output_dir)


