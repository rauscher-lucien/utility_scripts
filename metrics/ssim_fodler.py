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

# Function to calculate SSIM scores between two TIFF stacks
def calculate_ssim_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    # Adjust the slice count to match both stacks
    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[:denoised_stack.shape[0]]

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    ssim_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score = ssim(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=65535)
        ssim_scores.append(score)

    return ssim_scores

# Function to plot SSIM scores and include the average SSIM in the title
def plot_ssim_scores(all_ssim_scores, labels, output_dir):
    plt.figure(figsize=(15, 7))
    for ssim_scores, label in zip(all_ssim_scores, labels):
        average_ssim = np.mean(ssim_scores)
        plt.plot(ssim_scores, marker='o', linestyle='-', label=f'{label} (Avg SSIM: {average_ssim:.2f})')
    
    plt.title('SSIM Scores per Image in Stack for Different Denoised Images')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM Score')
    plt.legend()
    plt.grid(True)

    plot_filename = 'ssim_scores_all.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    print(f"SSIM scores plot saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    # Paths for ground truth and denoised image stacks
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema_avg\nema_avg_40.TIFF"
    denoised_folder_path = r"C:\Users\rausc\Documents\EMBL\data\test"

    all_ssim_scores = []
    labels = []

    for filename in os.listdir(denoised_folder_path):
        if filename.endswith('.TIFF'):
            denoised_stack_path = os.path.join(denoised_folder_path, filename)
            ssim_scores = calculate_ssim_for_stacks(ground_truth_path, denoised_stack_path)
            all_ssim_scores.append(ssim_scores)
            labels.append(filename)

    output_dir = denoised_folder_path
    plot_ssim_scores(all_ssim_scores, labels, output_dir)
