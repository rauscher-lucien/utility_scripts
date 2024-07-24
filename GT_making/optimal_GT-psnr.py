import os
import numpy as np
import tifffile
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# Natural sorting key function that converts numeric parts to integers
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', filename)]

# Function to read a TIFF file
def read_tiff(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

# Function to crop the images to a multiple of 16
def crop_to_multiple_of_16(img_stack):
    h, w = img_stack.shape[1:3]
    new_h = h - (h % 16)
    new_w = w - (w % 16)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img_stack[:, top:top+new_h, left:left+new_w]

# Function to compute the PSNR between ground truth and denoised stack
def calculate_psnr_for_stacks(ground_truth_stack, denoised_stack):
    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    psnr_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score = psnr(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=65535)
        psnr_scores.append(score)

    return psnr_scores

# Function to update the running average
def update_running_average(running_avg, new_img, count):
    running_avg = (running_avg * (count - 1) + new_img) / count
    running_avg = np.clip(running_avg, 0, 65535)  # Ensure values are within the valid range
    return running_avg

# Main function to compute and plot PSNR for varying numbers of averaged stacks
def main(input_folder, denoised_stack_path, max_stacks, output_dir, base_filename):
    psnr_results = []

    # Read the denoised stack
    denoised_stack = read_tiff(denoised_stack_path)

    # Apply natural sort on filenames
    file_list = sorted([f for f in os.listdir(input_folder) if f.endswith(".TIFF")], key=natural_sort_key)
    
    # Initialize the running average with the first stack
    first_stack = tifffile.imread(os.path.join(input_folder, file_list[0])).astype(np.float64)
    running_avg = first_stack
    count = 1

    for num_stacks in range(2, max_stacks + 1):
        # Read the next stack and update the running average
        new_stack = tifffile.imread(os.path.join(input_folder, file_list[num_stacks - 1])).astype(np.float64)
        print(file_list[num_stacks - 1])
        running_avg = update_running_average(running_avg, new_stack, num_stacks)
        
        # Convert running average to uint16 for PSNR calculation
        averaged_stack_uint16 = running_avg.astype(np.uint16)
        
        # Calculate PSNR
        psnr_scores = calculate_psnr_for_stacks(averaged_stack_uint16, denoised_stack)
        mean_psnr = np.mean(psnr_scores)
        std_psnr = np.std(psnr_scores)
        psnr_results.append((num_stacks, mean_psnr, std_psnr))

    # Write PSNR results to a text file
    psnr_filename = os.path.join(output_dir, f'{base_filename}.txt')
    with open(psnr_filename, 'w') as f:
        f.write("Num_Stacks\tMean_PSNR\tStd_PSNR\n")
        for num_stacks, mean_psnr, std_psnr in psnr_results:
            f.write(f"{num_stacks}\t{mean_psnr:.2f}\t{std_psnr:.2f}\n")
    print(f"PSNR results saved to {psnr_filename}")

    # Plot the results
    num_stacks_list, psnr_scores_list, psnr_std_list = zip(*psnr_results)
    plt.figure(figsize=(10, 6))
    plt.errorbar(num_stacks_list, psnr_scores_list, yerr=psnr_std_list, fmt='o', capsize=5)
    plt.xlabel('Number of Averaged Stacks')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs Number of Averaged Stacks')
    plt.grid(True)
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f'{base_filename}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f"PSNR plot saved to {plot_filename}")

if __name__ == "__main__":
    input_folder = r"C:\Users\rausc\Documents\EMBL\data\big_data\MouseEmbryo20230602LogScaleMouse_embyo_10hour"
    denoised_stack_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF"
    max_stacks = 100  # Specify the maximum number of stacks to average
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\mouse-results"
    base_filename = 'optimal_GT-mouse-psnr-2'  # Specify the base filename for the output files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(input_folder, denoised_stack_path, max_stacks, output_dir, base_filename)