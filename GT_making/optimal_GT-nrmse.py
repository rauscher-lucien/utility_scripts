import os
import numpy as np
import tifffile
import re
from skimage.metrics import normalized_root_mse as nrmse
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

# Function to compute the NRMSE between ground truth and averaged stack
def calculate_nrmse_for_stacks(noisy_stack, averaged_stack):
    cropped_noisy_stack = crop_to_multiple_of_16(noisy_stack)
    cropped_averaged_stack = crop_to_multiple_of_16(averaged_stack)

    assert cropped_noisy_stack.shape == cropped_averaged_stack.shape, "Cropped stacks must have the same dimensions."

    nrmse_scores = []
    for i in range(cropped_noisy_stack.shape[0]):
        score = nrmse(cropped_noisy_stack[i], cropped_averaged_stack[i])
        nrmse_scores.append(score)

    return nrmse_scores

# Function to update the running average
def update_running_average(running_avg, new_img, count):
    running_avg = (running_avg * (count - 1) + new_img) / count
    running_avg = np.clip(running_avg, 0, 65535)  # Ensure values are within the valid range
    return running_avg

# Main function to compute and plot NRMSE for varying numbers of averaged stacks
def main(input_folder, max_stacks, output_dir, base_filename):
    nrmse_results = []

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
        
        # Convert running average to uint16 for NRMSE calculation
        averaged_stack_uint16 = running_avg.astype(np.uint16)
        
        # Calculate NRMSE
        nrmse_scores = calculate_nrmse_for_stacks(first_stack, averaged_stack_uint16)
        mean_nrmse = np.mean(nrmse_scores)
        std_nrmse = np.std(nrmse_scores)
        nrmse_results.append((num_stacks, mean_nrmse, std_nrmse))

    # Write NRMSE results to a text file
    nrmse_filename = os.path.join(output_dir, f'{base_filename}.txt')
    with open(nrmse_filename, 'w') as f:
        f.write("Num_Stacks\tMean_NRMSE\tStd_NRMSE\n")
        for num_stacks, mean_nrmse, std_nrmse in nrmse_results:
            f.write(f"{num_stacks}\t{mean_nrmse:.4f}\t{std_nrmse:.4f}\n")
    print(f"NRMSE results saved to {nrmse_filename}")

    # Plot the results
    num_stacks_list, nrmse_scores_list, nrmse_std_list = zip(*nrmse_results)
    plt.figure(figsize=(10, 6))
    plt.errorbar(num_stacks_list, nrmse_scores_list, yerr=nrmse_std_list, fmt='o', capsize=5)
    plt.xlabel('Number of Averaged Stacks')
    plt.ylabel('NRMSE')
    plt.title('NRMSE vs Number of Averaged Stacks')
    plt.grid(True)
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f'{base_filename}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f"NRMSE plot saved to {plot_filename}")

if __name__ == "__main__":
    input_folder = r"\\tier2.embl.de\prevedel\members\Wang\Data\Mouse\Embryo\20230602\LogScale\Mouse_embyo_10hour"
    max_stacks = 20  # Specify the maximum number of stacks to average
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\mouse-results"
    base_filename = 'optimal_GT-mouse-nrmse-1'  # Specify the base filename for the output files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(input_folder, max_stacks, output_dir, base_filename)
