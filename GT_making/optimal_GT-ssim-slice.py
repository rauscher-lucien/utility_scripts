import os
import numpy as np
import tifffile
import re
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Natural sorting key function that converts numeric parts to integers
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', filename)]

# Function to read a specific slice from a TIFF file
def read_tiff_slice(filepath, slice_idx):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
        slice_image = images[slice_idx]
    return slice_image

# Function to compute the SSIM between ground truth and denoised slice
def calculate_ssim_for_slices(ground_truth_slice, denoised_slice):
    # Ensure the slices have the same dimensions
    assert ground_truth_slice.shape == denoised_slice.shape, "Slices must have the same dimensions."

    # Calculate SSIM
    score = ssim(ground_truth_slice, denoised_slice, data_range=65535)
    return score

# Function to update the running average
def update_running_average(running_avg, new_img, count):
    running_avg = (running_avg * (count - 1) + new_img) / count
    running_avg = np.clip(running_avg, 0, 65535)  # Ensure values are within the valid range
    return running_avg

# Main function to compute and plot SSIM for varying numbers of averaged slices
def main(input_folder, denoised_slice_path, slice_idx, max_slices, output_dir, base_filename):
    ssim_results = []

    # Read the denoised slice
    denoised_slice = read_tiff_slice(denoised_slice_path, slice_idx)

    # Apply natural sort on filenames
    file_list = sorted([f for f in os.listdir(input_folder) if f.endswith(".TIFF")], key=natural_sort_key)
    
    # Initialize the running average with the first slice
    first_slice = read_tiff_slice(os.path.join(input_folder, file_list[0]), slice_idx).astype(np.float64)
    running_avg = first_slice
    count = 1

    for num_slices in range(2, max_slices + 1):
        # Read the next slice and update the running average
        new_slice = read_tiff_slice(os.path.join(input_folder, file_list[num_slices - 1]), slice_idx).astype(np.float64)
        print(file_list[num_slices - 1])
        running_avg = update_running_average(running_avg, new_slice, num_slices)
        
        # Convert running average to uint16 for SSIM calculation
        averaged_slice_uint16 = running_avg.astype(np.uint16)
        
        # Calculate SSIM
        ssim_score = calculate_ssim_for_slices(averaged_slice_uint16, denoised_slice)
        ssim_results.append((num_slices, ssim_score))

    # Write SSIM results to a text file
    ssim_filename = os.path.join(output_dir, f'{base_filename}.txt')
    with open(ssim_filename, 'w') as f:
        f.write("Num_Slices\tSSIM\n")
        for num_slices, ssim_score in ssim_results:
            f.write(f"{num_slices}\t{ssim_score:.4f}\n")
    print(f"SSIM results saved to {ssim_filename}")

    # Plot the results
    num_slices_list, ssim_scores_list = zip(*ssim_results)
    plt.figure(figsize=(10, 6))
    plt.plot(num_slices_list, ssim_scores_list, 'o-')
    plt.xlabel('Number of Averaged Slices')
    plt.ylabel('SSIM')
    plt.title('SSIM vs Number of Averaged Slices')
    plt.grid(True)
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f'{base_filename}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f"SSIM plot saved to {plot_filename}")

if __name__ == "__main__":
    input_folder = r"C:\Users\rausc\Documents\EMBL\data\big_data\MouseEmbryo20230602LogScaleMouse_embyo_10hour"
    denoised_slice_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0_filtered_gaussian_5.TIFF"
    slice_idx = 250  # Specify the slice index to use
    max_slices = 40  # Specify the maximum number of slices to average
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\mouse-results"
    base_filename = 'optimal_GT-mouse-ssim-compared_to_G_5-slice_250-1'  # Specify the base filename for the output files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(input_folder, denoised_slice_path, slice_idx, max_slices, output_dir, base_filename)
