import numpy as np
import tifffile
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
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

def calculate_psnr_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[0:-1]  # Adjust the slice as necessary

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    psnr_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score = psnr(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=65535)
        psnr_scores.append(score)

    return psnr_scores

def plot_psnr_scores(all_psnr_scores, labels, output_dir):
    plt.figure(figsize=(15, 7))
    for psnr_scores, label in zip(all_psnr_scores, labels):
        average_psnr = np.mean(psnr_scores)
        plt.plot(psnr_scores, marker='o', linestyle='-', label=f'{label} (Avg PSNR: {average_psnr:.2f} dB)')
    
    plt.title('PSNR Scores per Image in Stack for Different Denoised Images')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR Score (dB)')
    plt.legend()
    plt.grid(True)
    
    plot_filename = 'psnr_scores_all.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    print(f"PSNR scores plot saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema_avg\nema_avg_40.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\Nematostella_B\Nematostella_B_V0.TIFF",
        r"Z:\members\Rauscher\projects\one_adj_slice\Nema_B-rednet10-test_1\results\Nematostella_B\output_stack-Nema_B-rednet-test_1-Nematostella_B-epoch499.TIFF",
        r"Z:\members\Rauscher\projects\one_adj_slice\Nema_B-rednet20-test_1\results\Nematostella_B\output_stack-Nema_B-rednet20-test_1-Nematostella_B-epoch262.TIFF",
        r"Z:\members\Rauscher\projects\one_adj_slice\Nema_B-rednet30-test_1\results\Nematostella_B\output_stack-Nema_B-rednet30-test_1-Nematostella_B-epoch577.TIFF",
        # Add more file paths as needed
    ]
    
    all_psnr_scores = []
    labels = []
    
    for denoised_stack_path in denoised_files:
        psnr_scores = calculate_psnr_for_stacks(ground_truth_path, denoised_stack_path)
        all_psnr_scores.append(psnr_scores)
        labels.append(os.path.basename(denoised_stack_path))
    
    output_dir = os.path.dirname(denoised_files[0])
    plot_psnr_scores(all_psnr_scores, labels, output_dir)

