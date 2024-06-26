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

def normalize_image(image):
    # Normalize to 0-1 range based on the max value in the image
    return image.astype(np.float32) / image.max()

def calculate_psnr_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    ground_truth_stack = normalize_image(ground_truth_stack)
    denoised_stack = normalize_image(denoised_stack)

    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[0:-1]  # Adjust the slice as necessary

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    psnr_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        print(i)
        score = psnr(cropped_ground_truth_stack[i], cropped_denoised_stack[i], data_range=1.0)
        psnr_scores.append(score)

    return psnr_scores

def plot_psnr_scores(psnr_scores, output_dir):
    average_psnr = np.mean(psnr_scores)
    
    plt.figure(figsize=(10, 5))
    plt.plot(psnr_scores, marker='o', linestyle='-')
    plt.title(f'PSNR Scores per Image in Stack (Average PSNR: {average_psnr:.2f} dB)')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR Score (dB)')
    plt.grid(True)
    
    plot_filename = 'psnr_scores_2.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    print(f"PSNR scores plot saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema_avg\nema_avg_40.TIFF"
    denoised_stack_path = r"C:\Users\rausc\Documents\EMBL\data\Nematostella_B\Nematostella_B_V0.TIFF"
    psnr_scores = calculate_psnr_for_stacks(ground_truth_path, denoised_stack_path)
    output_dir = os.path.dirname(denoised_stack_path)
    plot_psnr_scores(psnr_scores, output_dir)


