import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import os

def read_image(filepath):
    """Reads an image from a file."""
    return io.imread(filepath)

def crop_to_multiple_of_16(img):
    """Crops the image to the nearest multiple of 16 for both dimensions."""
    h, w = img.shape[:2]
    new_h = h - (h % 16)
    new_w = w - (w % 16)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img[top:top+new_h, left:left+new_w]

def calculate_psnr_for_images(ground_truth_path, denoised_image_path):
    """Calculates the PSNR between a ground truth image and a denoised image."""
    ground_truth_img = read_image(ground_truth_path)
    denoised_img = read_image(denoised_image_path)

    cropped_ground_truth_img = crop_to_multiple_of_16(ground_truth_img)
    cropped_denoised_img = crop_to_multiple_of_16(denoised_img)

    assert cropped_ground_truth_img.shape == cropped_denoised_img.shape, "Cropped images must have the same dimensions."

    score = psnr(cropped_ground_truth_img, cropped_denoised_img, data_range=255)
    return score

def plot_psnr_scores_bar_plot(all_psnr_scores, labels, output_dir):
    """Plots a bar plot of PSNR scores for various denoised images."""
    plt.figure(figsize=(10, 6))
    
    positions = np.arange(len(all_psnr_scores))
    
    plt.bar(positions, all_psnr_scores, color='lightblue')
    
    plt.xticks(ticks=positions, labels=labels)
    plt.title('PSNR Scores for Different Denoised Images')
    plt.xlabel('Denoised Image')
    plt.ylabel('PSNR Score (dB)')
    plt.grid(True, axis='y')
    
    plot_filename = 'psnr_scores-bm3d-cuda.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"PSNR scores bar plot saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema_avg_40-single_slices\image_80.png"
    denoised_files = [
        r"Z:\members\Rauscher\data\big_data_small-test\Nematostella_B\image_80_denoised_10.png",
        r"Z:\members\Rauscher\data\big_data_small-test\Nematostella_B\image_80_denoised_20.png",
        r"Z:\members\Rauscher\data\big_data_small-test\Nematostella_B\image_80_denoised_30.png",
        r"Z:\members\Rauscher\data\big_data_small-test\Nematostella_B\image_80_denoised_40.png",
        r"Z:\members\Rauscher\data\big_data_small-test\Nematostella_B\image_80_denoised_50.png",
        r"Z:\members\Rauscher\data\big_data_small-test\Nematostella_B\image_80_denoised_60.png",
        r"Z:\members\Rauscher\data\big_data_small-test\Nematostella_B\image_80_denoised_70.png",
        r"Z:\members\Rauscher\data\big_data_small-test\Nematostella_B\image_80_denoised_80.png",
        r"Z:\members\Rauscher\data\big_data_small-test\Nematostella_B\image_80_denoised_90.png",
        r"Z:\members\Rauscher\data\big_data_small-test\Nematostella_B\image_80_denoised_100.png"
    ]
    
    custom_labels = [
        "10",
        "20",
        "30",
        "40",
        "50",
        "60",
        "70",
        "80",
        "90",
        "100"
    ]
    
    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_psnr_scores = []
    
    for denoised_image_path in denoised_files:
        psnr_score = calculate_psnr_for_images(ground_truth_path, denoised_image_path)
        all_psnr_scores.append(psnr_score)

    plot_psnr_scores_bar_plot(all_psnr_scores, custom_labels, output_dir)

