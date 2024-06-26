import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.fft import fft2 as scipy_fft2, fftshift
from skimage.draw import disk

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

def calculate_fourier_ring_correlation(img1, img2):
    fft1 = fftshift(scipy_fft2(img1))
    fft2 = fftshift(scipy_fft2(img2))
    
    conj_product = np.real(fft1 * np.conj(fft2))
    power1 = np.abs(fft1) ** 2
    power2 = np.abs(fft2) ** 2

    rad = np.hypot(*np.indices(img1.shape) - np.array(img1.shape)[:, None, None] / 2)
    max_rad = int(rad.max())
    frc = np.zeros(max_rad)

    for r in range(max_rad):
        mask = (rad >= r) & (rad < r + 1)
        frc[r] = np.sum(conj_product[mask]) / (np.sqrt(np.sum(power1[mask]) * np.sum(power2[mask])) + 1e-10)
    
    return frc

def calculate_frc_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[0:-1]  # Adjust the slice as necessary

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    frc_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        frc = calculate_fourier_ring_correlation(cropped_ground_truth_stack[i], cropped_denoised_stack[i])
        frc_scores.append(frc)

    return frc_scores

def plot_frc_curves(all_frc_scores, labels, output_dir, min_spatial_freq=0):
    plt.figure(figsize=(10, 15))

    for frc_scores, label in zip(all_frc_scores, labels):
        mean_frc = np.mean(frc_scores, axis=0)
        max_freq = len(mean_frc)
        freq_range = np.arange(max_freq)
        plt.plot(freq_range[min_spatial_freq:], mean_frc[min_spatial_freq:], label=label)

    plt.xlabel('Spatial Frequency')
    plt.ylabel('FRC')
    plt.title('Fourier Ring Correlation (FRC) Curves')
    plt.legend()
    plt.grid(True)

    plot_filename = 'frc_curves-nema-final.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"FRC curves plot saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\droso-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\droso_good_avg_40-offset-2.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1_filtered.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\output_stack-droso_good-test_1-droso_good-epoch503.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\output_stack-big_data_small-no_nema-no_droso-test_1-droso_good-epoch547.TIFF"
    ]
    
    custom_labels = [
        "noisy",
        "bm3d",
        "single",
        "general"
    ]
    
    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_frc_scores = []
    
    for denoised_stack_path in denoised_files:
        frc_scores = calculate_frc_for_stacks(ground_truth_path, denoised_stack_path)
        all_frc_scores.append(frc_scores)

    plot_frc_curves(all_frc_scores, custom_labels, output_dir, min_spatial_freq=50)





    # output_dir = r"C:\Users\rausc\Documents\EMBL\data\mouse-results"
    # ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour-average-10.TIFF"
    # denoised_files = [
    #     r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0_filtered.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\mouse-results\output_stack-mouse_embryo-test_1-mouse_embryo-epoch534.TIFF",
    #     r"C:\Users\rausc\Documents\EMBL\data\mouse-results-generalized\output_stack-big_data_small-no_nema-no_droso-test_1-mouse_embryo-epoch547.TIFF"
    # ]
    
    # custom_labels = [
    #     "noisy",
    #     "bm3d",
    #     "single",
    #     "general"
    # ]