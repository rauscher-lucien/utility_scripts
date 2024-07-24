import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.util import img_as_float
from scipy.ndimage import sobel

def phase_congruency(img, scales=4, orientations=4, k=2.0, epsilon=1e-4, sigma_f=0.55, mult=2.1, min_wavelength=3):
    # This is a simplified version of phase congruency calculation
    img = img_as_float(img)
    if img.ndim == 3:
        img = rgb2gray(img)
    fimg = np.fft.fft2(img)
    rows, cols = img.shape
    y, x = np.mgrid[-rows // 2:rows // 2, -cols // 2:cols // 2]
    radius = np.sqrt(x**2 + y**2)
    radius[rows // 2, cols // 2] = 1
    log_gabor = np.exp(-(np.log(radius / sigma_f))**2 / (2 * np.log(mult)**2))
    log_gabor[radius < min_wavelength] = 0
    log_gabor = np.fft.ifftshift(log_gabor)
    filtered = np.fft.ifft2(fimg * log_gabor)
    return np.abs(filtered)

def gradient_magnitude(img):
    img = img_as_float(img)
    dx = sobel(img, axis=0)
    dy = sobel(img, axis=1)
    grad_mag = np.hypot(dx, dy)
    return grad_mag

def fsim(img1, img2):
    pc1 = phase_congruency(img1)
    pc2 = phase_congruency(img2)
    gm1 = gradient_magnitude(img1)
    gm2 = gradient_magnitude(img2)
    epsilon = 1e-4

    pc_max = np.maximum(pc1, pc2)
    gm_max = np.maximum(gm1, gm2)
    sim = (2 * pc1 * pc2 + epsilon) / (pc1**2 + pc2**2 + epsilon) * (2 * gm1 * gm2 + epsilon) / (gm1**2 + gm2**2 + epsilon)
    fsim_index = np.sum(sim * pc_max * gm_max) / np.sum(pc_max * gm_max)
    return fsim_index

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

def calculate_fsim_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[0:-1]  # Adjust the slice as necessary

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    fsim_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score = fsim(cropped_ground_truth_stack[i], cropped_denoised_stack[i])
        fsim_scores.append(score)

    return fsim_scores

def plot_fsim_scores_boxplot_with_half_box_and_scatter(all_fsim_scores, labels, output_dir):
    plt.figure(figsize=(10, 15))  # Increased the height to make the plot longer vertically
    
    positions = np.arange(len(all_fsim_scores))
    
    # Create the half box plot
    for i, fsim_scores in enumerate(all_fsim_scores):
        box = plt.boxplot(fsim_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter
    for i, fsim_scores in enumerate(all_fsim_scores):
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(fsim_scores))
        plt.scatter(jittered_x, fsim_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('FSIM Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('FSIM Score')
    plt.grid(True)
    
    plot_filename = 'fsim_scores-droso-compare-all_methods-1.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"FSIM scores box plot with scatter saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\droso-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\droso_good_avg_40-offset-2.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1_filtered_gaussian_2.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1_filtered_nlm_h1.4_ps4_pd20.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1_filtered.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\output_stack-droso_good-test_1-droso_good-epoch503.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\droso-results\output_stack-big_data_small-no_nema-no_droso-test_1-droso_good-epoch547.TIFF"


        # Add more file paths as needed
    ]
    
    custom_labels = [
        "noisy",
        "gauss",
        "nlm",
        "bm3d",
        "single",
        "general"
        # Add more custom labels as needed
    ]
    
    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_fsim_scores = []
    
    for denoised_stack_path in denoised_files:
        fsim_scores = calculate_fsim_for_stacks(ground_truth_path, denoised_stack_path)
        all_fsim_scores.append(fsim_scores)

    plot_fsim_scores_boxplot_with_half_box_and_scatter(all_fsim_scores, custom_labels, output_dir)
