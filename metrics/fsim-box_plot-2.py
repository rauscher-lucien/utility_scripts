import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.util import img_as_float
from scipy.ndimage import sobel

def phase_congruency(img, scales=4, orientations=4, k=2.0, epsilon=1e-4, sigma_f=0.55, mult=2.1, min_wavelength=3):
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

def crop_to_multiple(img_stack, multiple):
    h, w = img_stack.shape[1:3]
    new_h = h - (h % multiple)
    new_w = w - (w % multiple)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img_stack[:, top:top+new_h, left:left+new_w]

def calculate_fsim_for_stacks(ground_truth_path, denoised_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    denoised_stack = read_tiff_stack(denoised_stack_path).squeeze()

    # Adjust ground truth stack depth if necessary
    if ground_truth_stack.shape[0] != denoised_stack.shape[0]:
        if ground_truth_stack.shape[0] > denoised_stack.shape[0]:
            ground_truth_stack = ground_truth_stack[:denoised_stack.shape[0]]
        else:
            denoised_stack = denoised_stack[:ground_truth_stack.shape[0]]

    cropped_ground_truth_stack_16 = crop_to_multiple(ground_truth_stack, 16)
    cropped_denoised_stack_16 = crop_to_multiple(denoised_stack, 16)

    if cropped_ground_truth_stack_16.shape == cropped_denoised_stack_16.shape:
        cropped_ground_truth_stack = cropped_ground_truth_stack_16
        cropped_denoised_stack = cropped_denoised_stack_16
    else:
        cropped_ground_truth_stack = crop_to_multiple(ground_truth_stack, 32)
        cropped_denoised_stack = crop_to_multiple(denoised_stack, 32)

    assert cropped_ground_truth_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    fsim_scores = []
    for i in range(cropped_ground_truth_stack.shape[0]):
        score = fsim(cropped_ground_truth_stack[i], cropped_denoised_stack[i])
        fsim_scores.append(score)

    return fsim_scores

def filter_outliers(scores, sensitivity=1.5):
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    lower_bound = q1 - sensitivity * iqr
    upper_bound = q3 + sensitivity * iqr
    filtered_scores = [score for score in scores if lower_bound <= score <= upper_bound]
    return filtered_scores

def plot_fsim_scores_boxplot_with_half_box_and_scatter(all_fsim_scores, labels, output_dir, sensitivity=1.5):
    plt.figure(figsize=(15, 10))
    
    positions = np.arange(len(all_fsim_scores))
    
    # Create the half box plot
    for i, fsim_scores in enumerate(all_fsim_scores):
        filtered_scores = filter_outliers(fsim_scores, sensitivity)
        box = plt.boxplot(filtered_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter
    for i, fsim_scores in enumerate(all_fsim_scores):
        filtered_scores = filter_outliers(fsim_scores, sensitivity)
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(filtered_scores))
        plt.scatter(jittered_x, filtered_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('FSIM Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('FSIM Score')
    plt.grid(True)
    
    plot_filename = 'fsim_scores-nema-general-1.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"FSIM scores box plot with scatter saved to {plot_path}")

    # Save mean FSIM and standard deviation to a text file
    text_filename = plot_filename.replace('.png', '.txt')
    text_path = os.path.join(output_dir, text_filename)
    
    with open(text_path, 'w') as f:
        for label, fsim_scores in zip(labels, all_fsim_scores):
            mean_fsim = np.mean(fsim_scores)
            std_fsim = np.std(fsim_scores)
            f.write(f"Label: {label}\n")
            f.write(f"Mean FSIM: {mean_fsim}\n")
            f.write(f"Standard Deviation: {std_fsim}\n")
            f.write("\n")
    
    print(f"Mean FSIM and standard deviation details saved to {text_path}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B-average-100.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_bm3d_sigma_0.09.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-2D-N2N-general\2D-N2N-general_output_stack-Nematostella_B-project-test_2_big_data_small_2_model_nameUNet4_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch506.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results-3D-N2N-general\3D-N2N-general_output_stack-nema-project-test_1_big_data_small_2_model_nameUNet4_UNet_base32_stack_depth32_num_epoch1000_batch_size8_lr1e-05_patience50-epoch435.TIFF"
        # Add more file paths as needed
    ]
    
    custom_labels = [
        "BM3D 0.09",
        "2D-N2N general 4 32",
        "3D-N2N general 4 32"
        # Add more custom labels as needed
    ]

    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_fsim_scores = []
    
    for denoised_stack_path in denoised_files:
        fsim_scores = calculate_fsim_for_stacks(ground_truth_path, denoised_stack_path)
        all_fsim_scores.append(fsim_scores)

    sensitivity = 1.5  # Adjust this value to change the outlier sensitivity
    plot_fsim_scores_boxplot_with_half_box_and_scatter(all_fsim_scores, custom_labels, output_dir, sensitivity)
