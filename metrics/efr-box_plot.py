import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

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

def edge_selection(image):
    # Manually select an edge in the image
    # For simplicity, we select a vertical edge at the middle of the image
    h, w = image.shape
    mid_h = h // 2
    x = np.linspace(0, w-1, w)
    y = image[mid_h, :]
    return x, y

def erf_function(x, a, b, c, d):
    return a * erf(b + x * c) + d

def fit_edge(x, y):
    popt, _ = curve_fit(erf_function, x, y, p0=[1, 0, 1, 0])
    return popt

def calculate_efr(ground_truth_path, distorted_stack_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    distorted_stack = read_tiff_stack(distorted_stack_path).squeeze()

    if ground_truth_stack.shape[0] != distorted_stack.shape[0]:
        ground_truth_stack = ground_truth_stack[0:-1]  # Adjust the slice as necessary

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_distorted_stack = crop_to_multiple_of_16(distorted_stack)

    assert cropped_ground_truth_stack.shape == cropped_distorted_stack.shape, "Cropped stacks must have the same dimensions."

    efr_scores = []

    for i in range(cropped_ground_truth_stack.shape[0]):
        ref_image = cropped_ground_truth_stack[i]
        dis_image = cropped_distorted_stack[i]

        x_ref, y_ref = edge_selection(ref_image)
        x_dis, y_dis = edge_selection(dis_image)

        if len(x_ref) != len(x_dis):
            min_length = min(len(x_ref), len(x_dis))
            x_ref, y_ref = x_ref[:min_length], y_ref[:min_length]
            x_dis, y_dis = x_dis[:min_length], y_dis[:min_length]

        popt_ref = fit_edge(x_ref, y_ref)
        popt_dis = fit_edge(x_dis, y_dis)

        c_ref = popt_ref[2]
        c_dis = popt_dis[2]

        efr = c_dis / c_ref if c_ref != 0 else 0
        efr_scores.append(efr)

    return efr_scores

def plot_efr_scores_boxplot_with_half_box_and_scatter(all_efr_scores, labels, output_dir):
    plt.figure(figsize=(10, 15))
    
    positions = np.arange(len(all_efr_scores))
    
    # Create the half box plot
    for i, efr_scores in enumerate(all_efr_scores):
        box = plt.boxplot(efr_scores, positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, 
                          manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

    # Overlay the scatter plot with jitter
    for i, efr_scores in enumerate(all_efr_scores):
        jittered_x = np.random.normal(positions[i] + 0.2, 0.04, size=len(efr_scores))
        plt.scatter(jittered_x, efr_scores, alpha=0.5, color='red')

    plt.xticks(ticks=positions, labels=labels)
    plt.title('EFR Scores Box Plot with Scatter for Different Denoised Images')
    plt.ylabel('EFR Score')
    plt.grid(True)
    
    plot_filename = 'efr_scores-nema-compare_all_methods.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"EFR scores box plot with scatter saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\nema_avg_40.TIFF"
    distorted_files = [
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_gaussian_2.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_nlm_h1.4_ps4_pd20.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-Nema_B-test_3-Nematostella_B-epoch501.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-big_data_small-no_nema-no_droso-test_1-Nematostella_B-epoch547.TIFF"

        # Add more file paths as needed
    ]
    
    custom_labels = [
        "noisy",
        "gauss",
        "NLM",
        "BM3D",
        "single",
        "general"
        # Add more custom labels as needed
    ]
    
    if len(custom_labels) != len(distorted_files):
        raise ValueError("The number of custom labels must match the number of distorted files.")
    
    all_efr_scores = []
    
    for distorted_stack_path in distorted_files:
        efr_scores = calculate_efr(ground_truth_path, distorted_stack_path)
        all_efr_scores.append(efr_scores)

    plot_efr_scores_boxplot_with_half_box_and_scatter(all_efr_scores, custom_labels, output_dir)
