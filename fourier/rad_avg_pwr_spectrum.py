import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
from scipy.fft import fft2 as scipy_fft2, fftshift

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

def radial_average(image):
    fft_image = np.log(np.abs(fftshift(scipy_fft2(image))) + 1).squeeze()
    center = np.array(fft_image.shape) / 2
    y, x = np.indices(fft_image.shape)
    rad = np.hypot(x - center[1], y - center[0])
    max_rad = int(rad.max())
    radial_prof = np.zeros(max_rad)
    for r in range(max_rad):
        mask = (rad >= r) & (rad < r + 1)
        radial_prof[r] = fft_image[mask].mean()
    return radial_prof

def calculate_radial_average_for_stacks(filepath):
    img_stack = read_tiff_stack(filepath)
    cropped_stack = crop_to_multiple_of_16(img_stack)
    radial_profiles = []
    for img in cropped_stack:
        radial_prof = radial_average(img)
        radial_profiles.append(radial_prof)
    return radial_profiles

def plot_radial_averages(all_radial_profiles, labels, output_dir):
    plt.figure(figsize=(10, 15))

    for radial_profiles, label in zip(all_radial_profiles, labels):
        mean_radial = np.mean(radial_profiles, axis=0)
        freq_range = np.arange(len(mean_radial))
        plt.plot(freq_range, mean_radial, label=label)

    plt.xlabel('Radius')
    plt.ylabel('Average Log Power Spectrum')
    plt.title('Radial Averaged Power Spectrum')
    plt.legend()
    plt.grid(True)

    plot_filename = 'radial_avg_power_spectrum.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"Radial averaged power spectrum plot saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\nema_avg_40.TIFF"
    denoised_files = [
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\nema_avg_40.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_gaussian_2.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_nlm_h1.4_ps4_pd20.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-Nema_B-test_3-Nematostella_B-epoch501.TIFF",
        r"C:\Users\rausc\Documents\EMBL\data\nema-results\output_stack-big_data_small-no_nema-no_droso-test_1-Nematostella_B-epoch547.TIFF"
        # Add more file paths as needed
    ]
    
    custom_labels = [
        "GT",
        "noisy",
        "gauss",
        "NLM",
        "BM3D",
        "single",
        "general"
        # Add more custom labels as needed
    ]
    
    if len(custom_labels) != len(denoised_files):
        raise ValueError("The number of custom labels must match the number of denoised files.")
    
    all_radial_profiles = []
    
    for denoised_stack_path in denoised_files:
        radial_profiles = calculate_radial_average_for_stacks(denoised_stack_path)
        all_radial_profiles.append(radial_profiles)

    plot_radial_averages(all_radial_profiles, custom_labels, output_dir)


