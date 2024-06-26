import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os

def read_tiff_stack(filepath):
    # Reads a TIFF file and returns it as a numpy array.
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def crop_to_multiple_of_16(img_stack):
    # Crops an image stack so that the height and width are multiples of 16.
    h, w = img_stack.shape[1:3]
    new_h = h - (h % 16)
    new_w = w - (w % 16)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img_stack[:, top:top+new_h, left:left+new_w]

def calculate_residuals(noisy_stack, denoised_stack):
    # Calculates the residuals between the noisy and denoised stacks.
    cropped_noisy_stack = crop_to_multiple_of_16(noisy_stack[2:-2])
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)
    return cropped_noisy_stack - cropped_denoised_stack

def analyze_residuals(residuals, output_dir):
    # Performs statistical analysis on the residuals and saves the results.
    mean_residual = np.mean(residuals)
    std_dev_residual = np.std(residuals)
    mad_residual = np.mean(np.abs(residuals - np.mean(residuals)))

    results = {
        'Mean of residuals': mean_residual,
        'Standard deviation of residuals': std_dev_residual,
        'Mean absolute deviation of residuals': mad_residual
    }

    # Save the results to a text file
    output_filepath = os.path.join(output_dir, 'residuals_analysis.txt')
    with open(output_filepath, 'w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")

    print(f"Results saved to {output_filepath}")

    # Plot and save normalized histogram of residuals
    plt.hist(residuals.ravel(), bins=50, color='blue', alpha=0.7, density=True)
    plt.title("Normalized Histogram of Residuals")
    plt.xlabel("Intensity")
    plt.ylabel("Probability Density")
    # Save plot as PNG
    plot_filename = 'normalized_residuals_histogram.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    print(f"Normalized histogram saved to {os.path.join(output_dir, plot_filename)}")
    plt.close()

    return results


if __name__ == "__main__":
    path_to_noisy_image = r"Z:\members\Rauscher\data\big_data_small\good_sample_unidentified\good_sample_unidentified.tif"
    path_to_denoised_image = r"Z:\members\Rauscher\projects\FastDVDNet\TREC-test_2\results\inference_220-good_sample_unidentified\output_stack-TREC-test_2-inference_220-good_sample_unidentified.TIFF"
    output_directory = r"Z:\members\Rauscher\projects\FastDVDNet\TREC-test_2\results\inference_220-good_sample_unidentified"

    noisy_stack = read_tiff_stack(path_to_noisy_image)
    denoised_stack = read_tiff_stack(path_to_denoised_image)

    if noisy_stack.shape[0] < 5:
        raise ValueError("The noisy stack is too small to crop!")

    residuals = calculate_residuals(noisy_stack, denoised_stack)
    results = analyze_residuals(residuals, output_directory)
