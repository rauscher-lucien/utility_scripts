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

def generate_noise_model_plot(noisy_stack_path, denoised_stack_path, output_dir, sample_rate=0.01):
    noisy_stack = read_tiff_stack(noisy_stack_path)
    denoised_stack = read_tiff_stack(denoised_stack_path)
    
    # Crop both stacks
    cropped_noisy_stack = crop_to_multiple_of_16(noisy_stack[2:-2])
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack)
    
    # Ensure the cropped stacks are the same size for sampling
    assert cropped_noisy_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    # Flatten and sample pixels based on the cropped stack size
    total_pixels_cropped = cropped_noisy_stack.size
    sample_size = int(total_pixels_cropped * sample_rate)
    sampled_indices = np.random.choice(total_pixels_cropped, size=sample_size, replace=False)

    noisy_sample = cropped_noisy_stack.flat[sampled_indices]
    denoised_sample = cropped_denoised_stack.flat[sampled_indices]
    
    # Generate and save scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(noisy_sample, denoised_sample, alpha=0.1, edgecolors='none')
    plt.title('Noise Model: Original vs. Denoised Pixel Values')
    plt.xlabel('Original Noisy Pixel Values')
    plt.ylabel('Denoised Pixel Values')
    plt.grid(True)
    plot_filename = 'noise_model_plot.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()  # Ensure the plot is closed after saving
    print(f"Noise model plot saved to {os.path.join(output_dir, plot_filename)}")


if __name__ == "__main__":
    path_to_noisy_image = r"Z:\members\Rauscher\data\big_data_small\TREC_val\Acantharia_G_5us_20240322_112348\Acantharia_G_5us_20240322_112348_T0.TIFF"
    path_to_denoised_image = r"Z:\members\Rauscher\projects\FastDVDNet\TREC-test_2\results\inference_220-Acantharia_G_5us_20240322_112348\output_stack-TREC-test_2-inference_220-Acantharia_G_5us_20240322_112348.TIFF"
    output_directory = r"Z:\members\Rauscher\projects\FastDVDNet\TREC-test_2\results\inference_220-Acantharia_G_5us_20240322_112348"

    generate_noise_model_plot(path_to_noisy_image, path_to_denoised_image, output_directory, sample_rate=0.01)

