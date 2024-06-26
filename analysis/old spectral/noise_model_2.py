import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

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

def generate_noise_model_plot(noisy_stack_path, denoised_stack_path, sample_rate=0.01):
    noisy_stack = read_tiff_stack(noisy_stack_path)
    denoised_stack = read_tiff_stack(denoised_stack_path)

    if noisy_stack.shape[0] != denoised_stack.shape[0]:
        noisy_stack = noisy_stack[:-1]
    
    cropped_noisy_stack = crop_to_multiple_of_16(noisy_stack)
    cropped_denoised_stack = crop_to_multiple_of_16(denoised_stack).squeeze()
    
    assert cropped_noisy_stack.shape == cropped_denoised_stack.shape, "Cropped stacks must have the same dimensions."

    total_pixels_cropped = cropped_noisy_stack.size
    sample_size = int(total_pixels_cropped * sample_rate)
    sampled_indices = np.random.choice(total_pixels_cropped, size=sample_size, replace=False)

    noisy_sample = cropped_noisy_stack.flat[sampled_indices]
    denoised_sample = cropped_denoised_stack.flat[sampled_indices]
    
    weights = np.ones_like(noisy_sample) / sample_size  # Normalize histogram to sum to 1

    plt.figure(figsize=(10, 10))
    plt.hist2d(noisy_sample, denoised_sample, bins=[np.linspace(0, 65535, 51), np.linspace(0, 65535, 51)],
               cmap='plasma', norm=colors.LogNorm(), weights=weights)
    plt.colorbar(label='Log Density')
    plt.title('Noise Model: Original vs. Denoised Pixel Values')
    plt.xlabel('Original Noisy Pixel Values')
    plt.ylabel('Denoised Pixel Values')
    plt.grid(True)
    
    output_dir = os.path.dirname(denoised_stack_path)
    plot_filename = 'noise_model_2d_histogram.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    print(f"Noise model 2D histogram saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == "__main__":
    path_to_noisy_image = r"C:\Users\rausc\Documents\EMBL\data\test_3\Mouse_embyo_10hour_V0.TIFF"
    path_to_denoised_image = r"C:\Users\rausc\Documents\EMBL\data\test_3\output_stack-mouse_embryo-test_1-mouse_embryo-epoch534.TIFF"

    generate_noise_model_plot(path_to_noisy_image, path_to_denoised_image, sample_rate=0.01)


