import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

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

def preprocess_stacks(ground_truth_path, noisy_image_path):
    ground_truth_stack = read_tiff_stack(ground_truth_path)
    noisy_image_stack = read_tiff_stack(noisy_image_path).squeeze()

    min_slices = min(ground_truth_stack.shape[0], noisy_image_stack.shape[0])
    ground_truth_stack = ground_truth_stack[:min_slices].squeeze()
    noisy_image_stack = noisy_image_stack[:min_slices].squeeze()

    cropped_ground_truth_stack = crop_to_multiple_of_16(ground_truth_stack)
    cropped_noisy_image_stack = crop_to_multiple_of_16(noisy_image_stack)

    assert cropped_ground_truth_stack.shape == cropped_noisy_image_stack.shape, "Cropped stacks must have the same dimensions."

    return cropped_ground_truth_stack, cropped_noisy_image_stack

def compute_fourier_spectrum(ground_truth_stack, noisy_image_stack, slice_index=None):
    if slice_index is not None:
        # Compute the Fourier spectrum for a single slice
        slice_noise = noisy_image_stack[slice_index] - ground_truth_stack[slice_index]
        f_transform = np.fft.fft2(slice_noise)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_transform_shifted)
        return magnitude_spectrum
    else:
        # Compute the average Fourier spectrum for the whole stack
        averaged_magnitude_spectrum = None
        for i in range(ground_truth_stack.shape[0]):
            slice_noise = noisy_image_stack[i] - ground_truth_stack[i]
            f_transform = np.fft.fft2(slice_noise)
            f_transform_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_transform_shifted)
            if averaged_magnitude_spectrum is None:
                averaged_magnitude_spectrum = magnitude_spectrum
            else:
                averaged_magnitude_spectrum += magnitude_spectrum

        averaged_magnitude_spectrum /= ground_truth_stack.shape[0]
        return averaged_magnitude_spectrum

def plot_and_save_fourier_spectrum(magnitude_spectrum, output_dir, species_name, ground_truth_filename, noisy_image_filename, slice_index=None, freq_range=None, log_scale=False, clip_range=None, save_plot=True):
    if log_scale:
        magnitude_spectrum = np.log1p(magnitude_spectrum)

    magnitude_spectrum -= magnitude_spectrum.min()
    magnitude_spectrum /= magnitude_spectrum.max()

    if clip_range:
        magnitude_spectrum = np.clip(magnitude_spectrum, clip_range[0], clip_range[1])

    freq_x = np.fft.fftfreq(magnitude_spectrum.shape[1])
    freq_y = np.fft.fftfreq(magnitude_spectrum.shape[0])
    freq_x = np.fft.fftshift(freq_x)
    freq_y = np.fft.fftshift(freq_y)

    extent = [freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]]
    if freq_range:
        freq_min, freq_max = freq_range
        x_mask = (freq_x >= freq_min) & (freq_x <= freq_max)
        y_mask = (freq_y >= freq_min) & (freq_y <= freq_max)
        magnitude_spectrum = magnitude_spectrum[np.ix_(y_mask, x_mask)]
        freq_x = freq_x[x_mask]
        freq_y = freq_y[y_mask]
        extent = [freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(magnitude_spectrum, cmap='gray', extent=extent, aspect='auto', interpolation='none')
    title_suffix = f" (Slice {slice_index})" if slice_index is not None else " (Average)"
    plt.title(f'{"Log " if log_scale else ""}FFT Magnitude Spectrum (Noise) - {species_name}{title_suffix}', fontsize=16)
    plt.xlabel('Frequency X-axis')
    plt.ylabel('Frequency Y-axis')
    plt.axis('on')

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Intensity')

    ground_truth_filename = os.path.basename(ground_truth_filename)
    noisy_image_filename = os.path.basename(noisy_image_filename)

    # Adjust the placement of the subtitle
    plt.figtext(0.5, 0.01, f"Ground truth: {ground_truth_filename}\nNoisy image: {noisy_image_filename}", ha="center", fontsize=10, wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_plot:
        freq_range_str = f'_{freq_min}_{freq_max}' if freq_range else ''
        clip_range_str = f'_clip{clip_range[0]}-{clip_range[1]}' if clip_range else ''
        slice_str = f'_slice{slice_index}' if slice_index is not None else '_average'
        plot_path_fft = os.path.join(output_dir, f'fourier_noise_analysis_{species_name}{slice_str}{freq_range_str}{clip_range_str}{"_log" if log_scale else ""}.png')
        plt.savefig(plot_path_fft, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot of FFT magnitude spectrum saved at {plot_path_fft}")
    else:
        plt.show()

def save_fourier_spectrum(ground_truth_path, noisy_image_path, species_name, slice_index=None, freq_range=None, log_scale=False, clip_range=None, save_plot=True):
    cropped_ground_truth_stack, cropped_noisy_image_stack = preprocess_stacks(ground_truth_path, noisy_image_path)
    magnitude_spectrum = compute_fourier_spectrum(cropped_ground_truth_stack, cropped_noisy_image_stack, slice_index)
    output_dir = os.path.dirname(ground_truth_path)
    plot_and_save_fourier_spectrum(magnitude_spectrum, output_dir, species_name, ground_truth_path, noisy_image_path, slice_index, freq_range, log_scale, clip_range, save_plot)

# Example usage
clean_image_path = r"Z:\members\Rauscher\projects\one_adj_slice\mouse_embryo-test_1\results\mouse_embryo\output_stack-mouse_embryo-test_1-mouse_embryo-epoch534.TIFF"
noisy_image_path = r"Z:\members\Rauscher\data\big_data_small\mouse_embryo\Mouse_embyo_10hour_V0.TIFF"
species_name = "Mouse Embryo"
log_scale = True  # Example to enable log scale
freq_range = None  # Example frequency range (can be None for full spectrum)
clip_range = (0.0, 0.4)  # Example clip range
slice_index = None  # Example slice index; set to None for average
save_fourier_spectrum(clean_image_path, noisy_image_path, species_name, slice_index=slice_index, freq_range=freq_range, log_scale=log_scale, clip_range=clip_range)


