import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import random

def read_tiff_stack(filepath):
    """
    Reads a TIFF file and returns it as a numpy array.
    """
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def crop_random_patch(image, patch_size=(64, 64)):
    """
    Crops a random patch of size `patch_size` from the image.
    """
    max_x = image.shape[0] - patch_size[0]
    max_y = image.shape[1] - patch_size[1]
    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)
    return image[start_x:start_x+patch_size[0], start_y:start_y+patch_size[1]]

def spectral_analysis(images, patch_size=(64, 64)):
    """
    Performs spectral analysis on randomly cropped patches from a stack of images and returns the average spectrum.
    """
    spectra = []
    for image in images:
        # Crop a random patch
        patch = crop_random_patch(image, patch_size)
        # Apply Fourier transform
        f_transform = np.fft.fftshift(np.fft.fft2(patch))
        # Compute spectrum and add to list
        spectrum = np.log(np.abs(f_transform) + 1)
        spectra.append(spectrum)
    # Calculate average spectrum
    avg_spectrum = np.mean(spectra, axis=0)
    return avg_spectrum

def plot_spectrum(spectrum):
    """
    Plots the spectrum with the origin (0,0 frequency) at the center of the plot.
    """
    # Assume spectrum is a square patch for simplicity in generating ticks
    n = spectrum.shape[0]  # Size of the spectrum
    extent = int(n // 2)  # Half the size, for setting bounds
    ticks = np.linspace(-extent, extent, num=5, dtype=int)  # Generate ticks for both axes
    
    plt.imshow(spectrum, cmap='hot', extent=[-extent, extent, -extent, extent])
    plt.colorbar()
    plt.title("Average Spectrum of Random Patches")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.show()

if __name__ == "__main__":
    filepath = os.path.join('Z:\\', 'members', 'Rauscher', 'data', 'big_data_small', 'TREC_val', 'Acantharia_G_5us_20240322_112348', 'Acantharia_G_5us_20240322_112348_T0.tiff')
    # filepath = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'projects', 'FastDVDNet')
    images = read_tiff_stack(filepath)
    avg_spectrum = spectral_analysis(images)
    plot_spectrum(avg_spectrum)
