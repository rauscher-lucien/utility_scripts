import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

def read_tiff_stack(filepath):
    """
    Reads a TIFF file and returns it as a numpy array.
    """
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def spectral_analysis(images):
    """
    Performs spectral analysis on a stack of images and returns the average spectrum.
    """
    spectra = []
    for image in images:
        # Apply Fourier transform
        f_transform = np.fft.fftshift(np.fft.fft2(image))
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
    plt.title("Average Spectrum")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.show()

if __name__ == "__main__":
    filepath = os.path.join('Z:\\', 'members', 'Rauscher', 'data', 'big_data_small', 'good_sample_unidentified', 'good_sample_unidentified.tif')
    images = read_tiff_stack(filepath)
    avg_spectrum = spectral_analysis(images)
    plot_spectrum(avg_spectrum)
