import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_intensity_distribution(file_paths, x_range=None, y_range=None, bins=256, log_scale=False, save_path=None, legend_fontsize=10, plot_type='line'):
    plt.figure(figsize=(10, 5))
    
    line_labels = {}
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        print(f"Current file: {filename}")
        new_label = input("Enter the label for this line (or press Enter to use the filename): ").strip()
        line_labels[file_path] = new_label if new_label else filename
    
    for file_path, label in line_labels.items():
        filename = os.path.basename(file_path)
        
        # Open the image and ensure it's in grayscale mode
        image = Image.open(file_path).convert('L')
        image = np.array(image)
        
        # Apply x and y range limits if specified
        if x_range:
            x_min, x_max = x_range
        else:
            x_min, x_max = 0, image.shape[1]
        
        if y_range:
            y_min, y_max = y_range
        else:
            y_min, y_max = 0, image.shape[0]
        
        # Crop the image based on the specified ranges
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Use numpy.histogram to bin the pixel intensity data
        intensity_values, bin_edges = np.histogram(cropped_image, bins=bins, range=(0, 255))
        # Calculate bin centers from edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if plot_type == 'line':
            plt.plot(bin_centers, intensity_values, label=label)
        elif plot_type == 'histogram':
            plt.hist(cropped_image.ravel(), bins=bins, range=(0, 255), alpha=0.5, label=label)

    plt.title('Pixel Intensity Distribution for 8-bit Grayscale Images')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)

    if log_scale:
        plt.yscale('log')  # Set y-axis to log scale and handle non-positive values

    # Save the figure
    if save_path:
        plt.savefig(os.path.join(save_path, 'intensity_distribution-poisson-8bit.png'), format='png', dpi=300)
        plt.close()  # Close the figure to free up memory
    else:
        plt.show()

# Example usage with a list of file paths, x and y range, log scale and legend font size
file_paths = [
    r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\single_noisy\noised_12003_8bit_gray.jpg"
    # Add more file paths as needed
]
x_range = None  # Specify x range as (min, max)
y_range = None  # Specify y range as (min, max)
bins = 256  # Specify number of bins
save_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\single_noisy"
plot_type = 'histogram'  # 'line' for line plot, 'histogram' for histogram

plot_intensity_distribution(file_paths, x_range=x_range, y_range=y_range, bins=bins, log_scale=False, save_path=save_path, legend_fontsize=12, plot_type=plot_type)
