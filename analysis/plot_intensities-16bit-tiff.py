import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile


def scale_back(img):

    # convert to 0-1 range
    img = img / 65536
    # convert back to dB scale
    img = img * 100 - 50
    # convert back to 16bit
    img = np.power(10, img)

    return img

def plot_intensity_distribution(file_paths, labels, z_range=None, x_range=None, y_range=None, bins=65536, log_scale=False, save_path=None, legend_fontsize=10, plot_type='line'):
    plt.figure(figsize=(10, 5))
    
    for file_path, label in zip(file_paths, labels):
        filename = os.path.basename(file_path)
        
        if filename.lower().endswith(('.tif', '.tiff')):
            # For 16-bit TIFF images
            image = tifffile.imread(file_path).squeeze()
        
        else:
            print(f"Skipping file {filename} as it is not a 16-bit TIFF file.")
            continue
        
        # Apply x and y range limits if specified
        if z_range:
            z_min, z_max = z_range
        else:
            z_min, z_max = 0, image.shape[0]

        if x_range:
            x_min, x_max = x_range
        else:
            x_min, x_max = 0, image.shape[1]
        
        if y_range:
            y_min, y_max = y_range
        else:
            y_min, y_max = 0, image.shape[2]
        
        # Crop the image based on the specified ranges
        cropped_image = image[z_min:z_max, y_min:y_max, x_min:x_max]

        # Use numpy.histogram to bin the pixel intensity data
        intensity_values, bin_edges = np.histogram(cropped_image, bins=bins, range=(0, 65535))
        # Calculate bin centers from edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if plot_type == 'line':
            plt.plot(bin_centers, intensity_values, label=label)
        elif plot_type == 'histogram':
            plt.hist(cropped_image.ravel(), bins=bins, range=(0, 65536), density=False, alpha=0.5, label=label)

    plt.title('Pixel Intensity Distribution for 16-bit TIFF Image Files')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)

    if log_scale:
        plt.yscale('log')  # Set y-axis to log scale and handle non-positive values
        plt.xscale('log')

    # Save the figure
    if save_path:
        plt.savefig(os.path.join(save_path, 'intensity_distribution-inside-line-1.png'), format='png', dpi=300)
        plt.close()  # Close the figure to free up memory
    else:
        plt.show()

# Example usage with a list of file paths, x and y range, log scale and legend font size
file_paths = [
    r"C:\Users\rausc\Documents\EMBL\data\test_2\droso_good_avg_40-offset-2.TIFF",
    r"C:\Users\rausc\Documents\EMBL\data\test_2\Good_Sample_02_t_1.TIFF",
    r"C:\Users\rausc\Documents\EMBL\data\test_2\output_stack-big_data_small-no_nema-no_droso-test_1-droso_good-epoch547.TIFF"
    # Add more file paths as needed
]

labels = [
    "GT",
    "noisy",
    "denoised"
    # Add more labels corresponding to the file paths
]

z_range = (61, 141) #(227, 358)
x_range = (72, 164)#(0, 70)#(138, 341)  # Specify x range as (min, max)
y_range = (48, 136)#(0, 70)#(60, 113)  # Specify y range as (min, max)
bins = 1000  # Specify number of bins to cover the full range of 16-bit values
save_path = r"C:\Users\rausc\Documents\EMBL\data\test_2"
plot_type = 'line'  # 'line' for line plot, 'histogram' for histogram

plot_intensity_distribution(file_paths, labels, z_range=z_range, x_range=x_range, y_range=y_range, bins=bins, log_scale=False, save_path=save_path, legend_fontsize=12, plot_type=plot_type)


