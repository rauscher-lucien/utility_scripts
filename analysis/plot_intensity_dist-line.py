import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

def plot_intensity_distribution_multiple_tiffs(file_paths, labels, slice_index, coord_index, direction='horizontal', save_plot=True, save_folder=None, custom_name=""):
    if len(file_paths) != len(labels):
        raise ValueError("The number of file paths must match the number of labels.")

    plt.figure(figsize=(10, 5))

    for tiff_path, label in zip(file_paths, labels):
        tiff_stack = tifffile.imread(tiff_path)

        # Validate slice index and coordinate index
        if slice_index < 0 or slice_index >= tiff_stack.shape[0]:
            raise IndexError(f"Slice index {slice_index} out of range for file {tiff_path}.")
        if direction == 'horizontal' and (coord_index < 0 or coord_index >= tiff_stack.shape[1]):
            raise IndexError(f"Y coordinate {coord_index} out of range for file {tiff_path}.")
        if direction == 'vertical' and (coord_index < 0 or coord_index >= tiff_stack.shape[2]):
            raise IndexError(f"X coordinate {coord_index} out of range for file {tiff_path}.")

        # Get the specified slice and the intensity values along the specified line
        slice_data = tiff_stack[slice_index]
        if direction == 'horizontal':
            line_data = slice_data[coord_index, :]
        elif direction == 'vertical':
            line_data = slice_data[:, coord_index]
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'.")

        # Plot the intensity distribution for this file
        plt.plot(range(len(line_data)), line_data, label=label)

    direction_label = 'Horizontal' if direction == 'horizontal' else 'Vertical'
    plt.title(f'{direction_label} Intensity Distribution at {"y" if direction == "horizontal" else "x"}={coord_index} on Slice {slice_index} for Multiple TIFF Files')
    plt.xlabel('Position along the line')
    plt.ylabel('Pixel Intensity')
    plt.legend(fontsize=8)
    plt.grid(True)

    # Save the plot in the specified folder with a custom name
    if save_plot:
        if save_folder is None:
            save_folder = os.path.dirname(file_paths[0])
        os.makedirs(save_folder, exist_ok=True)
        plot_path = os.path.join(save_folder, f'intensity_distribution_slice_{slice_index}_{direction}_{coord_index}_multiple_{custom_name}.png')
        plt.savefig(plot_path, format='png', dpi=300)
        plt.close()
        print(f"Plot saved at {plot_path}")
    else:
        plt.show()

# Example usage
file_paths = [
    r"C:\Users\rausc\Documents\EMBL\data\test_1\nema_avg_40.TIFF",
    r"C:\Users\rausc\Documents\EMBL\data\test_1\Nematostella_B_V0.TIFF",
    r"C:\Users\rausc\Documents\EMBL\data\test_1\Nematostella_B_V0_filtered.TIFF",
    r"C:\Users\rausc\Documents\EMBL\data\test_1\output_stack-Nema_B-test_3-Nematostella_B-epoch501.TIFF"
]

labels = [
    "GT",
    "noisy",
    "filtered",
    "denoised"
]

slice_index = 80  # Example slice index
coord_index = 134  # Example y or x coordinate
direction = 'vertical'  # 'horizontal' or 'vertical'
save_folder = r"C:\Users\rausc\Documents\EMBL\data\test_1"
custom_name = "nema-all-2"
plot_intensity_distribution_multiple_tiffs(file_paths, labels, slice_index, coord_index, direction, save_plot=True, save_folder=save_folder, custom_name=custom_name)

