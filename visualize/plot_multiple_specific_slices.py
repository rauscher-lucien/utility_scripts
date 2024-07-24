import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

def plot_specific_slice_from_all_tiff_stacks(folder_path, slice_index, title_fontsize=10):
    # Prepare to collect valid slices for plotting
    slices = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.tif', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            stack = tifffile.imread(image_path)
            
            if stack.ndim >= 3 and slice_index < stack.shape[0]:  # Ensure it's a stack and slice index is valid
                slices.append(stack[slice_index])
                filenames.append(filename)
            elif stack.ndim < 3:
                print(f'{filename} is not a stack or only contains a single image.')
            else:
                print(f'Slice index {slice_index} out of range for {filename}. Max index: {stack.shape[0]-1}')
    
    if not slices:
        print("No valid slices found to plot.")
        return

    # Determine the number of rows and columns for the subplot grid
    num_slices = len(slices)
    cols = int(np.ceil(np.sqrt(num_slices)))  # Aim for a roughly square grid
    rows = int(np.ceil(num_slices / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.suptitle(f'Slice {slice_index} Across Multiple Stacks', fontsize=title_fontsize+2)  # slightly larger title for the figure
    
    for i, (slice_img, fname) in enumerate(zip(slices, filenames)):
        ax = axs.flatten()[i]
        ax.imshow(slice_img, cmap='gray')
        ax.set_title(fname, fontsize=title_fontsize)  # set the font size here
        ax.axis('off')  # Turn off axis for a cleaner look

    # Turn off any unused subplots
    for j in range(i + 1, rows * cols):
        axs.flatten()[j].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the super title
    plt.show()

# Example usage:
folder_path = r"C:\Users\rausc\Documents\EMBL\data\test_2"
specific_slice_index = 60  # Specify the desired slice index here
plot_specific_slice_from_all_tiff_stacks(folder_path, specific_slice_index, title_fontsize=8)  # Smaller font size for subplot titles


