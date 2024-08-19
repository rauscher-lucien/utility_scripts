import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def normalize_slice(slice_data):
    slice_data = slice_data.astype(float)
    slice_data -= slice_data.min()  # Subtract the minimum value
    slice_data /= slice_data.max()  # Divide by the maximum value to scale to 0-1
    return slice_data

def plot_slice(slice_data, show_axis=True, scale_bar_length=50, scale_bar_label='50 um', scale_bar_color='white'):
    fig, ax = plt.subplots(figsize=(6, 6))  # Use subplots for better control
    im = ax.imshow(slice_data, cmap='gray', interpolation='nearest')
    
    if show_axis:
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
    else:
        ax.axis('off')

    # Add scale bar
    fontprops = fm.FontProperties(size=12)
    scalebar = AnchoredSizeBar(ax.transData,
                               scale_bar_length, 
                               scale_bar_label,
                               'lower right', 
                               pad=0.5,
                               color=scale_bar_color,
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)

    # Use make_axes_locatable to create an axis for the colorbar that matches the height of the image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust 'size' to control the width of the colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    return fig

def save_plot(fig, output_path):
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Plot saved to {output_path}")

def plot_and_save_slice_from_tiff(stack_path, slice_index, show_axis=True, scale_bar_length=50, scale_bar_label='50 um'):
    stack = read_tiff_stack(stack_path)
    
    if slice_index < 0 or slice_index >= stack.shape[0]:
        print(f"Slice index {slice_index} is out of range for the given stack. Max index: {stack.shape[0]-1}")
        return

    stack_dir = os.path.dirname(stack_path)
    stack_base = os.path.basename(stack_path)
    stack_name, _ = os.path.splitext(stack_base)
    
    output_path = os.path.join(stack_dir, f"{stack_name}_slice_{slice_index}.png")

    slice_data = normalize_slice(stack[slice_index])
    fig = plot_slice(slice_data, show_axis, scale_bar_length, scale_bar_label)
    save_plot(fig, output_path)

if __name__ == "__main__":
    stack_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0_filtered_bm3d_sigma_0.09.TIFF"
    slice_index = 60  # Specify the desired slice index here
    show_axis = False  # Set to False to hide the axis
    scale_bar_length = 100  # Specify the desired scale bar length here
    scale_bar_label = '100 um'  # Specify the desired scale bar label here
    plot_and_save_slice_from_tiff(stack_path, slice_index, show_axis, scale_bar_length, scale_bar_label)

