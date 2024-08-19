import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def normalize_slice(slice_data):
    slice_data = slice_data.astype(float)
    slice_data -= slice_data.min()  # Subtract the minimum value
    slice_data /= slice_data.max()  # Divide by the maximum value to scale to 0-1
    return slice_data

def crop_to_multiple_of_16(slice_data):
    h, w = slice_data.shape
    new_h = h - (h % 16)
    new_w = w - (w % 16)
    cropped_slice = slice_data[:new_h, :new_w]
    return cropped_slice

def plot_slice(slice_data, show_axis=True, show_scale_bar=True, show_color_bar=True,
               scale_bar_length=50, scale_bar_label='50 um', scale_bar_color='white',
               cbar_position=[0.85, 0.75, 0.02, 0.2], cbar_text_color='white'):
    fig, ax = plt.subplots(figsize=(6, 6))  # Use subplots for better control
    im = ax.imshow(slice_data, cmap='gray', interpolation='nearest')
    
    if show_axis:
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
    else:
        ax.axis('off')

    # Add scale bar if enabled
    if show_scale_bar:
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

    # Create a colorbar inside the image if enabled
    if show_color_bar:
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')

        # Adjust the position and size of the colorbar
        cbar.ax.set_position(cbar_position)  # Position of the color bar

        # Set colorbar tick labels and label color
        cbar.ax.yaxis.set_tick_params(color=cbar_text_color)
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color=cbar_text_color)

    return fig

def save_plot(fig, output_path):
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Plot saved to {output_path}")

def plot_and_save_slice_from_tiff(stack_path, slice_index, show_axis=True, show_scale_bar=True, show_color_bar=True,
                                  scale_bar_length=50, scale_bar_label='50 um', cbar_position=[0.85, 0.75, 0.02, 0.2],
                                  cbar_text_color='white'):
    stack = read_tiff_stack(stack_path)
    
    if slice_index < 0 or slice_index >= stack.shape[0]:
        print(f"Slice index {slice_index} is out of range for the given stack. Max index: {stack.shape[0]-1}")
        return

    stack_dir = os.path.dirname(stack_path)
    stack_base = os.path.basename(stack_path)
    stack_name, _ = os.path.splitext(stack_base)
    
    output_path = os.path.join(stack_dir, f"{stack_name}_slice_{slice_index}.png")

    slice_data = normalize_slice(stack[slice_index])
    
    # Crop the slice to make the dimensions divisible by 16
    cropped_slice_data = crop_to_multiple_of_16(slice_data.squeeze())
    
    fig = plot_slice(cropped_slice_data, show_axis, show_scale_bar, show_color_bar,
                     scale_bar_length, scale_bar_label, cbar_position=cbar_position, cbar_text_color=cbar_text_color)
    save_plot(fig, output_path)

if __name__ == "__main__":
    stack_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1_filtered_bm3d_sigma_0.11.TIFF"
    slice_index = 100  # Specify the desired slice index here
    show_axis = False  # Set to False to hide the axis
    show_scale_bar = True  # Set to True to display the scale bar
    show_color_bar = False  # Set to True to display the color bar
    scale_bar_length = 100  # Specify the desired scale bar length here
    scale_bar_label = '100 um'  # Specify the desired scale bar label here
    cbar_position = [0.76, 0.6, 0.05, 0.1]  # Specify the desired color bar position here; left, bottom, width, and height
    cbar_text_color = 'white'  # Set the color of the colorbar text to white
    plot_and_save_slice_from_tiff(stack_path, slice_index, show_axis, show_scale_bar, show_color_bar,
                                  scale_bar_length, scale_bar_label, cbar_position, cbar_text_color)


