import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def plot_with_insets(slice_data, regions, region_colors, show_axis=True, show_scale_bar=True, show_color_bar=True,
                     scale_bar_length=50, scale_bar_label='50 um', scale_bar_color='white',
                     scale_bar_position='lower right', scale_bar_font_size=12,
                     cbar_position=[0.85, 0.75, 0.02, 0.2], cbar_text_color='white'):
    fig, ax_main = plt.subplots(figsize=(8, 8))

    # Normalize the whole image slice
    normalized_slice = normalize_slice(slice_data)
    
    # Determine vmin and vmax for consistent scaling
    vmin, vmax = normalized_slice.min(), normalized_slice.max()
    
    im = ax_main.imshow(normalized_slice, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    
    if show_axis:
        ax_main.set_xlabel('X-axis')
        ax_main.set_ylabel('Y-axis')
    else:
        ax_main.axis('off')

    # Add scale bar if enabled
    if show_scale_bar:
        fontprops = fm.FontProperties(size=scale_bar_font_size)
        scalebar = AnchoredSizeBar(ax_main.transData,
                                   scale_bar_length, 
                                   scale_bar_label,
                                   scale_bar_position, 
                                   pad=0.5,
                                   color=scale_bar_color,
                                   frameon=False,
                                   size_vertical=1,
                                   fontproperties=fontprops)
        ax_main.add_artist(scalebar)

    # Add color bar if enabled
    if show_color_bar:
        cbar = fig.colorbar(im, ax=ax_main, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')

        # Adjust the position and size of the colorbar
        cbar.ax.set_position(cbar_position)  # Position of the color bar

        # Set colorbar tick labels and label color
        cbar.ax.yaxis.set_tick_params(color=cbar_text_color)
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color=cbar_text_color)

    # Inset configuration
    num_insets = len(regions)
    inset_width = 0.2  # Width of each inset (relative to the figure)
    inset_height = 0.2  # Height of each inset (relative to the figure)
    inset_spacing = 0.1  # Spacing between insets

    # Calculate the total width required by all insets and the spacing between them
    total_insets_width = num_insets * inset_width + (num_insets - 1) * inset_spacing

    # Adjust the main plot to allow space for the insets below and set it to the full width
    ax_main.set_position([0.1, 0.3, 0.8, 0.6])  # [left, bottom, width, height]

    # Calculate the starting x position for the insets so that they align with the main plot
    start_x = ax_main.get_position().x0  # Align with the left edge of the main plot

    # Position the insets below the main plot
    inset_y_pos = 0.14  # Y position for the insets

    # Plot squares on the main image for each region with different colors
    for idx, (region, color) in enumerate(zip(regions, region_colors)):
        rect = patches.Rectangle((region[1], region[0]), region[3], region[2], linewidth=2, edgecolor=color, facecolor='none')
        ax_main.add_patch(rect)

        # Extract the region for the inset subplot
        inset_region = normalized_slice[region[0]:region[0]+region[2], region[1]:region[1]+region[3]]

        # Calculate the x position for each inset
        inset_x_start = start_x + idx * (inset_width + inset_spacing)

        # Plot the inset
        inset_ax = fig.add_axes([inset_x_start, inset_y_pos, inset_width, inset_height], frameon=False)
        inset_ax.imshow(inset_region, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
        inset_ax.axis('off')

        # Draw a colored rectangle around the inset
        rect = patches.Rectangle((0, 0), 1, 1, transform=inset_ax.transAxes,
                                 linewidth=6, edgecolor=color, facecolor='none')
        inset_ax.add_patch(rect)

    return fig

def save_plot(fig, output_path):
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Plot saved to {output_path}")

def plot_and_save_slice_with_insets(stack_path, slice_index, regions, region_colors, show_axis=True, show_scale_bar=True, show_color_bar=True,
                                    scale_bar_length=50, scale_bar_label='50 um', scale_bar_color='white',
                                    scale_bar_position='lower right', scale_bar_font_size=12,
                                    cbar_position=[0.85, 0.75, 0.02, 0.2], cbar_text_color='white'):
    stack = read_tiff_stack(stack_path)
    
    if slice_index < 0 or slice_index >= stack.shape[0]:
        print(f"Slice index {slice_index} is out of range for the given stack. Max index: {stack.shape[0]-1}")
        return

    stack_dir = os.path.dirname(stack_path)
    stack_base = os.path.basename(stack_path)
    stack_name, _ = os.path.splitext(stack_base)
    
    output_path = os.path.join(stack_dir, f"{stack_name}_slice_{slice_index}_with_insets.png")

    slice_data = normalize_slice(stack[slice_index])
    fig = plot_with_insets(slice_data, regions, region_colors, show_axis, show_scale_bar, show_color_bar,
                           scale_bar_length, scale_bar_label, scale_bar_color, scale_bar_position, scale_bar_font_size,
                           cbar_position, cbar_text_color)
    save_plot(fig, output_path)

if __name__ == "__main__":
    stack_path = r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF"
    slice_index = 250  # Specify the desired slice index here

    # Define regions as [y, x, height, width] for insets
    regions = [
        [65, 125, 40, 40],  # Region 1
        [60, 210, 40, 40],  # Region 2
        [100, 250, 30, 30]   # Region 3
    ]
    
    # Define colors for each region
    region_colors = ['green', 'red', 'blue']  # Adjust colors as needed

    
    plot_and_save_slice_with_insets(stack_path, slice_index, regions, region_colors, show_axis=False, show_scale_bar=True, show_color_bar=True,
                                    scale_bar_length=100, scale_bar_label='100 um', scale_bar_color='white',
                                    scale_bar_position='lower right', scale_bar_font_size=12,
                                    cbar_position=[0.83, 0.6, 0.02, 0.12], cbar_text_color='white')
