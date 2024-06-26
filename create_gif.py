import tifffile
import imageio
import numpy as np

def create_gif_from_tiff(tiff_path, output_gif_path, duration_per_frame=0.1):
    # Load the TIFF file
    stack = tifffile.imread(tiff_path)

    # Normalize the image data to 0-255 scale for GIF creation
    stack = stack.astype(np.float32).squeeze()  # Convert to float to prevent clipping during scaling
    stack -= stack.min()  # Subtract the minimum value
    stack /= stack.max()  # Divide by the max value
    stack *= 255.0  # Scale to 0-255
    stack = stack.astype(np.uint8)  # Convert back to 8-bit integer

    # Create a list to hold the image data
    images = [stack[i] for i in range(stack.shape[0])]

    # Save the images as a GIF, making it loop forever
    imageio.mimsave(output_gif_path, images, duration=duration_per_frame, loop=0)

    print(f"GIF created and saved at {output_gif_path}")

# Example usage
tiff_path = r"C:\Users\rausc\Documents\EMBL\data\test_2\output_stack-big_data_small-no_nema-no_droso-test_1-droso_good-epoch547.TIFF"
output_gif_path = r"C:\Users\rausc\Documents\EMBL\data\Nema_B-analysis\Drosophila_twin_A_V0-gif.gif"
create_gif_from_tiff(tiff_path, output_gif_path, duration_per_frame=0.1)


