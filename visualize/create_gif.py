import tifffile
import imageio
import numpy as np
import os

def create_gif_from_tiff(tiff_path, output_folder, duration_per_frame=0.1):
    try:
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

        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Generate the GIF filename
        base_name = os.path.basename(tiff_path)
        gif_name = os.path.splitext(base_name)[0] + ".gif"
        output_gif_path = os.path.join(output_folder, gif_name)

        # Save the images as a GIF, making it loop forever
        imageio.mimsave(output_gif_path, images, duration=duration_per_frame, loop=0)

        print(f"GIF created and saved at {output_gif_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
tiff_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF"
output_folder = r"C:\Users\rausc\Documents\EMBL\data\gifs"
create_gif_from_tiff(tiff_path, output_folder, duration_per_frame=0.1)



