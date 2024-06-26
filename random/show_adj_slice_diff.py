import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity as ssim

def load_image_as_grayscale(image_path):
    """
    Load an image and convert it to grayscale.
    Return the image as a NumPy array.
    """
    image = io.imread(image_path, as_gray=True)
    return image

def compute_and_plot_ssim_map(image1_path, image2_path, output_path=None):
    """
    Compute the SSIM map between two images and plot the result.
    Optionally save the result to a file.
    """
    # Load the two images
    image1 = load_image_as_grayscale(image1_path)
    image2 = load_image_as_grayscale(image2_path)

    # Ensure the images have the same shape
    assert image1.shape == image2.shape, "Images must have the same shape."

    # Compute the SSIM map
    ssim_value, ssim_map = ssim(image1, image2, full=True)

    # Plot the SSIM map
    plt.figure(figsize=(8, 8))
    plt.imshow(ssim_map, cmap='gray')
    plt.colorbar(label='SSIM Similarity')
    plt.title(f'SSIM Map (Overall SSIM: {ssim_value:.4f})')
    plt.axis('off')

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"SSIM map saved to {output_path}")
    else:
        plt.show()

# Example usage
image1_path = r"C:\Users\rausc\Documents\EMBL\data\Nema_B-single_slices\image_60.png"
image2_path = r"C:\Users\rausc\Documents\EMBL\data\Nema_B-single_slices\image_61.png"
output_image_path = r"C:\Users\rausc\Documents\EMBL\data\Nema_B-diff\ssim_map.png"
compute_and_plot_ssim_map(image1_path, image2_path, output_path=output_image_path)


