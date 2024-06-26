import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to plot histogram of an RGB image
def plot_rgb_histogram(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert the image from BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split the image into its R, G, B channels
    r, g, b = cv2.split(image)

    # Plot the histograms for each channel
    plt.figure(figsize=(10, 7))
    plt.hist(r.ravel(), bins=256, range=(0, 256), color='red', alpha=0.5, label='Red channel')
    plt.hist(g.ravel(), bins=256, range=(0, 256), color='green', alpha=0.5, label='Green channel')
    plt.hist(b.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.5, label='Blue channel')
    
    # Add titles and labels
    plt.title('Histogram of RGB Channels')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Show the plot
    plt.show()

# Example usage
image_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\12003.jpg"  # Replace with your image path
plot_rgb_histogram(image_path)
