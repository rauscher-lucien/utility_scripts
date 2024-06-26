import numpy as np
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt

def plot_as_image(data, title='Image Display', cmap='gray', colorbar=True):
    plt.figure(figsize=(6, 6))

    if isinstance(data, torch.Tensor):
        # Ensure it's on the CPU and convert to NumPy
        data = data.detach().cpu().numpy()

    # Check if the data has multiple channels and select the first one if so
    if data.ndim == 3 and (data.shape[0] == 3 or data.shape[0] == 1):
        data = data[0]  # Assume the first channel for visualization if it's a 3-channel image

    data = data.squeeze()

    img = plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.axis('off')  # Turn off axis numbers and ticks

    if colorbar:
        plt.colorbar(img)

    plt.show()

    print(data.min())
    print(data.max())

def extract_and_save_images(npz_filename, output_folder):
    data = np.load(npz_filename)
    
    for idx, (file_key, array) in enumerate(data.items()):
        # for i in range(0, array.shape[0]):
        #     print(idx)
        #     print(file_key)
        #     plot_as_image(array[i, :, :])
        print(file_key)
        

# Example usage
npz_filename = r"C:\Users\rausc\Documents\EMBL\data\OCTMNIST\octmnist.npz"
output_folder = r"C:\Users\rausc\Documents\EMBL\data\OCTMNIST"  # Change to your desired output directory

extract_and_save_images(npz_filename, output_folder)
