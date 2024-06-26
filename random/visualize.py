import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile

# Step 1: Load TIF Image Stack
# Define the path to your TIF image stack
path_to_tif = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'test_data_4', 'tiff_stack.TIFF')

# Load the TIFF stack
image_stack = tifffile.imread(path_to_tif)

# Step 2: Downsample the Image Stack (optional)
downsample_factor = 5  # Adjust this factor as needed
downsampled_stack = image_stack[::downsample_factor, ::downsample_factor, ::downsample_factor]

# Step 3: Rescale the Image Stack to 0-1 range
rescaled_stack = downsampled_stack / np.max(downsampled_stack)

# Step 4: Threshold the Rescaled Image Stack
threshold_value = 0.6  # Adjust this threshold value as needed
binary_stack = (rescaled_stack > threshold_value)

# Step 5: Visualize the 3D Segmented Volume
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Voxel grid
x, y, z = np.indices(binary_stack.shape)
ax.scatter(x[binary_stack], y[binary_stack], z[binary_stack], c='red', alpha=0.5)  # Plotting only where binary_stack is True

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
