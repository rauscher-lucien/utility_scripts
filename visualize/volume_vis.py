import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import tifffile

# Load the TIFF stack
file_path = r"C:\Users\rausc\Documents\EMBL\data\test_2\output_stack-big_data_small-no_nema-no_droso-test_1-droso_good-epoch547.TIFF"
image_stack = tifffile.imread(file_path).squeeze()

# Create a figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Use marching cubes to obtain a surface mesh
verts, faces, _, _ = measure.marching_cubes(image_stack, level=np.mean(image_stack))

# Plot the surface
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                cmap='Spectral', lw=1, alpha=0.5)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the aspect ratio
ax.set_box_aspect([np.ptp(image_stack.shape[0]),
                   np.ptp(image_stack.shape[1]),
                   np.ptp(image_stack.shape[2])])

# Show the plot
plt.show()
