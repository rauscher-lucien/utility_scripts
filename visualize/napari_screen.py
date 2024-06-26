import napari
import tifffile
import numpy as np
import imageio
import os

# Load the TIFF stack
file_path = r"C:\Users\rausc\Documents\EMBL\data\test\output_stack-Nema_B-test_3-Nematostella_B-epoch501.TIFF"
image_stack = tifffile.imread(file_path).squeeze()

# Get the directory and name of the input file
output_dir = os.path.dirname(file_path)
file_name = os.path.basename(file_path).replace('.TIFF', '')

# Create a napari viewer
viewer = napari.Viewer()
viewer.add_image(image_stack, name='3D Image Stack', rendering='mip')  # Use Maximum Intensity Projection (MIP) for full volume view

# Add axis lines
z, y, x = image_stack.shape
axis_length = max(z, y, x) * 0.1  # 10% of the maximum dimension

# X-axis (red)
x_axis = np.array([[0, 0, 0], [axis_length, 0, 0]])
viewer.add_shapes([x_axis], shape_type='line', edge_width=2, edge_color='red', name='X-axis')

# Y-axis (green)
y_axis = np.array([[0, 0, 0], [0, axis_length, 0]])
viewer.add_shapes([y_axis], shape_type='line', edge_width=2, edge_color='green', name='Y-axis')

# Z-axis (blue)
z_axis = np.array([[0, 0, 0], [0, 0, axis_length]])
viewer.add_shapes([z_axis], shape_type='line', edge_width=2, edge_color='blue', name='Z-axis')

# Ensure 3D display
viewer.dims.ndisplay = 3

# Define specific camera angles (azimuth, elevation, roll)
angles = [
    (0, 0, 90),  # front view
    (0, 45, 90)  # angled view
]

# Zoom out factor
zoom_out_factor = 0.5

# Take snapshots from different angles and save them
for idx, (azimuth, elevation, roll) in enumerate(angles):
    viewer.camera.angles = (azimuth, elevation, roll)
    viewer.camera.zoom *= zoom_out_factor  # Zoom out
    snapshot = viewer.screenshot(canvas_only=True)  # Take the snapshot
    output_path = os.path.join(output_dir, f'{file_name}_snapshot_{idx}.png')
    imageio.imwrite(output_path, snapshot)

# Close the viewer after capturing snapshots
viewer.close()
