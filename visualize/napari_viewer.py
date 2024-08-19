import napari
import tifffile
import numpy as np

# Load the TIFF stack
file_path = r"C:\Users\rausc\Documents\EMBL\data\Plankton20230712LogScaleDiatom_Ciliate_A\Diatom_Ciliate_A_T0.TIFF"
image_stack = tifffile.imread(file_path).squeeze()

# Create a Napari viewer
viewer = napari.Viewer()

# Add the image stack to the viewer
viewer.add_image(image_stack, name='3D Image Stack', colormap='gray', rendering='mip')

# Start the Napari event loop
napari.run()

