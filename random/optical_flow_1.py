import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import numpy as np

# Load two input images
image_1_path = r"C:\Users\rausc\Documents\EMBL\data\test_data_x\image_0.png"
image_2_path = r"C:\Users\rausc\Documents\EMBL\data\test_data_x\image_1.png"
image1 = cv2.imread(image_1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image_2_path, cv2.IMREAD_GRAYSCALE)

# Convert images to PyTorch tensors
image1_tensor = TF.to_tensor(image1).unsqueeze(0).float()
image2_tensor = TF.to_tensor(image2).unsqueeze(0).float()

# Compute optical flow between the two images
optical_flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Generate grid from optical flow
grid = np.zeros_like(optical_flow)
grid_height, grid_width = optical_flow.shape[:2]
grid_y, grid_x = np.indices((grid_height, grid_width))

grid[:, :, 0] = grid_x
grid[:, :, 1] = grid_y
grid += optical_flow


# Normalize grid to [-1, 1]
grid_norm = np.zeros_like(grid)
grid_norm[:, :, 0] = 2 * (grid[:, :, 0] / (image1.shape[1] - 1)) - 1
grid_norm[:, :, 1] = 2 * (grid[:, :, 1] / (image1.shape[0] - 1)) - 1

# Convert to PyTorch tensor
grid_tensor = torch.tensor(grid_norm).unsqueeze(0).float()

# Perform image warping
warped_image = F.grid_sample(image1_tensor, grid_tensor, align_corners=True)

# Convert back to numpy array
warped_image_np = warped_image.squeeze().numpy()

# Convert back to uint8
warped_image_np_uint8 = (warped_image_np * 255).astype(np.uint8)

# Display warped image
cv2.imshow('Original Image 1', image1)
cv2.imshow('Original Image 2', image2)
cv2.imshow('Warped Image', warped_image_np_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()
