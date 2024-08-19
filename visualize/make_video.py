import os
import cv2
import numpy as np
import tifffile
import glob
import matplotlib.pyplot as plt

def create_video_from_slices(folder_path, slice_index, output_dir, fps=5):
    # Get all TIFF stack filenames
    filenames = glob.glob(os.path.join(folder_path, "*.tif")) + glob.glob(os.path.join(folder_path, "*.tiff"))
    filenames.sort()  # Sort to ensure order

    if not filenames:
        print("No TIFF files found in the specified directory.")
        return

    # Read the first image to get the dimensions
    first_stack = tifffile.imread(filenames[0])
    if slice_index >= first_stack.shape[0]:
        print(f"Slice index {slice_index} is out of bounds for the given TIFF stacks.")
        return

    # Create output video file name based on the folder name
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_video_path = os.path.join(output_dir, f"{folder_name}_slice{slice_index}.avi")

    height, width, _ = first_stack[slice_index].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    for filename in filenames:
        stack = tifffile.imread(filename).squeeze()
        if slice_index < stack.shape[0]:
            frame = stack[slice_index]

            # Normalize 16-bit frame to 8-bit
            frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame_8bit = frame_normalized.astype(np.uint8)

            # Plot the frame for debugging
            # plt.imshow(frame_8bit, cmap='gray')
            # plt.title(f'Frame from {os.path.basename(filename)}')
            # plt.show()

            video_writer.write(frame_8bit)

    video_writer.release()
    print(f"Video saved as {output_video_path}")

def main():
    # Hardcoded parameters
    folder_path = r"\\tier2.embl.de\prevedel\members\Rauscher\data\denoised_videos\Drosophila20220310LogScaleDrosophila_twin_A"
    slice_index = 95
    output_dir = r"\\tier2.embl.de\prevedel\members\Rauscher\data\denoised_videos\Drosophila20220310LogScaleDrosophila_twin_A"

    create_video_from_slices(folder_path, slice_index, output_dir, fps=5)

if __name__ == '__main__':
    main()



