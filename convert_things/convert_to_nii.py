import os
import tifffile
import nibabel as nib
import numpy as np

def tiff_to_nifti(tiff_path, output_dir):
    """
    Convert a TIFF stack to a NIFTI file, rearranging dimensions to match the NIFTI format.

    Parameters:
    - tiff_path: Path to the input TIFF file.
    - output_dir: Directory where the output NIFTI file will be saved.
    """
    # Load the TIFF file
    image_data = tifffile.imread(tiff_path)

    # Rearrange dimensions from (Z, Y, X) to (X, Y, Z)
    image_data = np.transpose(image_data, (2, 1, 0))

    # Create a NIFTI image using an identity affine matrix
    affine = np.eye(4)  # This sets a default affine matrix with no scaling

    # Create a NIFTI image
    nifti_img = nib.Nifti1Image(image_data, affine)

    # Define the output file path, ensuring it ends with a proper extension
    base_name = os.path.basename(tiff_path)
    output_file = os.path.splitext(base_name)[0] + '.nii.gz'
    output_file_path = os.path.join(output_dir, output_file)

    # Save the NIFTI image
    nib.save(nifti_img, output_file_path)
    print(f"NIFTI file saved to {output_file_path}")


# Example usage
tiff_path = r"C:\Users\rausc\Documents\EMBL\data\big_data_small\OCT-data-1\Good_Sample_03_t_1.TIFF"
output_dir = r"C:\Users\rausc\Documents\EMBL\data\big_data_small\OCT-data-1"
tiff_to_nifti(tiff_path, output_dir)
