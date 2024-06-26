import os
import tifffile as tf

def crop_stack_to_match(base_file_path, input_file_path, output_file_path):
    # Load the base and input stacks
    with tf.TiffFile(base_file_path) as tif:
        base_stack = tif.asarray()
    
    with tf.TiffFile(input_file_path) as tif:
        input_stack = tif.asarray()
    
    # Ensure the input stack matches the base stack's dimensions
    if base_stack.shape != input_stack.shape:
        # Calculate new dimensions, ensuring we don't exceed the input stack's size
        new_depth = min(base_stack.shape[0], input_stack.shape[0])
        new_height = min(base_stack.shape[1], input_stack.shape[1])
        new_width = min(base_stack.shape[2], input_stack.shape[2])
        
        # Crop the input stack
        cropped_stack = input_stack[:new_depth, :new_height, :new_width]
        
        # Save the cropped stack
        tf.imwrite(output_file_path, cropped_stack)
        print(f"Input stack was cropped and saved to {output_file_path}.")
    else:
        print("Base and input stacks have the same dimensions. No action taken.")

# Example usage
path=os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'noise_model_calibration', 'droso_embryo_oil')
base_file_path = os.path.join(path, 'clean', 'output_stack-droso_embryo_oil-test_1-inference_80-droso_embryo_oil-2.tiff')
input_file_path = os.path.join(path, 'noisy', 'rescaled-droso_embryo_oil.tiff')
output_file_path = os.path.join(path, 'noisy', 'reshaped-rescaled-droso_embryo_oil.tiff')

crop_stack_to_match(base_file_path, input_file_path, output_file_path)
