import os
from PIL import Image
import numpy as np

def convert_to_16bit_grayscale(image_path):
    if os.path.exists(image_path) and image_path.lower().endswith(".jpg"):  # Check if the file exists and is a JPEG
        with Image.open(image_path) as img:
            gray_img = img.convert("L")  # Convert the image to 8-bit grayscale
            gray_np = np.array(gray_img, dtype=np.uint8)  # Convert to numpy array with 8-bit values

            gray_np_16bit = (gray_np.astype(np.uint16) << 8)  # Scale to 16-bit by shifting bits

            gray_16bit_img = Image.fromarray(gray_np_16bit, mode="I;16")  # Convert back to PIL Image
            gray_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_16bit_gray.tif"
            gray_path = os.path.join(os.path.dirname(image_path), gray_filename)
            gray_16bit_img.save(gray_path)  # Save the 16-bit grayscale image
            print(f"Converted {os.path.basename(image_path)} to 16-bit grayscale and saved as {gray_filename}")
    else:
        print("The specified file does not exist or is not a JPEG image.")

# Specify the path to the image you want to convert
image_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\12003.jpg"
convert_to_16bit_grayscale(image_path)




