import os
from PIL import Image

def convert_to_8bit_grayscale(image_path):
    if os.path.exists(image_path) and image_path.lower().endswith(".jpg"):  # Check if the file exists and is a JPEG
        with Image.open(image_path) as img:
            gray_img = img.convert("L")  # Convert the image to 8-bit grayscale
            gray_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_8bit_gray.jpg"
            gray_path = os.path.join(os.path.dirname(image_path), gray_filename)
            gray_img.save(gray_path)  # Save the 8-bit grayscale image
            print(f"Converted {os.path.basename(image_path)} to 8-bit grayscale and saved as {gray_filename}")
    else:
        print("The specified file does not exist or is not a JPEG image.")

# Specify the path to the image you want to convert
image_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\12003.jpg"
convert_to_8bit_grayscale(image_path)
