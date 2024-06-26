import numpy as np
import matplotlib.pyplot as plt

def scale_img(img):
    img = img.astype(np.float64)
    img = img / 65536
    img = img * 100 - 50
    img = np.power(10, img)
    return img

# Step 1: Generate a 16-bit array with a single mid-value
def generate_mid_value_image(shape, value=32768):
    return np.full(shape, value, dtype=np.int32)  # Changed to int32 to accommodate negative values

# Step 2: Add Gaussian noise to the array
def add_gaussian_noise(image, mean=0, std=1e20):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return noisy_image  # Return as int32 to include negative values

# Step 3: Plot the original and noisy images
def plot_images(original_image, noisy_image):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=65535)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(noisy_image, cmap='gray', vmin=np.min(noisy_image), vmax=np.max(noisy_image))
    axes[1].set_title('Noisy Image')
    axes[1].axis('off')

    plt.show()

# Step 4: Plot the noise distribution
def plot_noise_distribution(original_image, noisy_image):
    noise = noisy_image - original_image
    plt.figure(figsize=(8, 6))
    plt.hist(noise.flatten(), bins=100, density=True, alpha=0.7, color='b')
    plt.title('Noise Distribution')
    plt.xlabel('Noise Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

# Main function to run the steps
def main():
    image_shape = (512, 512)  # Define the shape of the image

    original_image = generate_mid_value_image(image_shape)

    scaled_original_image = scale_img(original_image)
    print(scaled_original_image.min())
    print(scaled_original_image.max())

    scaled_noisy_image = add_gaussian_noise(scaled_original_image)
    print(scaled_noisy_image.min())
    print(scaled_noisy_image.max())

    clean_image = np.log(scaled_original_image)
    noisy_image = np.log(scaled_noisy_image)


    #plot_images(clean_image, noisy_image)
    plot_noise_distribution(clean_image, noisy_image)

if __name__ == "__main__":
    main()
