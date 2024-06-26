import numpy as np
import matplotlib.pyplot as plt

# Function to generate and log scale a gamma distribution
def generate_and_plot_gamma_distribution(shape, scale, size):
    # Generate a gamma distribution
    gamma_data = np.random.gamma(shape, scale, size)

    # Log scale the data
    log_gamma_data = np.log1p(gamma_data)

    # Plot original gamma distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(gamma_data, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.title(f'Gamma Distribution\n(shape={shape}, scale={scale}, size={size})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Plot log-scaled gamma distribution
    plt.subplot(1, 2, 2)
    plt.hist(log_gamma_data, bins=100, color='green', alpha=0.7, edgecolor='black')
    plt.title('Log-Scaled Gamma Distribution')
    plt.xlabel('Log Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
shape = 1.0
scale = 100.0
size = 100000
generate_and_plot_gamma_distribution(shape, scale, size)
