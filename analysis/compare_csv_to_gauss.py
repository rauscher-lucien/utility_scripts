import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def read_noise_csv(filepath):
    """Read noise data from a CSV file."""
    noise_df = pd.read_csv(filepath)
    return noise_df['noise'].values

def generate_multiplicative_gaussian_data(mean, std_dev, size):
    """Generate Gaussian distribution data in a multiplicative way."""
    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, std_dev, size)
    
    # Create uniform data spanning a wide range
    base = np.random.uniform(1, 10, size)  # Random base between 1 and 10
    exponent = np.random.uniform(-50, 50, size)  # Random exponent between -50 and 50
    uniform_data = base ** exponent  # Generate the uniform data
    
    # Apply multiplicative noise
    noisy_data = uniform_data * gaussian_noise
    
    return noisy_data

def plot_noise_with_gaussian(log_noise_data, mean, std_dev, bins=1000):
    """Plot noise distribution and overlay log-transformed Gaussian distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot log-scaled noise histogram
    ax.hist(log_noise_data, bins=bins, density=True, alpha=0.6, color='b', label='Log-Scaled Noise Distribution')

    # Generate Gaussian distribution data
    noisy_data = generate_multiplicative_gaussian_data(mean, std_dev, 100000)

    # Log-transform the Gaussian data
    log_gaussian_data = np.log(noisy_data)

    # Generate density for log-transformed Gaussian distribution
    log_gaussian_density, log_gaussian_bins = np.histogram(log_gaussian_data, bins=bins, density=True)
    log_gaussian_bin_centers = (log_gaussian_bins[:-1] + log_gaussian_bins[1:]) / 2

    # Plot log-transformed Gaussian distribution
    ax.plot(log_gaussian_bin_centers, log_gaussian_density, 'r-', lw=2, label=f'Log-Transformed Gaussian Distribution\n(mean={mean}, std_dev={std_dev})')
    ax.set_title('Log-Scaled Noise Distribution with Log-Transformed Gaussian Distribution Overlay')
    ax.set_xlabel('Log Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
csv_filepath = r"C:\Users\rausc\Documents\EMBL\data\nema-results\noise_data_nema-noise-using_GT.csv"
log_noise_data = read_noise_csv(csv_filepath)

# Hardcoded mean and standard deviation for the Gaussian distribution
mean = 10  # Mean for log space noise
std_dev = 10  # Standard deviation for log space noise

# Set number of bins for the histogram
bins = 1000

# Plot the noise data with log-transformed Gaussian distribution overlay
plot_noise_with_gaussian(log_noise_data, mean, std_dev, bins=bins)


