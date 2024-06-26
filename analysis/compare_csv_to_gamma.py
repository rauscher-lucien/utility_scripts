import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def read_noise_csv(filepath):
    """Read noise data from a CSV file."""
    noise_df = pd.read_csv(filepath)
    return noise_df['noise'].values

def generate_gamma_data(gamma_shape, gamma_scale, size, range_min=0, range_max=1e30):
    """Generate Gamma distribution data within a specified range."""
    gamma_data = np.random.gamma(gamma_shape, gamma_scale, size)
    gamma_data = gamma_data[(gamma_data >= range_min) & (gamma_data <= range_max)]
    return gamma_data

def plot_noise_with_gamma(noise_data, gamma_shape, gamma_scale, shift=0, bins=100, log_scale=False):
    """Plot noise distribution and overlay log-transformed Gamma distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot log-scaled noise histogram
    ax.hist(noise_data, bins=bins, density=True, alpha=0.6, color='b', label='Log-Scaled Noise Distribution')

    # Generate Gamma distribution data within the specified range
    gamma_data = generate_gamma_data(gamma_shape, gamma_scale, 100000) + shift
    log_gamma_data = np.log(gamma_data)

    # Generate density for log-transformed Gamma distribution
    log_gamma_density, log_gamma_bins = np.histogram(log_gamma_data, bins=bins, density=True)
    log_gamma_bin_centers = (log_gamma_bins[:-1] + log_gamma_bins[1:]) / 2

    # Plot log-transformed Gamma distribution
    ax.plot(log_gamma_bin_centers, log_gamma_density, 'r-', lw=2, label=f'Log-Transformed Gamma Distribution\n(shape={gamma_shape}, scale={gamma_scale}, shift={shift})')
    ax.set_title('Log-Scaled Noise Distribution with Log-Transformed Gamma Distribution Overlay')
    ax.set_xlabel('Log Value')
    ax.set_ylabel('Density')
    ax.set_xlim(-50, 50)
    ax.legend()
    ax.grid(True)

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

# Example usage
csv_filepath = r"C:\Users\rausc\Documents\EMBL\data\nema-results\noise_data_nema-noise-using_denoised.csv"
noise_data = read_noise_csv(csv_filepath)

gamma_shape = 0.055  # Adjust as needed
gamma_scale = 0.001  # Adjust as needed
shift = 0.0  # Adjust as needed
bins = 1000

plot_noise_with_gamma(noise_data, gamma_shape, gamma_scale, shift=shift, bins=bins, log_scale=False)





