import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def read_noise_csv(filepath):
    """Read noise data from a CSV file."""
    noise_df = pd.read_csv(filepath)
    return noise_df['noise'].values

def generate_continuous_poisson_data(lam, size, shift, range_min=1e-10):
    """Generate a continuous approximation of Poisson distribution data within a specified range."""
    gamma_shape = lam
    gamma_scale = 1.0
    poisson_approx_data = np.random.gamma(gamma_shape, gamma_scale, size) + shift
    poisson_approx_data = poisson_approx_data[poisson_approx_data >= range_min]
    return poisson_approx_data

def plot_noise_with_poisson(log_noise_data, lam, shift=0, bins=100):
    """Plot noise distribution and overlay log-transformed Poisson-like distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot log-scaled noise histogram
    ax.hist(log_noise_data, bins=bins, density=True, alpha=0.6, color='b', label='Log-Scaled Noise Distribution')

    # Generate continuous Poisson-like distribution data
    poisson_data = generate_continuous_poisson_data(lam, 100000, shift)

    # Log-transform the Poisson data
    log_poisson_data = np.log(poisson_data + 1e-10)

    # Generate density for log-transformed Poisson-like distribution
    log_poisson_density, log_poisson_bins = np.histogram(log_poisson_data, bins=bins, density=True)
    log_poisson_bin_centers = (log_poisson_bins[:-1] + log_poisson_bins[1:]) / 2

    # Plot log-transformed Poisson-like distribution
    ax.plot(log_poisson_bin_centers, log_poisson_density, 'r-', lw=2, label=f'Log-Transformed Poisson-like Distribution\n(lambda={lam}, shift={shift})')
    ax.set_title('Log-Scaled Noise Distribution with Log-Transformed Poisson-like Distribution Overlay')
    ax.set_xlabel('Log Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
csv_filepath = r"C:\Users\rausc\Documents\EMBL\data\nema-results\noise_data_nema-noise.csv"
log_noise_data = read_noise_csv(csv_filepath)

# Hardcoded lambda for the Poisson-like distribution
lam = 0.05  # Adjust as needed
shift = 1e-10  # Adjust as needed

# Set number of bins for the histogram
bins = 50

# Plot the noise data with log-transformed Poisson-like distribution overlay
plot_noise_with_poisson(log_noise_data, lam, shift=shift, bins=bins)

