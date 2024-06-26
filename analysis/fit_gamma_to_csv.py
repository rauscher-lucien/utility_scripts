import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def read_noise_csv(filepath):
    """Read noise data from a CSV file."""
    noise_df = pd.read_csv(filepath)
    return noise_df['noise'].values

def fit_gamma_distribution(data):
    """Fit a Gamma distribution to the data."""
    # Fit a Gamma distribution to the data
    gamma_params = stats.gamma.fit(data)
    return gamma_params

def plot_noise_with_fitted_gamma(noise_data, bins=100):
    """Plot noise distribution and overlay both log-transformed and original Gamma distributions."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot log-scaled noise histogram
    axs[0].hist(noise_data, bins=bins, density=True, alpha=0.6, color='b', label='Log-Scaled Noise Distribution')

    # Fit Gamma distribution to the original (exponentially scaled) data
    exp_noise_data = np.expm1(noise_data)  # Transform log-scaled data back to original scale
    gamma_params = fit_gamma_distribution(exp_noise_data)
    
    # Generate Gamma distribution data based on the fitted parameters
    gamma_shape, gamma_loc, gamma_scale = gamma_params
    gamma_data = np.random.gamma(gamma_shape, gamma_scale, 100000) + gamma_loc

    # Transform fitted Gamma distribution to log scale
    log_gamma_data = np.log1p(gamma_data)

    # Generate density for log-transformed Gamma distribution
    log_gamma_density, log_gamma_bins = np.histogram(log_gamma_data, bins=bins, density=True)
    log_gamma_bin_centers = (log_gamma_bins[:-1] + log_gamma_bins[1:]) / 2

    # Plot log-transformed Gamma distribution
    axs[0].plot(log_gamma_bin_centers, log_gamma_density, 'r-', lw=2, label=f'Log-Transformed Gamma Distribution\n(shape={gamma_shape:.2f}, scale={gamma_scale:.2f})')
    axs[0].set_title('Log-Scaled Noise Distribution with Log-Transformed Fitted Gamma Distribution Overlay')
    axs[0].set_xlabel('Log Value')
    axs[0].set_ylabel('Density')
    axs[0].legend()
    axs[0].grid(True)

    # Plot log-scaled noise histogram and original fitted Gamma distribution
    axs[1].hist(noise_data, bins=bins, density=True, alpha=0.6, color='b', label='Log-Scaled Noise Distribution')

    # Generate density for original fitted Gamma distribution
    gamma_density, gamma_bins = np.histogram(gamma_data, bins=bins, density=True)
    gamma_bin_centers = (gamma_bins[:-1] + gamma_bins[1:]) / 2

    # Plot original fitted Gamma distribution
    axs[1].plot(gamma_bin_centers, gamma_density, 'r-', lw=2, label=f'Original Fitted Gamma Distribution\n(shape={gamma_shape:.2f}, scale={gamma_scale:.2f})')
    axs[1].set_title('Log-Scaled Noise Distribution with Original Fitted Gamma Distribution Overlay')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Density')
    axs[1].set_xlim(0, 1)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
csv_filepath = r"C:\Users\rausc\Documents\EMBL\data\test_3\noise_data_ME-scale_back-rescaled-1.csv"
noise_data = read_noise_csv(csv_filepath)

bins = 1000

plot_noise_with_fitted_gamma(noise_data, bins=bins)
