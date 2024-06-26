import numpy as np
import matplotlib.pyplot as plt

# Function to generate and log scale a continuous Poisson-like distribution
def generate_and_plot_continuous_poisson_approximation(lambda_value, size):
    # Generate a continuous approximation of the Poisson distribution using a normal distribution
    mean = lambda_value
    std_dev = np.sqrt(lambda_value)
    continuous_poisson_data = np.random.normal(loc=mean, scale=std_dev, size=size)

    # Clip the values to ensure no negative values (since Poisson distribution is non-negative)
    continuous_poisson_data = np.clip(continuous_poisson_data, a_min=0, a_max=None)

    # Log scale the data
    log_continuous_poisson_data = np.log1p(continuous_poisson_data)

    # Plot original continuous Poisson-like distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(continuous_poisson_data, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.title(f'Continuous Poisson-like Distribution\n(lambda={lambda_value}, size={size})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Plot log-scaled continuous Poisson-like distribution
    plt.subplot(1, 2, 2)
    plt.hist(log_continuous_poisson_data, bins=100, color='green', alpha=0.7, edgecolor='black')
    plt.title('Log-Scaled Continuous Poisson-like Distribution')
    plt.xlabel('Log Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
lambda_value = 5
size = 100000
generate_and_plot_continuous_poisson_approximation(lambda_value, size)

