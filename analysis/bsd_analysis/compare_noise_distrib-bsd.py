import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

def load_noise_values(file_paths):
    noise_values = []
    labels = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            noise_values.append(data['Noise'].values)
            labels.append(os.path.basename(file_path))
        else:
            print(f"File {file_path} does not exist.")
    return noise_values, labels

def plot_multiple_noise_distributions(file_paths, save_plot=True, output_dir=None):
    # Load noise values from the given file paths
    noise_values, labels = load_noise_values(file_paths)

    # Plot the noise distributions
    fig, ax = plt.subplots(figsize=(10, 8))
    for noise, label in zip(noise_values, labels):
        # Fit a Gaussian to the data
        mu, std = norm.fit(noise)
        
        # Plot the histogram
        ax.hist(noise, bins=200, density=True, alpha=0.6, label=f'{label}\n$\mu={mu:.2f}$, $\sigma={std:.2f}$')
        
        # Plot the fitted Gaussian
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
    
    ax.set_title('Noise Distributions')
    ax.set_xlabel('Noise Value')
    ax.set_ylabel('Density')
    ax.grid(True)
    ax.legend(loc='upper right')

    # Save the figure as one plot
    if save_plot:
        if output_dir is None:
            output_dir = os.path.dirname(file_paths[0])
        plot_path = os.path.join(output_dir, 'multiple_noise_distributions2.png')
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Noise distributions plot saved at {plot_path}")
    else:
        plt.show()

# Example usage
file_paths = [
    r"Z:\members\Rauscher\projects\N2N-BSD300\BSD-test_1\results\BSD2\noise_values_BSD-denoised.csv",
    r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\noise_values_BSD-clean.csv",
    r"Z:\members\Rauscher\projects\N2N-BSD300\BSD-test_2\results\BSD1\noise_values_BSD-denoised.csv"
]

plot_multiple_noise_distributions(file_paths)
