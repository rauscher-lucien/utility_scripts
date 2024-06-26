import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

def load_noise_values(file_paths):
    noise_values = []
    labels = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            noise_values.append(data['noise'].values)
            labels.append(os.path.basename(file_path))
        else:
            print(f"File {file_path} does not exist.")
    return noise_values, labels

def plot_multiple_noise_distributions(file_paths, custom_labels, use_line_plot=False, save_plot=True, output_dir=None):
    # Load noise values from the given file paths
    noise_values, file_labels = load_noise_values(file_paths)

    if len(custom_labels) != len(file_paths):
        raise ValueError("The number of custom labels must match the number of file paths.")

    # Define colors for the plots
    colors = ['blue', 'orange', 'green']

    # Plot the noise distributions
    fig, ax = plt.subplots(figsize=(10, 8))
    for noise, label, color in zip(noise_values, custom_labels, colors):
        if use_line_plot:
            # Use kernel density estimation (KDE) for line plot
            kde = gaussian_kde(noise, bw_method='silverman')
            noise_range = np.linspace(min(noise), max(noise), 100)
            ax.plot(noise_range, kde(noise_range), label=label, color=color, linewidth=2)
        else:
            # Use histogram for plot
            ax.hist(noise, bins=500, density=True, alpha=0.6, label=label)

    ax.set_title('Noise Distributions')
    ax.set_xlabel('Noise Value')
    ax.set_ylabel('Density')
    ax.grid(True)
    ax.legend(loc='upper right')

    # Save the figure as one plot
    if save_plot:
        if output_dir is None:
            output_dir = os.path.dirname(file_paths[0])
        plot_path = os.path.join(output_dir, 'multiple_noise_distributions-mouse-inside.png')
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Noise distributions plot saved at {plot_path}")
    else:
        plt.show()

# Example usage
file_paths = [
    r"C:\Users\rausc\Documents\EMBL\data\mouse-results\noise_data_mouse-noise-GT-inside.csv",
    r"C:\Users\rausc\Documents\EMBL\data\mouse-results\noise_data_mouse-noise-filter-inside.csv",
    r"C:\Users\rausc\Documents\EMBL\data\mouse-results\noise_data_mouse-noise-network-inside.csv"
]

custom_labels = [
    "GT",
    "filter",
    "network"
]

plot_multiple_noise_distributions(file_paths, custom_labels, use_line_plot=False)
