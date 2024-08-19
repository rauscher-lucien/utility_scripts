import matplotlib.pyplot as plt
import numpy as np
import os

# Read and parse the text file
file_path = r"C:\Users\rausc\Documents\EMBL\data\general-results\inference_time.txt"
times = []
labels = []

with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if ':' in line:
            label, time = line.split(':')
            labels.append(label.strip())
            times.append(float(time.strip()))

# Convert times to numpy array for easier manipulation
times = np.array(times)

# Calculate the mean of the times
mean_time = np.mean(times)

# Normalize the times to the mean
normalized_times = times / mean_time

# Plotting the normalized times
plt.figure(figsize=(10, 6))
plt.bar(labels, normalized_times, color='skyblue')
plt.axhline(y=1, color='r', linestyle='--')  # Line at mean=1
plt.xlabel('Architecture (Layer and Base)')
plt.ylabel('Normalized Inference Time')
plt.title('Normalized Inference Times for Different Network Architectures')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot in the same directory as the text file
output_dir = os.path.dirname(file_path)
output_path = os.path.join(output_dir, 'normalized_inference_times.png')
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

