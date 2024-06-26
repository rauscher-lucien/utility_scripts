import matplotlib.pyplot as plt
import os

# Hardcoded heights of the bars
heights = [1221, 5527, 22661, 91085, 364616, 1458473]

# Corresponding x-axis values and labels
x = range(len(heights))
labels = ['1', '2', '3', '4', '5', '6']

# Create a bar plot
plt.bar(x, heights)

# Add title and labels
plt.title('Bar Plot of Deep Learning Model Sizes')
plt.xlabel('Model')
plt.ylabel('Size (log scale)')

# Set x-axis labels
plt.xticks(x, labels)

# Set y-axis to log scale
#plt.yscale('log')

# Specify the directory and filename
directory = r"C:\Users\rausc\Documents\EMBL\data\nema-results"  # Replace with your desired directory
filename = 'model_sizes.png'  # Replace with your desired filename

# Create the full path
save_location = os.path.join(directory, filename)

# Save the plot to the specified location
plt.savefig(save_location)

# Show the plot
plt.show()


