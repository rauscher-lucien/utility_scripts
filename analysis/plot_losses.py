import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def smooth_values(values, smoothing_factor=0.9):
    smoothed_values = []
    for i, value in enumerate(values):
        if i == 0:
            smoothed_values.append(value)
        else:
            smoothed_value = smoothing_factor * smoothed_values[-1] + (1 - smoothing_factor) * value
            smoothed_values.append(smoothed_value)
    return smoothed_values

def plot_multiple_tensorboard_logs(logdirs, labels, output_dir, x_range=None, y_range=None, smoothing_factor=0.9):
    if len(logdirs) != len(labels):
        raise ValueError("The number of log directories must match the number of labels.")

    plt.figure(figsize=(10, 5))

    for logdir, label in zip(logdirs, labels):
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()

        # Retrieve loss data from TensorBoard logs
        training_loss = event_acc.Scalars('Loss/train')

        # Extract the steps and values
        train_steps = [x.step for x in training_loss]
        train_values = [x.value for x in training_loss]

        # Smooth the values
        train_values_smoothed = smooth_values(train_values, smoothing_factor)

        # Plot the losses
        plt.plot(train_steps, train_values_smoothed, label=f'Train Loss - {label}')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    if x_range:
        plt.xlim(x_range)

    if y_range:
        plt.ylim(y_range)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = 'loss_plot-nema-3.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {os.path.join(output_dir, plot_filename)}")

if __name__ == '__main__':
    logdirs = [
        r"\\tier2.embl.de\prevedel\members\Rauscher\final_projects\2D-N2N-single_volume\test_1_Nematostella_B_model_nameUNet5_UNet_base16_num_epoch10000_batch_size8_lr1e-05\results\tensorboard_logs",
        r"\\tier2.embl.de\prevedel\members\Rauscher\final_projects\2D-N2N-single_volume\test_1_Nematostella_B_model_nameUNet5_UNet_base32_num_epoch10000_batch_size8_lr1e-05\results\tensorboard_logs",
        r"\\tier2.embl.de\prevedel\members\Rauscher\final_projects\2D-N2N-single_volume\test_1_Nematostella_B_model_nameUNet5_UNet_base64_num_epoch10000_batch_size8_lr1e-05\results\tensorboard_logs"
        # Add more log directories as needed
    ]

    labels = [
        '2D-N2N - 5 16',
        '2D-N2N - 5 32',
        '2D-N2N -  5 64'
        # Add more labels corresponding to the log directories
    ]

    output_dir = r"C:\Users\rausc\Documents\EMBL\data\nema-results"

    # Set the desired x-range and y-range here
    x_range = (0, 10000)  # Example range, adjust as needed
    y_range = (0.59, 0.65)  # Example range, adjust as needed

    # Smoothing factor for the loss values
    smoothing_factor = 0.99  # Adjust as needed

    plot_multiple_tensorboard_logs(logdirs, labels, output_dir, x_range, y_range, smoothing_factor)



