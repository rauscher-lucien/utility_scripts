import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os

def parse_scores_file(file_path, score_type):
    labels = []
    scores = []
    std_devs = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Label:"):
                label = line.split(":")[1].strip()
                labels.append(label)
            elif line.startswith(f"Mean {score_type}:"):
                score = float(line.split(":")[1].strip())
                scores.append(score)
            elif line.startswith("Standard Deviation:"):
                std_dev = float(line.split(":")[1].strip())
                std_devs.append(std_dev)
    return labels, scores, std_devs

def aggregate_ssim_scores(ssim_files):
    aggregated_ssim_scores = defaultdict(list)
    aggregated_ssim_std_devs = defaultdict(list)
    
    for ssim_file in ssim_files:
        ssim_labels, ssim_scores, ssim_std_devs = parse_scores_file(ssim_file, "SSIM")

        for label, ssim_score, ssim_std_dev in zip(ssim_labels, ssim_scores, ssim_std_devs):
            aggregated_ssim_scores[label].append(ssim_score)
            aggregated_ssim_std_devs[label].append(ssim_std_dev)

    return aggregated_ssim_scores, aggregated_ssim_std_devs

def calculate_best_ssim(ssim_files):
    aggregated_ssim_scores, aggregated_ssim_std_devs = aggregate_ssim_scores(ssim_files)
    
    # Calculate mean scores and standard deviations for each label
    mean_ssim_scores = {label: np.mean(scores) for label, scores in aggregated_ssim_scores.items()}
    mean_ssim_std_devs = {label: np.mean(std_devs) for label, std_devs in aggregated_ssim_std_devs.items()}

    # Find the best score and corresponding label
    best_label = max(mean_ssim_scores, key=mean_ssim_scores.get)
    best_score = mean_ssim_scores[best_label]

    return best_label, best_score, mean_ssim_scores, mean_ssim_std_devs

def plot_ssim_scores(mean_ssim_scores, mean_ssim_std_devs, output_path):
    labels = list(mean_ssim_scores.keys())
    scores = list(mean_ssim_scores.values())
    std_devs = list(mean_ssim_std_devs.values())
    
    plt.figure(figsize=(14, 10))
    plt.errorbar(labels, scores, yerr=std_devs, fmt='o', ecolor='darkblue', capsize=5, marker='o', markersize=8, linestyle='None', color='darkblue')
    plt.ylabel('Mean SSIM Score', fontsize=14)
    plt.xlabel('Labels', fontsize=14)
    plt.title('Mean SSIM Scores for Different Denoising Networks with Error Bars', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def save_ssim_scores_to_file(mean_ssim_scores, mean_ssim_std_devs, output_path):
    # Change the file extension from .png to .txt
    txt_file_path = os.path.splitext(output_path)[0] + ".txt"
    
    with open(txt_file_path, 'w') as file:
        file.write("Label\tMean SSIM Score\tStandard Deviation\n")
        for label in mean_ssim_scores:
            file.write(f"{label}\t{mean_ssim_scores[label]:.4f}\t{mean_ssim_std_devs[label]:.4f}\n")
    
    print(f"SSIM scores saved to {txt_file_path}")

# Example usage
ssim_files = [
    r"C:\Users\rausc\Documents\EMBL\data\nema-results\ssim_scores-nema-2D-general-2.txt",
    r"C:\Users\rausc\Documents\EMBL\data\droso-results\ssim_scores-droso-2D-general-2.txt",
    r"C:\Users\rausc\Documents\EMBL\data\mouse-results\ssim_scores-mouse-2D-general-2.txt"
    # Add more PSNR file paths as needed
]

best_label, best_score, mean_ssim_scores, mean_ssim_std_devs = calculate_best_ssim(ssim_files)
print(f"Best Label: {best_label}")
print(f"Best Score: {best_score}")
print(f"Mean SSIM Scores: {mean_ssim_scores}")
print(f"Mean SSIM Standard Deviations: {mean_ssim_std_devs}")

# Plotting the results
output_path = r"C:\Users\rausc\Documents\EMBL\data\general-results\2D-general-overall-SSIM-1.png"
plot_ssim_scores(mean_ssim_scores, mean_ssim_std_devs, output_path)

# Saving the results to a text file
save_ssim_scores_to_file(mean_ssim_scores, mean_ssim_std_devs, output_path)

