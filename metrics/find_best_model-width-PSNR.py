import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

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

def aggregate_psnr_scores(psnr_files):
    aggregated_psnr_scores = defaultdict(list)
    aggregated_psnr_std_devs = defaultdict(list)
    
    for psnr_file in psnr_files:
        psnr_labels, psnr_scores, psnr_std_devs = parse_scores_file(psnr_file, "PSNR")

        for label, psnr_score, psnr_std_dev in zip(psnr_labels, psnr_scores, psnr_std_devs):
            second_number = label.split()[1]
            aggregated_psnr_scores[second_number].append(psnr_score)
            aggregated_psnr_std_devs[second_number].append(psnr_std_dev)

    return aggregated_psnr_scores, aggregated_psnr_std_devs

def calculate_best_psnr(psnr_files):
    aggregated_psnr_scores, aggregated_psnr_std_devs = aggregate_psnr_scores(psnr_files)
    
    # Calculate mean scores and standard deviations for each second number
    mean_psnr_scores = {label: np.mean(scores) for label, scores in aggregated_psnr_scores.items()}
    mean_psnr_std_devs = {label: np.mean(std_devs) for label, std_devs in aggregated_psnr_std_devs.items()}

    # Find the best score and corresponding label
    best_label = max(mean_psnr_scores, key=mean_psnr_scores.get)
    best_score = mean_psnr_scores[best_label]

    return best_label, best_score, mean_psnr_scores, mean_psnr_std_devs

def plot_psnr_scores(mean_psnr_scores, mean_psnr_std_devs, output_path):
    labels = list(mean_psnr_scores.keys())
    scores = list(mean_psnr_scores.values())
    std_devs = list(mean_psnr_std_devs.values())
    
    plt.figure(figsize=(14, 10))
    plt.errorbar(labels, scores, yerr=std_devs, fmt='o', ecolor='darkblue', capsize=5, marker='o', markersize=8, linestyle='None', color='darkblue')
    plt.ylabel('Mean PSNR Score', fontsize=14)
    plt.xlabel('Second Number in Labels', fontsize=14)
    plt.title('Mean PSNR Scores for Different Second Numbers in Labels with Error Bars', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

# Example usage
psnr_files = [
    r"C:\Users\rausc\Documents\EMBL\data\nema-results\psnr_scores-nema-single-1.txt",
    r"C:\Users\rausc\Documents\EMBL\data\droso-results\psnr_scores-droso-single-1.txt",
    r"C:\Users\rausc\Documents\EMBL\data\mouse-results\psnr_scores-mouse-single-1.txt"
]

best_label, best_score, mean_psnr_scores, mean_psnr_std_devs = calculate_best_psnr(psnr_files)
print(f"Best Second Number: {best_label}")
print(f"Best Score: {best_score}")
print(f"Mean PSNR Scores: {mean_psnr_scores}")
print(f"Mean PSNR Standard Deviations: {mean_psnr_std_devs}")

# Plotting the results
output_path = r"C:\Users\rausc\Documents\EMBL\data\general-results\single-overall-width-PSNR.png"
plot_psnr_scores(mean_psnr_scores, mean_psnr_std_devs, output_path)

