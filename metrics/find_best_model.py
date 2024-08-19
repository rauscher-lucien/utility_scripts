import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_scores_file(file_path, score_type):
    labels = []
    scores = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Label:"):
                label = line.split(":")[1].strip()
                labels.append(label)
            elif line.startswith(f"Mean {score_type}:"):
                score = float(line.split(":")[1].strip())
                scores.append(score)
    return labels, scores

def aggregate_scores(psnr_files, ssim_files):
    aggregated_psnr_scores = defaultdict(list)
    aggregated_ssim_scores = defaultdict(list)
    
    for psnr_file, ssim_file in zip(psnr_files, ssim_files):
        psnr_labels, psnr_scores = parse_scores_file(psnr_file, "PSNR")
        ssim_labels, ssim_scores = parse_scores_file(ssim_file, "SSIM")

        # Ensure the labels match
        assert psnr_labels == ssim_labels, f"Labels in PSNR and SSIM files do not match for {psnr_file} and {ssim_file}"
        
        for label, psnr_score, ssim_score in zip(psnr_labels, psnr_scores, ssim_scores):
            aggregated_psnr_scores[label].append(psnr_score)
            aggregated_ssim_scores[label].append(ssim_score)

    return aggregated_psnr_scores, aggregated_ssim_scores

def calculate_overall_best(psnr_files, ssim_files):
    aggregated_psnr_scores, aggregated_ssim_scores = aggregate_scores(psnr_files, ssim_files)
    
    # Calculate mean scores for each label
    mean_psnr_scores = {label: np.mean(scores) for label, scores in aggregated_psnr_scores.items()}
    mean_ssim_scores = {label: np.mean(scores) for label, scores in aggregated_ssim_scores.items()}

    # Normalize the scores
    min_psnr, max_psnr = get_min_max_scores(mean_psnr_scores)
    min_ssim, max_ssim = get_min_max_scores(mean_ssim_scores)

    normalized_psnr = {label: (score - min_psnr) / (max_psnr - min_psnr) for label, score in mean_psnr_scores.items()}
    normalized_ssim = {label: (score - min_ssim) / (max_ssim - min_ssim) for label, score in mean_ssim_scores.items()}

    # Calculate overall scores
    overall_scores = {label: (normalized_psnr[label] + normalized_ssim[label]) / 2 for label in normalized_psnr}

    # Find the best score and corresponding label
    best_label = max(overall_scores, key=overall_scores.get)
    best_score = overall_scores[best_label]

    return best_label, best_score, overall_scores

def get_min_max_scores(scores_dict):
    min_score = min(scores_dict.values())
    max_score = max(scores_dict.values())
    return min_score, max_score

def plot_overall_scores(overall_scores, output_path):
    labels = list(overall_scores.keys())
    scores = list(overall_scores.values())
    
    plt.figure(figsize=(14, 10))
    plt.bar(labels, scores, color='darkblue')
    plt.ylabel('Normalized Score', fontsize=14)
    plt.xlabel('Labels', fontsize=14)
    plt.title('Overall Normalized Scores for Different Denoising Networks', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

# Example usage
psnr_files = [
    r"C:\Users\rausc\Documents\EMBL\data\nema-results\psnr_scores-nema-general-1.txt",
    r"C:\Users\rausc\Documents\EMBL\data\droso-results\psnr_scores-droso-general-1.txt",
    r"C:\Users\rausc\Documents\EMBL\data\mouse-results\psnr_scores-mouse-general-1.txt"
]
ssim_files = [
    r"C:\Users\rausc\Documents\EMBL\data\nema-results\ssim_scores-nema-general-1.txt",
    r"C:\Users\rausc\Documents\EMBL\data\droso-results\ssim_scores-droso-general-1.txt",
    r"C:\Users\rausc\Documents\EMBL\data\mouse-results\ssim_scores-mouse-general-1.txt"
    # Add more SSIM file paths as needed
]

best_label, best_score, overall_scores = calculate_overall_best(psnr_files, ssim_files)
print(f"Best Label: {best_label}")
print(f"Best Score: {best_score}")
print(f"Overall Scores: {overall_scores}")

# Plotting the results
output_path = r"C:\Users\rausc\Documents\EMBL\data\general-results\general-overall-1.png"
plot_overall_scores(overall_scores, output_path)
