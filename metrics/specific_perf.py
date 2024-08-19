import numpy as np
import matplotlib.pyplot as plt

def plot_mean_psnr_scores(labels, all_scores, all_std_devs, output_path):
    mean_scores = [np.mean(scores) for scores in all_scores]
    mean_std_devs = [np.mean(std_devs) for std_devs in all_std_devs]
    
    plt.figure(figsize=(14, 10))
    
    # Plotting with diamonds as markers, customized line thickness, and fonts
    plt.errorbar(
        labels, 
        mean_scores, 
        yerr=mean_std_devs, 
        fmt='D',  # Diamond marker
        ecolor='black',  # Error bar color
        capsize=10,  # Size of the caps on error bars
        markerfacecolor=(0/255, 101/255, 189/255),  # Blue inside the diamond
        markeredgecolor='black',  # Black border around the diamond
        markersize=20,  # Size of the diamond markers
        linestyle='None',  # No connecting line between points
        elinewidth=2  # Line thickness for the error bars
    )
    
    plt.ylabel('Mean PSNR Score', fontsize=24)
    #plt.xlabel('Labels', fontsize=24)
    #plt.title('Mean PSNR Scores for Different Denoising Networks with Error Bars', fontsize=26)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7, linewidth=1.5)  # Grid with thicker lines
    
    # Set the thickness of the plot spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)  # Save with high DPI for better quality
    plt.show()

# Example usage
labels = ['BM3D', 'trained on singular similar', 'trained on similar', 'trained on singular different', 'trained on different']
all_scores = [
    [0.9474790889613567, 0.8970727059634689, 0.869611236979446],
    [0.9513080266218862, 0.7743036977510825, 0.8575714653135367],
    [0.9487976801962692, 0.8736599217121254, 0.862397418959768], 
    [0.8502579092807034, 0.8823162855202631, 0.8321528320417336, 0.7324091273222473, 0.862299269307587, 0.7258981997565879],
    [0.9504435533520113, 0.9090573511860365, 0.8812719822222741, 0.7293513953490248, 0.8661241624061896, 0.8610628195715491]
    ]
all_std_devs = [
    [0.006860634721613761, 0.012529554279909353, 0.012419135202782882],
    [0.009108388706597527, 0.02304465767357071, 0.023599454781221264],
    [0.007330258538701675, 0.015857880571553053, 0.024108785935909847], 
    [0.009654316478680808, 0.009976970714397139, 0.018259104467261283, 0.011394083776510815, 0.012785073140227659, 0.013039569771566345],
    [0.007684295871270387, 0.007798571494388244, 0.013764813190580157, 0.0096908112337646, 0.014355877829150814, 0.023790100285201815]
    ]
output_path = r"C:\Users\rausc\Documents\EMBL\data\general-results\SSIM-specific-comp_sim_diff-4.png"

plot_mean_psnr_scores(labels, all_scores, all_std_devs, output_path)

