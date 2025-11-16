import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import necessary libraries
import sys
sys.path.append(os.path.expanduser("~/workspace/amyAMP/"))
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from Bio import pairwise2
from scripts.util import ProcessSeqs, get_batchAlignment


def plot_hist(data, save_file, bins = 70):
    fig, axs = plt.subplots(1,1, figsize=(3.5, 2), constrained_layout =True, dpi = 300)
    label = ["Uperin", "AMYs", "Random-peptides"]
    data_l = [data]  ## list of list
    for i, data_l in enumerate(data_l):
        mu = np.mean(data_l)
        var = np.var(data_l)
        axs.set_ylabel("Peptide number")
        axs.set_xlabel("Identity (%)")
        axs.hist(data_l, color = 'blue', edgecolor = 'red', bins = bins)
        axs.text(0.57,0.7, label[i] +"\n" +  r"$\bar{x}$ = "+str(int(mu)) +"\n" +  r"$\sigma^2$ = "+ str(int(var)), transform=axs.transAxes, fontsize = 8)
    plt.savefig(save_file)


def plot_multihist(data_list, save_file, bins = 70):
    fig, axs = plt.subplots(3,1, figsize=(3.5, 6), constrained_layout =True, dpi = 300)
    mini, maxi = min([min(l) for l in data_list]), max([max(l) for l in data_list])
    label = ["AMPs", "AMYs", "Random-peptides"]
    for i, data_l in enumerate(data_list):
        mu = np.mean(data_l)
        var = np.var(data_l)
        axs[i].set_ylabel("Peptide number", fontsize = 8)
        axs[i].set_xlabel("Identity (%)", fontsize = 8)
        axs[i].hist(data_l, color = 'blue', edgecolor = 'red', bins = bins)
        axs[i].set_xlim(mini, maxi)
        axs[i].text(0.57,0.7, label[i] +"\n" +  r"$\bar{x}$ = "+str(int(mu)) +"\n" +  r"$\sigma^2$ = "+ str(int(var)), transform=axs[i].transAxes, fontsize = 8)
    plt.savefig(save_file)




# Define paths
path_data = os.path.expanduser("~/workspace/amyAMP/")
path_result = os.path.expanduser("~/workspace/amyAMP/results/")
path_model = os.path.expanduser("~/workspace/amyAMP/model_saved/")


# Define FASTA files
fasta_amp = [path_result + "sequence/seqs_realAMPs1000.fasta"]
fasta_amy = [path_result + "sequence/seqs_realAMYs1000.fasta"]
fasta_random = [path_result + "sequence/random_peptides_1000.fasta"]


collected_seqs = {}
for r in range(2):  # `run_num: training run` is 2
    collection = torch.load(path_model + 'collectedseqs_loss_epochinfo_r' + str(r + 1) + '.json')
    collected_seqs.update(collection["collected_seqs"])

# Select epochs exponentially from all available epochs
all_epochs = sorted(collected_seqs.keys())  # Get all available epochs
selected_epochs = [all_epochs[0]]  # Start with the first epoch

# Select epochs as powers of 2, ensuring they exist in all_epochs
while selected_epochs[-1] * 2 <= all_epochs[-1]:
    next_epoch = min(all_epochs, key=lambda x: abs(x - selected_epochs[-1] * 2))
    selected_epochs.append(next_epoch)

# Add specific epochs near 30000, 35000, and the last epoch
additional_epochs = [29999, 34999, all_epochs[-1]]
for epoch in additional_epochs:
    if epoch in all_epochs and epoch not in selected_epochs:
        selected_epochs.append(epoch)

# Sort the selected epochs
selected_epochs = sorted(selected_epochs)

# Debugging: Print the selected epochs
print("Selected epochs (updated):", selected_epochs)

# Process reference datasets
seqs_amp = ProcessSeqs(fasta_amp[0]).get_seqs()
seqs_amy = ProcessSeqs(fasta_amy[0]).get_seqs()
seqs_random = ProcessSeqs(fasta_random[0]).get_seqs()

# Initialize lists to store percent identity values
identity_amp = []
identity_amy = []
identity_random = []

# Calculate percent identity for selected epochs
for epoch in selected_epochs:
    if epoch not in collected_seqs:
        continue
    seqs_generated = collected_seqs[epoch]
    
    # Compare with AMP dataset
    identity_amp_epoch = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_amp)["score"][:, 2]
    identity_amp_epoch = np.array(identity_amp_epoch, dtype=float)  # Ensure numeric conversion
    identity_amp.append(identity_amp_epoch)  # Store all percent identity values
    
    # Compare with AMY dataset
    identity_amy_epoch = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_amy)["score"][:, 2]
    identity_amy_epoch = np.array(identity_amy_epoch, dtype=float)  # Ensure numeric conversion
    identity_amy.append(identity_amy_epoch)  # Store all percent identity values
    
    # Compare with random dataset
    identity_rand_epoch = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_random)["score"][:, 2]
    identity_rand_epoch = np.array(identity_rand_epoch, dtype=float)  # Ensure numeric conversion
    identity_random.append(identity_rand_epoch)  # Store all percent identity values

# Plot percent identity over epochs as box plots
fig, axs = plt.subplots(3, 1, figsize=(5, 10), dpi=600)  # 3 rows, 1 column

# Increment x-tick labels by +1 for customization
custom_xticks = [tick + 1 for tick in selected_epochs]

# Plot for AMP dataset
axs[0].boxplot(identity_amp, positions=range(len(selected_epochs)), patch_artist=True,
               medianprops=dict(color="red", linewidth=1.5),
               flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'},
               boxprops=dict(facecolor='lightblue', color='blue', linewidth=1.5),
               whiskerprops=dict(color='blue', linewidth=1.5),
               capprops=dict(color='blue', linewidth=1.5))
axs[0].set_xticks(range(len(selected_epochs)))
axs[0].set_xticklabels(custom_xticks, rotation=25, fontsize=10)
axs[0].set_xlabel("Epochs", fontsize=10)
axs[0].set_ylabel("Percent Identity (%)", fontsize=10)
axs[0].set_title("(a) amyAMP similarity w.r.t. AMP", loc='left', fontweight='bold', fontsize=12)
axs[0].grid(alpha=0.3, linestyle='--')

# Plot for AMY dataset
axs[1].boxplot(identity_amy, positions=range(len(selected_epochs)), patch_artist=True,
               medianprops=dict(color="red", linewidth=1.5),
               flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'},
               boxprops=dict(facecolor='lightgreen', color='green', linewidth=1.5),
               whiskerprops=dict(color='green', linewidth=1.5),
               capprops=dict(color='green', linewidth=1.5))
axs[1].set_xticks(range(len(selected_epochs)))
axs[1].set_xticklabels(custom_xticks, rotation=25, fontsize=10)
axs[1].set_xlabel("Epochs", fontsize=10)
axs[1].set_ylabel("Percent Identity (%)", fontsize=10)
axs[1].set_title("(b) amyAMP similarity w.r.t. AMY", loc='left', fontweight='bold', fontsize=12)
axs[1].grid(alpha=0.3, linestyle='--')

# Plot for random dataset
axs[2].boxplot(identity_random, positions=range(len(selected_epochs)), patch_artist=True,
               medianprops=dict(color="red", linewidth=1.5),
               flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'},
               boxprops=dict(facecolor='lightcoral', color='red', linewidth=1.5),
               whiskerprops=dict(color='red', linewidth=1.5),
               capprops=dict(color='red', linewidth=1.5))
axs[2].set_xticks(range(len(selected_epochs)))
axs[2].set_xticklabels(custom_xticks, rotation=25, fontsize=10)
axs[2].set_xlabel("Epochs", fontsize=10)
axs[2].set_ylabel("Percent Identity (%)", fontsize=10)
axs[2].set_title("(c) amyAMP similarity w.r.t. randPep", loc='left', fontweight='bold', fontsize=12)
axs[2].grid(alpha=0.3, linestyle='--')

for ax in axs:
    ax.set_ylim(30, 80)  # Set y-axis limits for better visualization
# Save and show the plot
plt.tight_layout()
plt.savefig(path_result + "percent_identity_over_epochs_boxplot_three_datasets.png", dpi=300, bbox_inches='tight')
plt.show()


################# histogram plots for last epoch ####################
# Extract data for the last epoch
last_epoch_index = -1  # Index for the last epoch
identity_amp_last = identity_amp[last_epoch_index]
identity_amy_last = identity_amy[last_epoch_index]
identity_rand_last = identity_random[last_epoch_index]

# Calculate mean and standard deviation for the last epoch
mean_amp = np.mean(identity_amp_last)
std_amp = np.std(identity_amp_last)
mean_amy = np.mean(identity_amy_last)
std_amy = np.std(identity_amy_last)
mean_rand = np.mean(identity_rand_last)
std_rand = np.std(identity_rand_last)

# Create distribution plots
fig, axs = plt.subplots(3, 1, figsize=(4, 10), dpi=600, constrained_layout=True)  # 3 rows, 1 column

# Distribution plot for AMP dataset
bins_amp = np.linspace(0, 100, 41)  # 40 bins from 0 to 100 for finer granularity
counts_amp, edges_amp = np.histogram(identity_amp_last, bins=bins_amp)
axs[0].bar(edges_amp[:-1], counts_amp, width=np.diff(edges_amp), color='lightblue', edgecolor='blue')
axs[0].set_xlabel("Percent Identity (%)", fontsize=10)
axs[0].set_ylabel("Peptide Number", fontsize=10)
axs[0].set_title("(d)", loc='left', fontweight='bold', fontsize=12)
axs[0].annotate(f"Mean: {mean_amp:.2f}\nStd: {std_amp:.2f}", xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=8, color='blue', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='blue'))
axs[0].grid(alpha=0.3, linestyle='--')

# Distribution plot for AMY dataset
bins_amy = np.linspace(0, 100, 41)  # 40 bins from 0 to 100 for finer granularity
counts_amy, edges_amy = np.histogram(identity_amy_last, bins=bins_amy)
axs[1].bar(edges_amy[:-1], counts_amy, width=np.diff(edges_amy), color='lightgreen', edgecolor='green')
axs[1].set_xlabel("Percent Identity (%)", fontsize=10)
axs[1].set_ylabel("Peptide Number", fontsize=10)
axs[1].set_title("(e)", loc='left', fontweight='bold', fontsize=12)
axs[1].annotate(f"Mean: {mean_amy:.2f}\nStd: {std_amy:.2f}", xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=8, color='green', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='green'))
axs[1].grid(alpha=0.3, linestyle='--')

# Distribution plot for random dataset
bins_rand = np.linspace(0, 100, 41)  # 40 bins from 0 to 100 for finer granularity
counts_rand, edges_rand = np.histogram(identity_rand_last, bins=bins_rand)
axs[2].bar(edges_rand[:-1], counts_rand, width=np.diff(edges_rand), color='lightcoral', edgecolor='red')
axs[2].set_xlabel("Percent Identity (%)", fontsize=10)
axs[2].set_ylabel("Peptide Number", fontsize=10)
axs[2].set_title("(f)", loc='left', fontweight='bold', fontsize=12)
axs[2].annotate(f"Mean: {mean_rand:.2f}\nStd: {std_rand:.2f}", xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=8, color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red'))
axs[2].grid(alpha=0.3, linestyle='--')

# Save and show the plot
plt.tight_layout()
plt.savefig(path_result + "percent_identity_distribution_last_epoch_fixed.png", dpi=300, bbox_inches='tight')
plt.show()

