import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import necessary libraries
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
path_model = os.path.expanduser("~/workspace/amyAMP/scripts/model_saved_231206/")


# Define FASTA files
fasta_training = [path_data + "data_master/amps/dbaasp/dbaasp_APR_processed.fasta"]
fasta_random = [path_result + "sequence/random_peptides_1000.fasta"]


collected_seqs = {}
for r in range(2):  # Assuming `run_num` is 1
    collection = torch.load(path_data + 'model_saved_231206/collectedseqs_loss_epochinfo_r' + str(r + 1) + '.json')
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
seqs_training = ProcessSeqs(fasta_training[0]).get_seqs()
seqs_random = ProcessSeqs(fasta_random[0]).get_seqs()

# Initialize lists to store percent identity values
identity_training = []
identity_random = []

# Calculate percent identity for selected epochs
for epoch in selected_epochs:
    if epoch not in collected_seqs:
        continue
    seqs_generated = collected_seqs[epoch]
    
    # Compare with training dataset
    identity_train = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_training)["score"][:, 2]
    identity_train = np.array(identity_train, dtype=float)  # Ensure numeric conversion
    identity_training.append(identity_train)  # Store all percent identity values
    
    # Compare with random dataset
    identity_rand = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_random)["score"][:, 2]
    identity_rand = np.array(identity_rand, dtype=float)  # Ensure numeric conversion
    identity_random.append(identity_rand)  # Store all percent identity values

# Plot percent identity over epochs as box plots
fig, axs = plt.subplots(2, 1, figsize=(5, 8), dpi=600)  # 2 rows, 1 column

# Increment x-tick labels by +1 for customization
custom_xticks = [tick + 1 for tick in selected_epochs]

# Plot for training dataset
axs[0].boxplot(identity_training, positions=range(len(selected_epochs)), patch_artist=True,
               medianprops=dict(color="red", linewidth=1.5),
               flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'},
               boxprops=dict(facecolor='lightblue', color='blue', linewidth=1.5),
               whiskerprops=dict(color='blue', linewidth=1.5),
               capprops=dict(color='blue', linewidth=1.5))
axs[0].set_xticks(range(len(selected_epochs)))
axs[0].set_xticklabels(custom_xticks, rotation=45, fontsize=8)
axs[0].set_xlabel("Epochs", fontsize=10)
axs[0].set_ylabel("Percent Identity (%)", fontsize=10)
axs[0].set_title("(a) amyAMP percent identity with trainPep", loc='left', fontweight='bold', fontsize=12)
axs[0].grid(alpha=0.3, linestyle='--')

# Plot for random dataset
axs[1].boxplot(identity_random, positions=range(len(selected_epochs)), patch_artist=True,
               medianprops=dict(color="red", linewidth=1.5),
               flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'},
               boxprops=dict(facecolor='lightcoral', color='red', linewidth=1.5),
               whiskerprops=dict(color='red', linewidth=1.5),
               capprops=dict(color='red', linewidth=1.5))
axs[1].set_xticks(range(len(selected_epochs)))
axs[1].set_xticklabels(custom_xticks, rotation=45, fontsize=8)
axs[1].set_xlabel("Epochs", fontsize=10)
axs[1].set_ylabel("Percent Identity (%)", fontsize=10)
axs[1].set_title("(b) amyAMP percent identity with randPep", loc='left', fontweight='bold', fontsize=12)
axs[1].grid(alpha=0.3, linestyle='--')

# Save and show the plot
plt.tight_layout()
plt.savefig(path_result + "percent_identity_over_epochs_boxplot.png", dpi=300, bbox_inches='tight')
plt.show()


################# distribution plots for last epoch ####################

gen_pep = path_result + "sequence/seqs_generated1000.fasta"
# Extract data for the last epoch
last_epoch_index = -1  # Index for the last epoch
identity_train_last = identity_training[last_epoch_index]
identity_rand_last = identity_random[last_epoch_index]

# Calculate mean and standard deviation for the last epoch
mean_train = np.mean(identity_train_last)
std_train = np.std(identity_train_last)
mean_rand = np.mean(identity_rand_last)
std_rand = np.std(identity_rand_last)

# Create distribution plots
fig, axs = plt.subplots(2, 1, figsize=(4, 8), dpi=300)  # 2 rows, 1 column

# Distribution plot for training dataset
bins_train = np.linspace(0, 100, 41)  # 40 bins from 0 to 100 for finer granularity
counts_train, edges_train = np.histogram(identity_train_last, bins=bins_train)
axs[0].bar(edges_train[:-1], counts_train, width=np.diff(edges_train), color='lightblue', edgecolor='blue')
axs[0].set_xlabel("Percent Identity (%)", fontsize=10)
axs[0].set_ylabel("Peptide Number", fontsize=10)
axs[0].set_title("(c) trainPep", loc='left', fontweight='bold', fontsize=12)
axs[0].text(0.7, 0.9, f"Mean: {mean_train:.2f}\nStd: {std_train:.2f}", transform=axs[0].transAxes, fontsize=8, color='blue')
axs[0].grid(alpha=0.3, linestyle='--')

# Distribution plot for random dataset
bins_rand = np.linspace(0, 100, 41)  # 40 bins from 0 to 100 for finer granularity
counts_rand, edges_rand = np.histogram(identity_rand_last, bins=bins_rand)
axs[1].bar(edges_rand[:-1], counts_rand, width=np.diff(edges_rand), color='lightcoral', edgecolor='red')
axs[1].set_xlabel("Percent Identity (%)", fontsize=10)
axs[1].set_ylabel("Peptide Number", fontsize=10)
axs[1].set_title("(d) randPep", loc='left', fontweight='bold', fontsize=12)
axs[1].text(0.7, 0.9, f"Mean: {mean_rand:.2f}\nStd: {std_rand:.2f}", transform=axs[1].transAxes, fontsize=8, color='red')
axs[1].grid(alpha=0.3, linestyle='--')

# Save and show the plot
plt.tight_layout()
plt.savefig(path_result + "percent_identity_distribution_last_epoch.png", dpi=300, bbox_inches='tight')
plt.show()


# Read peptides from gen_pep FASTA file
gen_pep_seqs = ProcessSeqs(gen_pep).get_seqs()

# Calculate percent identity for gen_pep peptides
identity_train_gen_pep = get_batchAlignment(seqs1=gen_pep_seqs, seqs2=seqs_training)["score"][:, 2]
identity_rand_gen_pep = get_batchAlignment(seqs1=gen_pep_seqs, seqs2=seqs_random)["score"][:, 2]

# Debugging: Print the output of get_batchAlignment to check data types
print("identity_train_gen_pep (raw):", identity_train_gen_pep)
print("identity_rand_gen_pep (raw):", identity_rand_gen_pep)

# Ensure numeric conversion
try:
    identity_train_gen_pep = np.array(identity_train_gen_pep, dtype=float)
    identity_rand_gen_pep = np.array(identity_rand_gen_pep, dtype=float)
except ValueError as e:
    print("Error converting percent identity to float:", e)
    print("identity_train_gen_pep:", identity_train_gen_pep)
    print("identity_rand_gen_pep:", identity_rand_gen_pep)
    raise

# Calculate mean and standard deviation for gen_pep peptides
mean_train = np.mean(identity_train_gen_pep)
std_train = np.std(identity_train_gen_pep)
mean_rand = np.mean(identity_rand_gen_pep)
std_rand = np.std(identity_rand_gen_pep)

# Create distribution plots
fig, axs = plt.subplots(2, 1, figsize=(4, 8), dpi=300)  # 2 rows, 1 column

# Distribution plot for training dataset
bins_train = np.linspace(0, 100, 41)  # 40 bins from 0 to 100 for finer granularity
counts_train, edges_train = np.histogram(identity_train_gen_pep, bins=bins_train)
axs[0].bar(edges_train[:-1], counts_train, width=np.diff(edges_train), color='lightblue', edgecolor='blue')
axs[0].set_xlabel("Percent Identity (%)", fontsize=10)
axs[0].set_ylabel("Peptide Number", fontsize=10)
axs[0].set_title("(c) trainPep", loc='left', fontweight='bold', fontsize=12)
axs[0].text(0.7, 0.9, f"Mean: {mean_train:.2f}\nStd: {std_train:.2f}", transform=axs[0].transAxes, fontsize=8, color='blue')
axs[0].grid(alpha=0.3, linestyle='--')

# Distribution plot for random dataset
bins_rand = np.linspace(0, 100, 41)  # 40 bins from 0 to 100 for finer granularity
counts_rand, edges_rand = np.histogram(identity_rand_gen_pep, bins=bins_rand)
axs[1].bar(edges_rand[:-1], counts_rand, width=np.diff(edges_rand), color='lightcoral', edgecolor='red')
axs[1].set_xlabel("Percent Identity (%)", fontsize=10)
axs[1].set_ylabel("Peptide Number", fontsize=10)
axs[1].set_title("(d) randPep", loc='left', fontweight='bold', fontsize=12)
axs[1].text(0.7, 0.9, f"Mean: {mean_rand:.2f}\nStd: {std_rand:.2f}", transform=axs[1].transAxes, fontsize=8, color='red')
axs[1].grid(alpha=0.3, linestyle='--')

# Save and show the plot
plt.tight_layout()
plt.savefig(path_result + "percent_identity_distribution_gen_pep.png", dpi=300, bbox_inches='tight')
plt.show()


