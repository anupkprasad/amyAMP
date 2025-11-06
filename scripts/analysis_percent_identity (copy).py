import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import sys
sys.path.append(os.path.expanduser("~/workspace/amyAMP"))
from scripts.util import ProcessSeqs, get_batchAlignment


# Define paths
path_data = os.path.expanduser("~/workspace/amyAMP/")
path_result = os.path.expanduser("~/workspace/amyAMP/results/")
path_model = os.path.expanduser("~/workspace/amyAMP/scripts/model_saved_231206/")

# Define FASTA files
fasta_AMPs = [path_data + "data_master/amps/dbaasp/dbaasp_APR_processed.fasta"]
fasta_generated = [path_model + "results_2/seqs_generated1000.fasta"]
fasta_random = [path_data + "data_master/random_pep_uni.fasta"]

# Combine datasets into a single list
fasta_all = fasta_generated + fasta_AMPs + fasta_random

# Identity plot
seqs1 = ProcessSeqs(fasta_all[0]).get_seqs()  # Generated sequences
identity_l = []
for fasta in fasta_all[1:]:  # Compare with AMPs and Random peptides
    identity = get_batchAlignment(seqs1=seqs1, seqs2=ProcessSeqs(fasta).get_seqs())
    identity_l.append(identity)

# Plot multi-histogram for the datasets
data_list = [identity_l[i]["score"][:, 2] for i in range(2)]  # AMPs and Random peptides
data_list = np.array(data_list).astype(float)

def plot_multihist(data_list, save_file, bins=70):
    """
    Plot multiple histograms for percent identity.
    """
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 4), constrained_layout=True, dpi=300)
    mini, maxi = min([min(l) for l in data_list]), max([max(l) for l in data_list])
    labels = ["AMPs", "Random-peptides"]
    for i, data_l in enumerate(data_list):
        mu = np.mean(data_l)
        var = np.var(data_l)
        axs[i].set_ylabel("Peptide number", fontsize=8)
        axs[i].set_xlabel("Identity (%)", fontsize=8)
        axs[i].hist(data_l, color='blue', edgecolor='red', bins=bins)
        axs[i].set_xlim(mini, maxi)
        axs[i].text(0.57, 0.7, labels[i] + "\n" + r"$\bar{x}$ = " + str(int(mu)) +
                    "\n" + r"$\sigma^2$ = " + str(int(var)), transform=axs[i].transAxes, fontsize=8)
    plt.savefig(save_file)
    plt.show()

# Plot the multi-histogram
plot_multihist(data_list, path_result + "identity_multihist.png")

# Top similarity with AMPs
top_matched = identity_l[0]['topmatched']  # Top matches with AMPs
for t in top_matched:
    alignments = pairwise2.align.globalxx(t[0], t[1])
    print(format_alignment(*alignments[0]))

# Similarity of time series of generated sequences
epochi = list(range(99, 1000, 100))
epochf = list(range(1199, 40000, 2000))
epoch_l = epochi + epochf
epoch_axis = list(range(1, 10)) + list(range(10, 410, 20)) + [400]
epoch_axis = [0] + epoch_axis[0:-1:2] + [400]
labels = ["AMPs", "Random"]

collected_seqs = {}
for r in range(2):  # Assuming `run_num` is 1
    collection = torch.load(path_model + 'collectedseqs_loss_epochinfo_r' + str(r + 1) + '.json')
    collected_seqs.update(collection["collected_seqs"])

for idx, fasta in enumerate(fasta_all[1:]):  # Compare with AMPs and Random peptides
    identity_tl = []
    seqs2 = ProcessSeqs(fasta).get_seqs()
    for epoch in epoch_l:
        seqs1 = collected_seqs[epoch]
        identity = get_batchAlignment(seqs1=seqs1, seqs2=seqs2)
        identity_tl.append(identity["score"][:, 2])
    identity_tl = np.array(identity_tl).astype(float).T

    # Create boxplot
    fig, axs = plt.subplots(1, 1, figsize=(3.5, 2), constrained_layout=True, dpi=300)
    axs.boxplot(identity_tl, medianprops=dict(color="red", linewidth=1.5),
                flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'})
    axs.set_xlim([0, 31])
    axs.xaxis.set_major_locator(MultipleLocator(2))
    axs.set_xticklabels(epoch_axis)
    axs.set_xlabel(r"Epochs ($\times$100)", fontsize=8)
    axs.set_ylabel("Identity (%)", fontsize=8)
    fig.savefig(path_result + "timeseries_boxplot_" + labels[idx] + ".png")





