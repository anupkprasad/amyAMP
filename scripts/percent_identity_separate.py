"""

Percent identity analysis for generated peptides across training epochs.

This script calculates and visualizes the sequence similarity between
generated peptides and reference datasets (AMPs, AMYs, and random peptides)
at different training epochs.

Features:
1. Box plots showing identity distribution across epochs
2. Histogram plots for the final epoch
3. Statistical analysis (mean, std) for each comparison

"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to system path (relative to this file)
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_ROOT not in sys.path:
    sys.path.append(PATH_ROOT)

# Import project modules
from scripts.util import ProcessSeqs, get_batchAlignment
from scripts import plotStyle
plotStyle.setPlotStyle()

# ==============================================================================
# Configuration
# ==============================================================================

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (relative to project root)
PATH_DATA = os.path.join(PATH_ROOT, "data_master")
PATH_RESULT = os.path.join(PATH_ROOT, "results")
PATH_MODEL = os.path.join(PATH_ROOT, "model_saved")

# FASTA file paths
FASTA_AMP = os.path.join(PATH_RESULT, "sequence", "seqs_realAMPs1000.fasta")
FASTA_AMY = os.path.join(PATH_RESULT, "sequence", "seqs_realAMYs1000.fasta")
FASTA_RANDOM = os.path.join(PATH_RESULT, "sequence", "random_peptides_1000.fasta")

# Training parameters
RUN_NUM = 2  # Number of training runs
NUM_BINS = 41  # Number of bins for histogram (0-100 in 40 bins)


# ==============================================================================
# Epoch Selection
# ==============================================================================

def select_epochs_exponentially(all_epochs):
    """
    Select epochs exponentially (powers of 2) plus specific important epochs.
    
    Args:
        all_epochs (list): List of all available epochs
    
    Returns:
        list: Selected epochs sorted in ascending order
    """
    selected_epochs = [all_epochs[0]]  # Start with the first epoch
    
    # Select epochs as powers of 2
    while selected_epochs[-1] * 2 <= all_epochs[-1]:
        next_epoch = min(all_epochs, key=lambda x: abs(x - selected_epochs[-1] * 2))
        if next_epoch not in selected_epochs:
            selected_epochs.append(next_epoch)
    
    # Add specific important epochs
    additional_epochs = [29999, 34999, all_epochs[-1]]
    for epoch in additional_epochs:
        if epoch in all_epochs and epoch not in selected_epochs:
            selected_epochs.append(epoch)
    
    # Sort and return
    selected_epochs = sorted(selected_epochs)
    print(f"Selected {len(selected_epochs)} epochs for analysis")
    print(f"Epochs: {selected_epochs}")
    
    return selected_epochs


# ==============================================================================
# Data Loading
# ==============================================================================

def load_collected_sequences(path_model, run_num):
    """
    Load collected sequences from all training runs.
    
    Args:
        path_model (str): Path to model directory
        run_num (int): Number of training runs
    
    Returns:
        dict: Dictionary mapping epoch to generated sequences
    """
    print("=" * 60)
    print("Loading Collected Sequences")
    print("=" * 60)
    
    collected_seqs = {}
    for r in range(run_num):
        checkpoint_file = os.path.join(
            path_model,
            f'collectedseqs_loss_epochinfo_r{r + 1}.json'
        )
        
        if not os.path.exists(checkpoint_file):
            print(f"Warning: Checkpoint file not found: {checkpoint_file}")
            continue
        
        collection = torch.load(checkpoint_file, map_location=DEVICE)
        collected_seqs.update(collection["collected_seqs"])
        print(f"Loaded sequences from run {r + 1}")
    
    print(f"Total epochs loaded: {len(collected_seqs)}")
    return collected_seqs


def load_reference_sequences():
    """
    Load reference sequences (AMP, AMY, random).
    
    Returns:
        tuple: (seqs_amp, seqs_amy, seqs_random)
    """
    print("\n" + "=" * 60)
    print("Loading Reference Sequences")
    print("=" * 60)
    
    seqs_amp = ProcessSeqs(FASTA_AMP).get_seqs()
    seqs_amy = ProcessSeqs(FASTA_AMY).get_seqs()
    seqs_random = ProcessSeqs(FASTA_RANDOM).get_seqs()
    
    print(f"AMP sequences: {len(seqs_amp)}")
    print(f"AMY sequences: {len(seqs_amy)}")
    print(f"Random sequences: {len(seqs_random)}")
    
    return seqs_amp, seqs_amy, seqs_random


# ==============================================================================
# Identity Calculation
# ==============================================================================

def calculate_identity_over_epochs(collected_seqs, selected_epochs, seqs_amp, seqs_amy, seqs_random):
    """
    Calculate percent identity for selected epochs against reference datasets.
    
    Args:
        collected_seqs (dict): Collected sequences from training
        selected_epochs (list): List of epochs to analyze
        seqs_amp (dict): AMP reference sequences
        seqs_amy (dict): AMY reference sequences
        seqs_random (dict): Random peptide sequences
    
    Returns:
        tuple: (identity_amp, identity_amy, identity_random)
    """
    print("\n" + "=" * 60)
    print("Calculating Percent Identity")
    print("=" * 60)
    
    identity_amp = []
    identity_amy = []
    identity_random = []
    
    for epoch in selected_epochs:
        if epoch not in collected_seqs:
            print(f"Warning: Epoch {epoch} not found in collected sequences")
            continue
        
        seqs_generated = collected_seqs[epoch]
        print(f"Processing epoch {epoch}...")
        
        # Calculate identity with AMP dataset
        result_amp = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_amp)
        identity_amp_epoch = np.array(result_amp["score"][:, 2], dtype=float)
        identity_amp.append(identity_amp_epoch)
        
        # Calculate identity with AMY dataset
        result_amy = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_amy)
        identity_amy_epoch = np.array(result_amy["score"][:, 2], dtype=float)
        identity_amy.append(identity_amy_epoch)
        
        # Calculate identity with random dataset
        result_random = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_random)
        identity_random_epoch = np.array(result_random["score"][:, 2], dtype=float)
        identity_random.append(identity_random_epoch)
    
    print(f"Identity calculation completed for {len(identity_amp)} epochs")
    return identity_amp, identity_amy, identity_random


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_identity_boxplots(identity_amp, identity_amy, identity_random, selected_epochs, save_path):
    """
    Create box plots showing percent identity distribution across epochs.
    
    Args:
        identity_amp (list): Identity with AMP dataset
        identity_amy (list): Identity with AMY dataset
        identity_random (list): Identity with random dataset
        selected_epochs (list): Selected epochs
        save_path (str): Path to save the plot
    """
    print("\n" + "=" * 60)
    print("Creating Box Plots")
    print("=" * 60)
    
    fig, axs = plt.subplots(3, 1, figsize=(5, 10), dpi=600)
    
    # Increment x-tick labels by +1
    custom_xticks = [tick + 1 for tick in selected_epochs]
    
    # Plot for AMP dataset
    axs[0].boxplot(identity_amp, 
                   positions=range(len(selected_epochs)),
                   patch_artist=True,
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
    axs[0].set_ylim(30, 80)
    axs[0].grid(alpha=0.3, linestyle='--')
    
    # Plot for AMY dataset
    axs[1].boxplot(identity_amy,
                   positions=range(len(selected_epochs)),
                   patch_artist=True,
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
    axs[1].set_ylim(30, 80)
    axs[1].grid(alpha=0.3, linestyle='--')
    
    # Plot for random dataset
    axs[2].boxplot(identity_random,
                   positions=range(len(selected_epochs)),
                   patch_artist=True,
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
    axs[2].set_ylim(30, 80)
    axs[2].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Box plots saved to: {save_path}")


def plot_identity_histograms(identity_amp, identity_amy, identity_random, save_path):
    """
    Create histogram plots for the final epoch.
    
    Args:
        identity_amp (list): Identity with AMP dataset
        identity_amy (list): Identity with AMY dataset
        identity_random (list): Identity with random dataset
        save_path (str): Path to save the plot
    """
    print("\n" + "=" * 60)
    print("Creating Histogram Plots")
    print("=" * 60)
    
    # Extract data for the last epoch
    identity_amp_last = identity_amp[-1]
    identity_amy_last = identity_amy[-1]
    identity_random_last = identity_random[-1]
    
    # Calculate statistics
    stats = {
        'AMP': (np.mean(identity_amp_last), np.std(identity_amp_last)),
        'AMY': (np.mean(identity_amy_last), np.std(identity_amy_last)),
        'Random': (np.mean(identity_random_last), np.std(identity_random_last))
    }
    
    print(f"Statistics for final epoch:")
    print(f"  AMP: Mean={stats['AMP'][0]:.2f}, Std={stats['AMP'][1]:.2f}")
    print(f"  AMY: Mean={stats['AMY'][0]:.2f}, Std={stats['AMY'][1]:.2f}")
    print(f"  Random: Mean={stats['Random'][0]:.2f}, Std={stats['Random'][1]:.2f}")
    
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(4, 10), dpi=600, constrained_layout=True)
    
    # Common histogram configuration
    bins = np.linspace(0, 100, NUM_BINS)
    
    # Plot for AMP dataset
    counts_amp, edges_amp = np.histogram(identity_amp_last, bins=bins)
    axs[0].bar(edges_amp[:-1], counts_amp, width=np.diff(edges_amp),
              color='lightblue', edgecolor='blue')
    axs[0].set_xlabel("Percent Identity (%)", fontsize=10)
    axs[0].set_ylabel("Peptide Number", fontsize=10)
    axs[0].set_title("(d)", loc='left', fontweight='bold', fontsize=12)
    axs[0].annotate(f"Mean: {stats['AMP'][0]:.2f}\nStd: {stats['AMP'][1]:.2f}",
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=8, color='blue',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor='blue'))
    axs[0].grid(alpha=0.3, linestyle='--')
    
    # Plot for AMY dataset
    counts_amy, edges_amy = np.histogram(identity_amy_last, bins=bins)
    axs[1].bar(edges_amy[:-1], counts_amy, width=np.diff(edges_amy),
              color='lightgreen', edgecolor='green')
    axs[1].set_xlabel("Percent Identity (%)", fontsize=10)
    axs[1].set_ylabel("Peptide Number", fontsize=10)
    axs[1].set_title("(e)", loc='left', fontweight='bold', fontsize=12)
    axs[1].annotate(f"Mean: {stats['AMY'][0]:.2f}\nStd: {stats['AMY'][1]:.2f}",
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=8, color='green',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor='green'))
    axs[1].grid(alpha=0.3, linestyle='--')
    
    # Plot for random dataset
    counts_random, edges_random = np.histogram(identity_random_last, bins=bins)
    axs[2].bar(edges_random[:-1], counts_random, width=np.diff(edges_random),
              color='lightcoral', edgecolor='red')
    axs[2].set_xlabel("Percent Identity (%)", fontsize=10)
    axs[2].set_ylabel("Peptide Number", fontsize=10)
    axs[2].set_title("(f)", loc='left', fontweight='bold', fontsize=12)
    axs[2].annotate(f"Mean: {stats['Random'][0]:.2f}\nStd: {stats['Random'][1]:.2f}",
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=8, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor='red'))
    axs[2].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram plots saved to: {save_path}")


# ==============================================================================
# Main Analysis Function
# ==============================================================================

def analyze_percent_identity():
    """
    Main function to run percent identity analysis.
    """
    print("\n" + "=" * 60)
    print("Percent Identity Analysis")
    print("=" * 60 + "\n")
    
    # Load data
    collected_seqs = load_collected_sequences(PATH_MODEL, RUN_NUM)
    seqs_amp, seqs_amy, seqs_random = load_reference_sequences()
    
    # Select epochs
    all_epochs = sorted(collected_seqs.keys())
    selected_epochs = select_epochs_exponentially(all_epochs)
    
    # Calculate identity
    identity_amp, identity_amy, identity_random = calculate_identity_over_epochs(
        collected_seqs, selected_epochs, seqs_amp, seqs_amy, seqs_random
    )
    
    # Create visualizations
    boxplot_path = os.path.join(
        PATH_RESULT,
        "percent_identity_over_epochs_boxplot_three_datasets.png"
    )
    plot_identity_boxplots(
        identity_amp, identity_amy, identity_random,
        selected_epochs, boxplot_path
    )
    
    histogram_path = os.path.join(
        PATH_RESULT,
        "percent_identity_distribution_last_epoch_fixed.png"
    )
    plot_identity_histograms(
        identity_amp, identity_amy, identity_random,
        histogram_path
    )
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    analyze_percent_identity()

