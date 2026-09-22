"""

Percent identity analysis for generated peptides across training epochs.

This script calculates and visualizes the sequence similarity between
generated peptides and reference datasets (trainPep, AMP(D), AMY(D), randPep)
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
import warnings

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
FASTA_TRAIN_AMP = os.path.join(PATH_RESULT, "sequence", "dbaasp_APR_processed_removedJZ.fasta")
FASTA_TRAIN_AMY = os.path.join(PATH_RESULT, "sequence", "amys_uniqueAI4AMP_normalized.fasta")
FASTA_AMP_DB = os.path.join(PATH_ROOT, "data_master", "amps", "dbaasp", "dbaasp_non_APR.fasta")
FASTA_AMY_DB = os.path.join(PATH_ROOT, "data_master", "amyloid", "AMY_nonAntibacterial.fasta")
FASTA_RANDOM = os.path.join(PATH_RESULT, "sequence", "random_peptides_1000.fasta")

# Training parameters
RUN_NUM = 2  # Number of training runs
NUM_BINS = 41  # Number of bins for histogram (0-100 in 40 bins)
IDENTITY_CACHE_FILE = os.path.join(
    PATH_RESULT,
    "percent_identity_cache_trainpep_ampd_amyd_randpep.npz"
)


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
    Load reference sequences (trainPep, AMP(D), AMY(D), randPep).
    
    Returns:
        tuple: (seqs_train, seqs_amp_db, seqs_amy_db, seqs_random)
    """
    print("\n" + "=" * 60)
    print("Loading Reference Sequences")
    print("=" * 60)
    
    seqs_train_amp = ProcessSeqs(FASTA_TRAIN_AMP).get_seqs()
    seqs_train_amy = ProcessSeqs(FASTA_TRAIN_AMY).get_seqs()
    seqs_train = {**seqs_train_amp, **seqs_train_amy}
    seqs_amp_db = ProcessSeqs(FASTA_AMP_DB).get_seqs()
    seqs_amy_db = ProcessSeqs(FASTA_AMY_DB).get_seqs()
    seqs_random = ProcessSeqs(FASTA_RANDOM).get_seqs()
    
    print(f"trainPep sequences (AMP+AMY): {len(seqs_train)}")
    print(f"AMP(D) sequences: {len(seqs_amp_db)}")
    print(f"AMY(D) sequences: {len(seqs_amy_db)}")
    print(f"Random sequences: {len(seqs_random)}")
    
    return seqs_train, seqs_amp_db, seqs_amy_db, seqs_random


# ==============================================================================
# Identity Calculation
# ==============================================================================

def load_identity_cache(cache_path, selected_epochs):
    """
    Load cached identity data if available and epoch list matches.

    Args:
        cache_path (str): Path to cache file
        selected_epochs (list): Selected epochs for current run

    Returns:
        tuple or None: Cached identity arrays or None if unavailable/mismatched
    """
    if not os.path.exists(cache_path):
        return None

    try:
        cached = np.load(cache_path, allow_pickle=True)
        cached_epochs = cached["selected_epochs"].astype(int).tolist()

        if cached_epochs != selected_epochs:
            print("Warning: Identity cache found but epoch selection differs. Recalculating identity.")
            return None

        warnings.warn(
            f"Using available identity cache: {cache_path}",
            UserWarning
        )

        identity_train = [np.asarray(arr, dtype=float) for arr in cached["identity_train"]]
        identity_amp_db = [np.asarray(arr, dtype=float) for arr in cached["identity_amp_db"]]
        identity_amy_db = [np.asarray(arr, dtype=float) for arr in cached["identity_amy_db"]]
        identity_random = [np.asarray(arr, dtype=float) for arr in cached["identity_random"]]

        return identity_train, identity_amp_db, identity_amy_db, identity_random
    except Exception as e:
        print(f"Warning: Failed to load identity cache ({e}). Recalculating identity.")
        return None


def save_identity_cache(cache_path, selected_epochs, identity_train, identity_amp_db, identity_amy_db, identity_random):
    """
    Save identity arrays to cache for reuse.

    Args:
        cache_path (str): Path to cache file
        selected_epochs (list): Selected epochs
        identity_train (list): trainPep identity arrays
        identity_amp_db (list): AMP(D) identity arrays
        identity_amy_db (list): AMY(D) identity arrays
        identity_random (list): randPep identity arrays
    """
    np.savez_compressed(
        cache_path,
        selected_epochs=np.asarray(selected_epochs, dtype=int),
        identity_train=np.asarray(identity_train, dtype=object),
        identity_amp_db=np.asarray(identity_amp_db, dtype=object),
        identity_amy_db=np.asarray(identity_amy_db, dtype=object),
        identity_random=np.asarray(identity_random, dtype=object),
    )
    print(f"Saved identity cache to: {cache_path}")

def calculate_identity_over_epochs(collected_seqs, selected_epochs, seqs_train, seqs_amp_db, seqs_amy_db, seqs_random):
    """
    Calculate percent identity for selected epochs against reference datasets.
    
    Args:
        collected_seqs (dict): Collected sequences from training
        selected_epochs (list): List of epochs to analyze
        seqs_train (dict): trainPep reference sequences (AMP + AMY)
        seqs_amp_db (dict): AMP(D) reference sequences
        seqs_amy_db (dict): AMY(D) reference sequences
        seqs_random (dict): Random peptide sequences
    
    Returns:
        tuple: (identity_train, identity_amp_db, identity_amy_db, identity_random)
    """
    print("\n" + "=" * 60)
    print("Calculating Percent Identity")
    print("=" * 60)
    
    identity_train = []
    identity_amp_db = []
    identity_amy_db = []
    identity_random = []
    
    for epoch in selected_epochs:
        if epoch not in collected_seqs:
            print(f"Warning: Epoch {epoch} not found in collected sequences")
            continue
        
        seqs_generated = collected_seqs[epoch]
        print(f"Processing epoch {epoch}...")
        
        # Calculate identity with trainPep dataset
        result_train = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_train)
        identity_train_epoch = np.array(result_train["score"][:, 2], dtype=float)
        identity_train.append(identity_train_epoch)

        # Calculate identity with AMP(D) dataset
        result_amp_db = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_amp_db)
        identity_amp_db_epoch = np.array(result_amp_db["score"][:, 2], dtype=float)
        identity_amp_db.append(identity_amp_db_epoch)

        # Calculate identity with AMY(D) dataset
        result_amy_db = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_amy_db)
        identity_amy_db_epoch = np.array(result_amy_db["score"][:, 2], dtype=float)
        identity_amy_db.append(identity_amy_db_epoch)
        
        # Calculate identity with random dataset
        result_random = get_batchAlignment(seqs1=seqs_generated, seqs2=seqs_random)
        identity_random_epoch = np.array(result_random["score"][:, 2], dtype=float)
        identity_random.append(identity_random_epoch)
    
    print(f"Identity calculation completed for {len(identity_train)} epochs")
    return identity_train, identity_amp_db, identity_amy_db, identity_random


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_identity_boxplots(identity_train, identity_amp_db, identity_amy_db, identity_random, selected_epochs, save_path):
    """
    Create box plots showing percent identity distribution across epochs.
    
    Args:
        identity_train (list): Identity with trainPep dataset
        identity_amp_db (list): Identity with AMP(D) dataset
        identity_amy_db (list): Identity with AMY(D) dataset
        identity_random (list): Identity with random dataset
        selected_epochs (list): Selected epochs
        save_path (str): Path to save the plot
    """
    print("\n" + "=" * 60)
    print("Creating Box Plots")
    print("=" * 60)
    
    fig, axs = plt.subplots(4, 1, figsize=(5, 13), dpi=600)
    
    # Increment x-tick labels by +1
    custom_xticks = [tick + 1 for tick in selected_epochs]
    
    # Plot for trainPep dataset
    axs[0].boxplot(identity_train,
                   positions=range(len(selected_epochs)),
                   patch_artist=True,
                   medianprops=dict(color="indigo", linewidth=1.5),
                   flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'},
                   boxprops=dict(facecolor='#E9B3F5', color='#B10DC9', linewidth=1.5),
                   whiskerprops=dict(color='#B10DC9', linewidth=1.5),
                   capprops=dict(color='#B10DC9', linewidth=1.5))
    axs[0].set_xticks(range(len(selected_epochs)))
    axs[0].set_xticklabels(custom_xticks, rotation=25, fontsize=10)
    axs[0].set_xlabel("Epochs", fontsize=10)
    axs[0].set_ylabel("Percent Identity (%)", fontsize=10)
    axs[0].tick_params(axis='both', labelsize=10)
    axs[0].set_title("(a) amyAMP similarity w.r.t. trainPep", loc='left', fontweight='bold', fontsize=12)
    axs[0].set_ylim(30, 80)
    axs[0].grid(alpha=0.3, linestyle='--')

    # Plot for AMP(D) dataset
    axs[1].boxplot(identity_amp_db,
                   positions=range(len(selected_epochs)),
                   patch_artist=True,
                   medianprops=dict(color="#B22222", linewidth=1.5),
                   flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'},
                   boxprops=dict(facecolor='#FFB3B3', color='#FF6B6B', linewidth=1.5),
                   whiskerprops=dict(color='#FF6B6B', linewidth=1.5),
                   capprops=dict(color='#FF6B6B', linewidth=1.5))
    axs[1].set_xticks(range(len(selected_epochs)))
    axs[1].set_xticklabels(custom_xticks, rotation=25, fontsize=10)
    axs[1].set_xlabel("Epochs", fontsize=10)
    axs[1].set_ylabel("Percent Identity (%)", fontsize=10)
    axs[1].tick_params(axis='both', labelsize=10)
    axs[1].set_title("(b) amyAMP similarity w.r.t. AMP(D)", loc='left', fontweight='bold', fontsize=12)
    axs[1].set_ylim(30, 80)
    axs[1].grid(alpha=0.3, linestyle='--')

    # Plot for AMY(D) dataset
    axs[2].boxplot(identity_amy_db,
                   positions=range(len(selected_epochs)),
                   patch_artist=True,
                   medianprops=dict(color="#1E3A8A", linewidth=1.5),
                   flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'},
                   boxprops=dict(facecolor='#B3D9FF', color='#4DA6FF', linewidth=1.5),
                   whiskerprops=dict(color='#4DA6FF', linewidth=1.5),
                   capprops=dict(color='#4DA6FF', linewidth=1.5))
    axs[2].set_xticks(range(len(selected_epochs)))
    axs[2].set_xticklabels(custom_xticks, rotation=25, fontsize=10)
    axs[2].set_xlabel("Epochs", fontsize=10)
    axs[2].set_ylabel("Percent Identity (%)", fontsize=10)
    axs[2].tick_params(axis='both', labelsize=10)
    axs[2].set_title("(c) amyAMP similarity w.r.t. AMY(D)", loc='left', fontweight='bold', fontsize=12)
    axs[2].set_ylim(30, 80)
    axs[2].grid(alpha=0.3, linestyle='--')
    
    # Plot for random dataset
    axs[3].boxplot(identity_random,
                   positions=range(len(selected_epochs)),
                   patch_artist=True,
                   medianprops=dict(color="goldenrod", linewidth=1.5),
                   flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'},
                   boxprops=dict(facecolor='#FFEB99', color='#FFD60A', linewidth=1.5),
                   whiskerprops=dict(color='#FFD60A', linewidth=1.5),
                   capprops=dict(color='#FFD60A', linewidth=1.5))
    axs[3].set_xticks(range(len(selected_epochs)))
    axs[3].set_xticklabels(custom_xticks, rotation=25, fontsize=10)
    axs[3].set_xlabel("Epochs", fontsize=10)
    axs[3].set_ylabel("Percent Identity (%)", fontsize=10)
    axs[3].tick_params(axis='both', labelsize=10)
    axs[3].set_title("(d) amyAMP similarity w.r.t. randPep", loc='left', fontweight='bold', fontsize=12)
    axs[3].set_ylim(30, 80)
    axs[3].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Box plots saved to: {save_path}")


def plot_identity_histograms(identity_train, identity_amp_db, identity_amy_db, identity_random, save_path):
    """
    Create histogram plots for the final epoch.
    
    Args:
        identity_train (list): Identity with trainPep dataset
        identity_amp_db (list): Identity with AMP(D) dataset
        identity_amy_db (list): Identity with AMY(D) dataset
        identity_random (list): Identity with random dataset
        save_path (str): Path to save the plot
    """
    print("\n" + "=" * 60)
    print("Creating Histogram Plots")
    print("=" * 60)
    
    # Extract data for the last epoch
    identity_train_last = identity_train[-1]
    identity_amp_db_last = identity_amp_db[-1]
    identity_amy_db_last = identity_amy_db[-1]
    identity_random_last = identity_random[-1]
    
    # Calculate statistics
    stats = {
        'trainPep': (np.mean(identity_train_last), np.std(identity_train_last)),
        'AMP(D)': (np.mean(identity_amp_db_last), np.std(identity_amp_db_last)),
        'AMY(D)': (np.mean(identity_amy_db_last), np.std(identity_amy_db_last)),
        'Random': (np.mean(identity_random_last), np.std(identity_random_last))
    }
    
    print(f"Statistics for final epoch:")
    print(f"  trainPep: Mean={stats['trainPep'][0]:.2f}, Std={stats['trainPep'][1]:.2f}")
    print(f"  AMP(D): Mean={stats['AMP(D)'][0]:.2f}, Std={stats['AMP(D)'][1]:.2f}")
    print(f"  AMY(D): Mean={stats['AMY(D)'][0]:.2f}, Std={stats['AMY(D)'][1]:.2f}")
    print(f"  Random: Mean={stats['Random'][0]:.2f}, Std={stats['Random'][1]:.2f}")
    
    # Create figure
    fig, axs = plt.subplots(4, 1, figsize=(4, 13), dpi=600, constrained_layout=True)
    
    # Common histogram configuration
    bins = np.linspace(0, 100, NUM_BINS)
    
    # Plot for trainPep dataset
    counts_train, edges_train = np.histogram(identity_train_last, bins=bins)
    axs[0].bar(edges_train[:-1], counts_train, width=np.diff(edges_train),
              color='#E9B3F5', edgecolor='#B10DC9')
    axs[0].set_xlabel("Percent Identity (%)", fontsize=10)
    axs[0].set_ylabel("Peptide Number", fontsize=10)
    axs[0].tick_params(axis='both', labelsize=10)
    axs[0].set_title("(e)", loc='left', fontweight='bold', fontsize=12)
    axs[0].annotate(f"Mean: {stats['trainPep'][0]:.2f}\nStd: {stats['trainPep'][1]:.2f}",
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=8, color='#B10DC9',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor='#B10DC9'))
    axs[0].grid(alpha=0.3, linestyle='--')

    # Plot for AMP(D) dataset
    counts_amp_db, edges_amp_db = np.histogram(identity_amp_db_last, bins=bins)
    axs[1].bar(edges_amp_db[:-1], counts_amp_db, width=np.diff(edges_amp_db),
              color='#FFB3B3', edgecolor='#FF6B6B')
    axs[1].set_xlabel("Percent Identity (%)", fontsize=10)
    axs[1].set_ylabel("Peptide Number", fontsize=10)
    axs[1].tick_params(axis='both', labelsize=10)
    axs[1].set_title("(f)", loc='left', fontweight='bold', fontsize=12)
    axs[1].annotate(f"Mean: {stats['AMP(D)'][0]:.2f}\nStd: {stats['AMP(D)'][1]:.2f}",
               xy=(0.95, 0.95), xycoords='axes fraction',
               ha='right', va='top', fontsize=8, color='#FF6B6B',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                   edgecolor='#FF6B6B'))
    axs[1].grid(alpha=0.3, linestyle='--')

    # Plot for AMY(D) dataset
    counts_amy_db, edges_amy_db = np.histogram(identity_amy_db_last, bins=bins)
    axs[2].bar(edges_amy_db[:-1], counts_amy_db, width=np.diff(edges_amy_db),
              color='#B3D9FF', edgecolor='#4DA6FF')
    axs[2].set_xlabel("Percent Identity (%)", fontsize=10)
    axs[2].set_ylabel("Peptide Number", fontsize=10)
    axs[2].tick_params(axis='both', labelsize=10)
    axs[2].set_title("(g)", loc='left', fontweight='bold', fontsize=12)
    axs[2].annotate(f"Mean: {stats['AMY(D)'][0]:.2f}\nStd: {stats['AMY(D)'][1]:.2f}",
               xy=(0.95, 0.95), xycoords='axes fraction',
               ha='right', va='top', fontsize=8, color='#4DA6FF',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                   edgecolor='#4DA6FF'))
    axs[2].grid(alpha=0.3, linestyle='--')
    
    # Plot for random dataset
    counts_random, edges_random = np.histogram(identity_random_last, bins=bins)
    axs[3].bar(edges_random[:-1], counts_random, width=np.diff(edges_random),
              color='#FFEB99', edgecolor='#FFD60A')
    axs[3].set_xlabel("Percent Identity (%)", fontsize=10)
    axs[3].set_ylabel("Peptide Number", fontsize=10)
    axs[3].tick_params(axis='both', labelsize=10)
    axs[3].set_title("(h)", loc='left', fontweight='bold', fontsize=12)
    axs[3].annotate(f"Mean: {stats['Random'][0]:.2f}\nStd: {stats['Random'][1]:.2f}",
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=8, color='#FFD60A',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor='#FFD60A'))
    axs[3].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram plots saved to: {save_path}")


# ==============================================================================
# Main Analysis Function
# ==============================================================================

def analyze_percent_identity(force_recalculate=False):
    """
    Main function to run percent identity analysis.
    """
    print("\n" + "=" * 60)
    print("Percent Identity Analysis")
    print("=" * 60 + "\n")
    
    # Load data
    collected_seqs = load_collected_sequences(PATH_MODEL, RUN_NUM)
    seqs_train, seqs_amp_db, seqs_amy_db, seqs_random = load_reference_sequences()
    
    # Select epochs
    all_epochs = sorted(collected_seqs.keys())
    selected_epochs = select_epochs_exponentially(all_epochs)

    # Load cached identity data when available, otherwise calculate and cache
    cached_identity = None if force_recalculate else load_identity_cache(IDENTITY_CACHE_FILE, selected_epochs)
    if cached_identity is not None:
        identity_train, identity_amp_db, identity_amy_db, identity_random = cached_identity
    else:
        identity_train, identity_amp_db, identity_amy_db, identity_random = calculate_identity_over_epochs(
            collected_seqs, selected_epochs, seqs_train, seqs_amp_db, seqs_amy_db, seqs_random
        )
        save_identity_cache(
            IDENTITY_CACHE_FILE,
            selected_epochs,
            identity_train,
            identity_amp_db,
            identity_amy_db,
            identity_random,
        )
    
    # Create visualizations
    boxplot_path = os.path.join(
        PATH_RESULT,
        "percent_identity_over_epochs_boxplot_trainpep_ampd_amyd_randpep.png"
    )
    plot_identity_boxplots(
        identity_train, identity_amp_db, identity_amy_db, identity_random,
        selected_epochs, boxplot_path
    )
    
    histogram_path = os.path.join(
        PATH_RESULT,
        "percent_identity_distribution_last_epoch_trainpep_ampd_amyd_randpep.png"
    )
    plot_identity_histograms(
        identity_train, identity_amp_db, identity_amy_db, identity_random,
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

