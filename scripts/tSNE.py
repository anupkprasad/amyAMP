"""
tSNE_separate_AMP_AMY.py
========================
Comprehensive dimensionality reduction and visualization for peptide datasets.

This script performs multiple analyses including:
1. PCA (2D and 3D)
2. t-SNE (2D and 3D)
3. UMAP
4. DBSCAN clustering
5. Violin plots for physicochemical properties
6. Amino acid frequency analysis

Author: [Your Name]
Date: [Date]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn import __version__ as sklearn_version

# Add project root to system path (relative to this file)
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_ROOT not in sys.path:
    sys.path.append(PATH_ROOT)

# Import project modules
from scripts import util, plotStyle
plotStyle.setPlotStyle()

# ==============================================================================
# Configuration
# ==============================================================================

# Visual styling
GROUP_LABELS = ["amyAMP", "trainPep", "AMP(D)", "AMY(D)", "randPep"]
CONTRAST_COLORS = [
    '#2ECC40',  # amyAMP - Green
    '#B10DC9',  # trainPep - Purple
    '#FF6B6B',  # AMP(D) - Light Red
    '#4DA6FF',  # AMY(D) - Light Blue
    '#FFD60A'   # randPep - Yellow
]
MARKERS = ['o'] * len(GROUP_LABELS)  # Same marker for all datasets

# Analysis parameters
BATCH_GENERATE = 1000
SAMPLES_PER_GROUP = 1000
RANDOM_SEED = 42
N_PCA_COMPONENTS = 50
T_SNE_PERPLEXITIES = [30]
DBSCAN_EPS = 3.0
DBSCAN_MIN_SAMPLES = 10

# Plot styling to match percent identity figures
PLOT_FONT_SIZE = 10
PLOT_LEGEND_SIZE = 10
UMAP_FIGSIZE = (4, 4)


# ==============================================================================
# Data Loading and Encoding
# ==============================================================================

def get_encoded_seqs(selected_seqs, table):
    """
    Encode sequences using the PC6 conversion table.
    
    Args:
        selected_seqs (dict): Dictionary of sequences
        table (dict): PC6 conversion table
    
    Returns:
        np.ndarray: Encoded sequences
    """
    try:
        return util.get_encoded_seqs(selected_seqs, table)
    except AttributeError:
        # Fallback manual encoding
        sequences = list(selected_seqs.values())
        max_len = 30
        n_features = 6
        
        encoded_seqs = []
        for seq in sequences:
            seq_padded = seq[:max_len].ljust(max_len, 'X')
            encoded_seq = np.zeros((max_len, n_features))
            for i, aa in enumerate(seq_padded):
                if aa in table:
                    encoded_seq[i] = table[aa]
                else:
                    encoded_seq[i] = np.zeros(n_features)
            encoded_seqs.append(encoded_seq)
        
        return np.array(encoded_seqs)


def _get_dataset_sequences(label):
    """
    Load sequences for a dataset label.

    Args:
        label (str): Dataset label

    Returns:
        dict: Sequence dictionary id->sequence
    """
    if label == "amyAMP":
        path = os.path.join(PATH_ROOT, "results", "sequence", "seqs_generated_postprocessed.fasta")
        return util.read_fasta(path) if os.path.exists(path) else {}

    if label == "trainPep":
        seqs_amp = {}
        seqs_amy = {}
        path_amp = os.path.join(PATH_ROOT, "results", "sequence", "dbaasp_APR_processed_removedJZ.fasta")
        path_amy = os.path.join(PATH_ROOT, "results", "sequence", "amys_uniqueAI4AMP_normalized.fasta")
        if os.path.exists(path_amp):
            seqs_amp = util.read_fasta(path_amp)
        if os.path.exists(path_amy):
            seqs_amy = util.read_fasta(path_amy)
        return {**seqs_amp, **seqs_amy}

    if label == "AMP(D)":
        path = os.path.join(PATH_ROOT, "data_master", "amps", "dbaasp", "dbaasp_non_APR.fasta")
        return util.read_fasta(path) if os.path.exists(path) else {}

    if label == "AMY(D)":
        path = os.path.join(PATH_ROOT, "data_master", "amyloid", "AMY_nonAntibacterial.fasta")
        return util.read_fasta(path) if os.path.exists(path) else {}

    if label == "randPep":
        path = os.path.join(PATH_ROOT, "results", "sequence", "random_peptides_1000.fasta")
        return util.read_fasta(path) if os.path.exists(path) else {}

    return {}


def get_embedded_data(table):
    """
    Load and embed data from configured datasets.
    
    Args:
        table (dict): PC6 conversion table
    
    Returns:
        tuple: (combined_data, group_sizes, group_labels, dataset_sequences)
    """
    all_data = []
    group_sizes = []
    dataset_sequences = {}
    rng = np.random.default_rng(RANDOM_SEED)

    for label in GROUP_LABELS:
        fasta_seqs = _get_dataset_sequences(label)

        if len(fasta_seqs) == 0:
            print(f"Warning: No sequences found for dataset: {label}")
            group_sizes.append(0)
            dataset_sequences[label] = {}
            continue

        selected_seqs = {id_: seq for id_, seq in fasta_seqs.items() if len(seq) < 30}

        if len(selected_seqs) == 0:
            print(f"Warning: No sequences < 30 residues in dataset: {label}")
            group_sizes.append(0)
            continue

        # Balance groups by using the same number of sequences per dataset.
        if len(selected_seqs) > SAMPLES_PER_GROUP:
            selected_ids = rng.choice(list(selected_seqs.keys()), size=SAMPLES_PER_GROUP, replace=False)
            selected_seqs = {seq_id: selected_seqs[seq_id] for seq_id in selected_ids}
        elif len(selected_seqs) < SAMPLES_PER_GROUP:
            print(
                f"Warning: {label} has only {len(selected_seqs)} sequences < 30 residues; "
                f"using all available instead of {SAMPLES_PER_GROUP}."
            )

        # Store exactly the sampled subset used across all analyses.
        dataset_sequences[label] = selected_seqs

        encoded = get_encoded_seqs(selected_seqs, table)
        print(f"Loaded {len(selected_seqs)} sequences from {label}")

        # Handle possible encoding shapes robustly
        arr = np.asarray(encoded)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = np.squeeze(arr, axis=1)
        if arr.ndim == 3:
            encoded_flat = np.mean(arr, axis=1)
        elif arr.ndim == 2:
            encoded_flat = arr
        else:
            print(f"Warning: Unexpected encoded shape {arr.shape} for dataset {label}; skipping")
            group_sizes.append(0)
            continue

        all_data.append(encoded_flat)
        group_sizes.append(len(encoded_flat))

    combined_data = np.vstack(all_data) if all_data else np.empty((0, 6))

    return combined_data, group_sizes, GROUP_LABELS, dataset_sequences


# ==============================================================================
# PCA Analysis
# ==============================================================================

def pca_analysis_3D(embedded_data, group_sizes, group_labels, filename_base, reference_vector=None):
    """
    3D PCA analysis with separate plots for variance, 2D, and 3D projections.
    
    Args:
        embedded_data (np.ndarray): Input data
        group_sizes (list): Sizes of each group
        group_labels (list): Labels for each group
        filename_base (str): Base filename for saving plots
        reference_vector (np.ndarray, optional): Reference vector to plot
    
    Returns:
        tuple: (pca_result, explained_variance_ratio)
    """
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(embedded_data)
    
    # Perform PCA
    n_components = min(N_PCA_COMPONENTS, data_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    explained = pca.explained_variance_ratio_
    
    # Calculate variance percentages
    pc1_var = explained[0] * 100
    pc2_var = explained[1] * 100
    pc3_var = explained[2] * 100 if len(explained) > 2 else 0
    total_var_2d = np.sum(explained[:2]) * 100
    total_var_3d = np.sum(explained[:3]) * 100
    
    # Plot 1: Explained variance (scree plot)
    fig1, ax1 = plt.subplots(figsize=(3.8, 3.8), dpi=600)
    ax1.plot(range(1, min(51, len(explained) + 1)), 
            np.cumsum(explained[:50]), 'o-', color='#E63946', 
            linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance')
    ax1.set_title('PCA Variance Explained')
    ax1.grid(alpha=0.3, linestyle='--')
    
    textstr = (f'PC1: {pc1_var:.1f}%\nPC2: {pc2_var:.1f}%\nPC3: {pc3_var:.1f}%\n'
              f'PC1+PC2: {total_var_2d:.1f}%\nPC1+PC2+PC3: {total_var_3d:.1f}%')
    ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(f"{filename_base}_pca_variance.png"), dpi=600, bbox_inches='tight')
    plt.close()
    
    # Plot 2: 2D PCA (PC1 vs PC2)
    fig2, ax2 = plt.subplots(figsize=(3.8, 3.8), dpi=600)
    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue
        end_idx = start_idx + group_size
        group_data = pca_result[start_idx:end_idx]
        
        ax2.scatter(group_data[:, 0], group_data[:, 1],
                   c=CONTRAST_COLORS[i % len(CONTRAST_COLORS)],
                   marker=MARKERS[i % len(MARKERS)], label=label, 
                   alpha=0.7, s=10, edgecolors='black', linewidth=0.3)
        start_idx = end_idx
    
    if reference_vector is not None:
        ref_pca = pca.transform(scaler.transform(reference_vector.reshape(1, -1)))
        ax2.scatter(ref_pca[0, 0], ref_pca[0, 1], color='black', 
                   s=200, marker='*', label='Reference', edgecolors='white', linewidth=2)
    
    ax2.set_xlabel(f'PC1 ({pc1_var:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pc2_var:.1f}%)')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(f"{filename_base}_pca_2d.png"), dpi=600, bbox_inches='tight')
    plt.close()
    
    # Plot 3: 3D PCA
    fig3 = plt.figure(figsize=(10, 8), dpi=600)
    ax3 = fig3.add_subplot(111, projection='3d')
    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue
        end_idx = start_idx + group_size
        group_data = pca_result[start_idx:end_idx]
        
        ax3.scatter(group_data[:, 0], group_data[:, 1], group_data[:, 2],
                   c=CONTRAST_COLORS[i % len(CONTRAST_COLORS)],
                   marker=MARKERS[i % len(MARKERS)], label=label, 
                   alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        start_idx = end_idx
    
    if reference_vector is not None:
        ref_pca = pca.transform(scaler.transform(reference_vector.reshape(1, -1)))
        ax3.scatter(ref_pca[0, 0], ref_pca[0, 1], ref_pca[0, 2], 
                   color='black', s=200, marker='*', label='Reference', 
                   edgecolors='white', linewidth=2)
    
    ax3.set_xlabel(f'PC1 ({pc1_var:.1f}%)')
    ax3.set_ylabel(f'PC2 ({pc2_var:.1f}%)')
    ax3.set_zlabel(f'PC3 ({pc3_var:.1f}%)')
    ax3.set_title('PCA 3D Projection')
    ax3.legend(loc='upper right')
    ax3.view_init(elev=40, azim=60)
    
    plt.tight_layout()
    plt.savefig(os.path.join(f"{filename_base}_pca_3d.png"), dpi=600, bbox_inches='tight')
    plt.close()
    
    # Print variance information
    print(f"\n{'='*60}")
    print("PCA Variance Analysis:")
    print(f"{'='*60}")
    print(f"PC1 variance: {pc1_var:.2f}%")
    print(f"PC2 variance: {pc2_var:.2f}%")
    print(f"PC3 variance: {pc3_var:.2f}%")
    print(f"Cumulative (PC1+PC2): {total_var_2d:.2f}%")
    print(f"Cumulative (PC1+PC2+PC3): {total_var_3d:.2f}%")
    print(f"{'='*60}\n")
    
    return pca_result, explained


# ==============================================================================
# t-SNE Analysis
# ==============================================================================

def improved_tSNE2D(embedded_data, group_sizes, group_labels, filename_base):
    """
    2D t-SNE visualization with PCA preprocessing.
    
    Args:
        embedded_data (np.ndarray): Input data
        group_sizes (list): Sizes of each group
        group_labels (list): Labels for each group
        filename_base (str): Base filename for saving plots
    
    Returns:
        dict: Mapping of perplexity to t-SNE coordinates
    """
    # Standardize data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(embedded_data)
    
    # PCA preprocessing
    n_features = combined_data_scaled.shape[1]
    n_pca_components = min(N_PCA_COMPONENTS, n_features)
    pca = PCA(n_components=n_pca_components, random_state=42)
    pca_data = pca.fit_transform(combined_data_scaled)
    
    fig, axes = plt.subplots(1, len(T_SNE_PERPLEXITIES),
                            figsize=UMAP_FIGSIZE, dpi=600)
    if len(T_SNE_PERPLEXITIES) == 1:
        axes = [axes]
    
    transformed_dict = {}
    use_n_iter = int(sklearn_version.split(".")[1]) >= 22

    for idx, perplexity in enumerate(T_SNE_PERPLEXITIES):
        # Ensure perplexity is within reasonable range
        actual_perplexity = min(perplexity, len(combined_data_scaled) // 4)
        if actual_perplexity < 5:
            actual_perplexity = 5
            
        tsne = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            **({"n_iter": 1000} if use_n_iter else {"max_iter": 1000}),
            random_state=42
        )
        
        transformed_data = tsne.fit_transform(pca_data)
        transformed_dict[actual_perplexity] = transformed_data
        
        ax = axes[idx]
        start_idx = 0
        for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
            if group_size == 0:
                continue
                
            end_idx = start_idx + group_size
            group_data = transformed_data[start_idx:end_idx]
            
            ax.scatter(
                group_data[:, 0], group_data[:, 1],
                c=CONTRAST_COLORS[i % len(CONTRAST_COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                label=label,
                alpha=0.7,
                s=40,
                edgecolors='black',
                linewidth=0.5
            )
            start_idx = end_idx
        
        ax.set_xlabel('t-SNE 1', fontsize=PLOT_FONT_SIZE)
        ax.set_ylabel('t-SNE 2', fontsize=PLOT_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=PLOT_FONT_SIZE)
        ax.set_title(f't-SNE 2D (perplexity={actual_perplexity})', fontsize=PLOT_FONT_SIZE)
        ax.legend(loc='best', fontsize=PLOT_LEGEND_SIZE)
        ax.grid(alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(f"{filename_base}_tsne_2d.png"), dpi=600, bbox_inches='tight')
    plt.close()
    
    return transformed_dict


# ==============================================================================
# UMAP Visualization
# ==============================================================================

def umap_visualization(embedded_data, group_sizes, group_labels, filename_base):
    """
    UMAP visualization for dimensionality reduction.
    
    Args:
        embedded_data (np.ndarray): Input data
        group_sizes (list): Sizes of each group
        group_labels (list): Labels for each group
        filename_base (str): Base filename for saving plots
    
    Returns:
        np.ndarray: UMAP embedding or None if UMAP not installed
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None
    
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(embedded_data)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(combined_data_scaled)
    
    plt.figure(figsize=UMAP_FIGSIZE, dpi=600)
    
    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue
            
        end_idx = start_idx + group_size
        group_data = embedding[start_idx:end_idx]
        
        plt.scatter(group_data[:, 0], group_data[:, 1],
               c=CONTRAST_COLORS[i % len(CONTRAST_COLORS)],
               marker=MARKERS[i % len(MARKERS)], label=label,
               alpha=0.45, s=6, edgecolors='black', linewidth=0.15)
        start_idx = end_idx
    
    plt.xlabel('UMAP 1', fontsize=PLOT_FONT_SIZE)
    plt.ylabel('UMAP 2', fontsize=PLOT_FONT_SIZE)
    plt.tick_params(axis='both', labelsize=PLOT_FONT_SIZE)
    plt.legend(fontsize=PLOT_LEGEND_SIZE)
    plt.grid(alpha=0.2, linestyle='--')
    plt.savefig(os.path.join(f"{filename_base}_umap.png"), dpi=600, bbox_inches='tight')
    plt.close()
    
    return embedding


def umap_visualization_3d(embedded_data, group_sizes, group_labels, filename_base):
    """
    3D UMAP visualization for better spatial separation inspection.

    Args:
        embedded_data (np.ndarray): Input data
        group_sizes (list): Sizes of each group
        group_labels (list): Labels for each group
        filename_base (str): Base filename for saving plots

    Returns:
        np.ndarray: 3D UMAP embedding or None if UMAP not installed
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None

    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(embedded_data)

    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(combined_data_scaled)

    fig = plt.figure(figsize=(7, 5.5), dpi=600)
    ax = fig.add_subplot(111, projection='3d')

    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue

        end_idx = start_idx + group_size
        group_data = embedding_3d[start_idx:end_idx]

        ax.scatter(
            group_data[:, 0], group_data[:, 1], group_data[:, 2],
            c=CONTRAST_COLORS[i % len(CONTRAST_COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            label=label,
            alpha=0.45,
            s=8,
            edgecolors='black',
            linewidth=0.2
        )
        start_idx = end_idx

    ax.set_xlabel('UMAP 1', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('UMAP 2', fontsize=PLOT_FONT_SIZE)
    ax.set_zlabel('UMAP 3', fontsize=PLOT_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=PLOT_FONT_SIZE)
    ax.legend(loc='best', fontsize=PLOT_LEGEND_SIZE)
    ax.view_init(elev=22, azim=42)

    plt.tight_layout()
    plt.savefig(os.path.join(f"{filename_base}_umap_3d.png"), dpi=600, bbox_inches='tight')
    plt.close()

    return embedding_3d


def umap_density_contours(embedding, group_sizes, group_labels, filename_base):
    """
    Plot UMAP distribution contours to better show spatial density by group.

    Args:
        embedding (np.ndarray): UMAP coordinates with shape (n_samples, 2)
        group_sizes (list): Sizes of each group
        group_labels (list): Labels for each group
        filename_base (str): Base filename for saving plots
    """
    if embedding is None or len(embedding) == 0:
        return

    plt.figure(figsize=UMAP_FIGSIZE, dpi=600)

    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue

        end_idx = start_idx + group_size
        group_data = embedding[start_idx:end_idx]
        color = CONTRAST_COLORS[i % len(CONTRAST_COLORS)]

        # Light scatter plus contour lines to make dense regions visible.
        plt.scatter(group_data[:, 0], group_data[:, 1],
                    c=color, alpha=0.2, s=4, linewidth=0, label=label)

        if group_size >= 20:
            sns.kdeplot(
                x=group_data[:, 0], y=group_data[:, 1],
                levels=4, color=color, linewidths=1.2, fill=False, thresh=0.05
            )

        start_idx = end_idx

    plt.xlabel('UMAP 1', fontsize=PLOT_FONT_SIZE)
    plt.ylabel('UMAP 2', fontsize=PLOT_FONT_SIZE)
    plt.tick_params(axis='both', labelsize=PLOT_FONT_SIZE)
    plt.legend(fontsize=PLOT_LEGEND_SIZE)
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(f"{filename_base}_umap_density.png"), dpi=600, bbox_inches='tight')
    plt.close()


def _build_group_slices(group_sizes, group_labels):
    """
    Build start/end slices for each group in the concatenated embedding.

    Args:
        group_sizes (list): Sizes of each group
        group_labels (list): Labels for each group

    Returns:
        dict: Mapping of label -> (start_idx, end_idx)
    """
    slices = {}
    start_idx = 0
    for group_size, label in zip(group_sizes, group_labels):
        end_idx = start_idx + group_size
        slices[label] = (start_idx, end_idx)
        start_idx = end_idx
    return slices


def umap_case_plot(embedded_data, group_sizes, group_labels, filename_base, case_labels, n_components=2, 
                   fitted_reducer=None, fitted_scaler=None, filename_suffix=""):
    """
    Plot case-specific UMAP distributions for selected groups only.

    Args:
        embedded_data (np.ndarray): Input data
        group_sizes (list): Sizes of each group
        group_labels (list): Labels for each group
        filename_base (str): Base filename for saving plots
        case_labels (list): Labels to include in this case
        n_components (int): UMAP dimensionality (2 or 3)
        fitted_reducer: Pre-fitted UMAP reducer (optional, for consistent reference)
        fitted_scaler: Pre-fitted StandardScaler (optional, for consistent reference)
        filename_suffix (str): Additional suffix for filename (e.g., "_fitted" or "_independent")

    Returns:
        tuple: (embedding, reducer, scaler) or (None, None, None) if UMAP not installed
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None, None, None

    case_labels = [label for label in case_labels if label in group_labels]
    if not case_labels:
        print("Warning: No valid labels found for case plot.")
        return None, None, None

    label_to_idx = {label: idx for idx, label in enumerate(group_labels)}
    slices = _build_group_slices(group_sizes, group_labels)

    case_indices = []
    case_sizes = []
    for label in case_labels:
        start_idx, end_idx = slices.get(label, (0, 0))
        if end_idx > start_idx:
            case_indices.extend(range(start_idx, end_idx))
            case_sizes.append(end_idx - start_idx)

    if not case_indices:
        print("Warning: No samples available for selected case labels.")
        return None, None, None

    # Use pre-fitted scaler/reducer if provided, otherwise fit new ones
    if fitted_scaler is not None and fitted_reducer is not None:
        # Use existing fit to transform the new data
        combined_data_scaled = fitted_scaler.transform(embedded_data[case_indices])
        embedding = fitted_reducer.transform(combined_data_scaled)
        scaler = fitted_scaler
        reducer = fitted_reducer
        print(f"  Using pre-fitted UMAP reducer for {case_labels} (trainPep distribution will match reference)")
    else:
        # Fit new scaler and reducer
        scaler = StandardScaler()
        combined_data_scaled = scaler.fit_transform(embedded_data[case_indices])

        reducer = umap.UMAP(
            n_neighbors=15 if n_components == 2 else 20,
            min_dist=0.1,
            n_components=n_components,
            random_state=42
        )
        embedding = reducer.fit_transform(combined_data_scaled)
        print(f"  Fitted new UMAP reducer for {case_labels}")

    palette = [CONTRAST_COLORS[label_to_idx[label] % len(CONTRAST_COLORS)] for label in case_labels]

    if n_components == 2:
        plt.figure(figsize=UMAP_FIGSIZE, dpi=600)
        start_idx = 0
        for label, color, group_size in zip(case_labels, palette, case_sizes):
            end_idx = start_idx + group_size
            group_data = embedding[start_idx:end_idx]
            plt.scatter(group_data[:, 0], group_data[:, 1], c=color, s=8, alpha=0.5,
                        edgecolors='black', linewidth=0.15, label=label)
            start_idx = end_idx

        plt.xlabel('UMAP 1', fontsize=PLOT_FONT_SIZE)
        plt.ylabel('UMAP 2', fontsize=PLOT_FONT_SIZE)
        plt.tick_params(axis='both', labelsize=PLOT_FONT_SIZE)
        plt.legend(fontsize=PLOT_LEGEND_SIZE)
        plt.grid(alpha=0.2, linestyle='--')
        plt.tight_layout()
        filename = f"{filename_base}_umap_case2d_{'_'.join(case_labels)}{filename_suffix}.png"
        plt.savefig(os.path.join(filename), dpi=600, bbox_inches='tight')
        plt.close()
    else:
        fig = plt.figure(figsize=(6.5, 5.5), dpi=600)
        ax = fig.add_subplot(111, projection='3d')
        start_idx = 0
        for label, color, group_size in zip(case_labels, palette, case_sizes):
            end_idx = start_idx + group_size
            group_data = embedding[start_idx:end_idx]
            ax.scatter(group_data[:, 0], group_data[:, 1], group_data[:, 2], c=color,
                       s=8, alpha=0.45, edgecolors='black', linewidth=0.2, label=label)
            start_idx = end_idx

        ax.set_xlabel('UMAP 1', fontsize=PLOT_FONT_SIZE)
        ax.set_ylabel('UMAP 2', fontsize=PLOT_FONT_SIZE)
        ax.set_zlabel('UMAP 3', fontsize=PLOT_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=PLOT_FONT_SIZE)
        ax.legend(fontsize=PLOT_LEGEND_SIZE)
        ax.view_init(elev=22, azim=42)
        plt.tight_layout()
        filename = f"{filename_base}_umap_case3d_{'_'.join(case_labels)}{filename_suffix}.png"
        plt.savefig(os.path.join(filename), dpi=600, bbox_inches='tight')
        plt.close()

    return embedding, reducer, scaler


def feature_correlation_heatmap(embedded_data, filename_base):
    """
    Plot correlation heatmap of encoded PC6 feature space.

    Args:
        embedded_data (np.ndarray): Encoded feature data with shape (n_samples, 6)
        filename_base (str): Base filename for saving plots
    """
    if embedded_data is None or embedded_data.size == 0:
        return

    feature_names = ["H1", "V", "P1", "Pl", "PKa", "NCI"]
    df_feat = pd.DataFrame(embedded_data, columns=feature_names)
    corr = df_feat.corr()

    plt.figure(figsize=(4.2, 3.7), dpi=600)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7})
    plt.title('PC6 Feature Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(f"{filename_base}_feature_correlation.png"), dpi=600, bbox_inches='tight')
    plt.close()


def case_feature_correlation_heatmap(embedded_data, group_sizes, group_labels, filename_base, case_labels):
    """
    Plot feature correlation heatmaps for selected groups only.

    Args:
        embedded_data (np.ndarray): Encoded feature data
        group_sizes (list): Sizes of each group
        group_labels (list): Labels for each group
        filename_base (str): Base filename for saving plots
        case_labels (list): Labels to include in this case
    """
    case_labels = [label for label in case_labels if label in group_labels]
    if not case_labels:
        return

    slices = _build_group_slices(group_sizes, group_labels)
    feature_names = ["H1", "V", "P1", "Pl", "PKa", "NCI"]
    start_idx = 0
    fig, axes = plt.subplots(1, len(case_labels), figsize=(4.2 * len(case_labels), 3.8), dpi=600)
    if len(case_labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, case_labels):
        start_idx, end_idx = slices[label]
        group_data = embedded_data[start_idx:end_idx]
        if len(group_data) == 0:
            ax.axis('off')
            continue
        corr = pd.DataFrame(group_data, columns=feature_names).corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, cbar=False, ax=ax, annot_kws={"size": 6})
        ax.set_title(label, fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(f"{filename_base}_feature_correlation_{'_'.join(case_labels)}.png"), dpi=600, bbox_inches='tight')
    plt.close()


# ==============================================================================
# Visualization Functions
# ==============================================================================

def violin_plots(embedded_data, group_sizes, group_labels, filename_base):
    """
    Create violin plots for physicochemical properties.
    
    Args:
        embedded_data (np.ndarray): Input data with shape (n_samples, 6)
        group_sizes (list): Sizes of each group
        group_labels (list): Labels for each group
        filename_base (str): Base filename for saving plots
    """
    property_names = ["H1", "V", "P1", "Pl", "PKa", "NCl"]
    
    # Create DataFrame
    data_list = []
    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue
            
        end_idx = start_idx + group_size
        group_data = embedded_data[start_idx:end_idx]
        
        for prop_idx, prop_name in enumerate(property_names):
            for value in group_data[:, prop_idx]:
                data_list.append({
                    'Property': prop_name,
                    'Value': value,
                    'Group': label
                })
        start_idx = end_idx
    
    df = pd.DataFrame(data_list)
    
    # Create violin plots
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5), dpi=600)
    axes = axes.flatten()
    
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    for prop_idx, prop_name in enumerate(property_names):
        ax = axes[prop_idx]
        
        prop_data = df[df['Property'] == prop_name]
        groups_present = [label for label in group_labels if label in prop_data['Group'].values]
        
        parts = ax.violinplot(
            [prop_data[prop_data['Group'] == label]['Value'].values 
             for label in groups_present],
            positions=range(len(groups_present)),
            showmeans=True,
            showmedians=True,
            widths=0.7
        )
        
        # Color the violins
        for pc, color in zip(parts['bodies'], CONTRAST_COLORS):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        # Style other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor('black')
                vp.set_linewidth(0.8)
        
        ax.set_title(f'{subplot_labels[prop_idx]} {prop_name}', loc='left', fontweight='bold')
        ax.set_xticks(range(len(groups_present)))
        ax.set_xticklabels(groups_present, rotation=25)
        
        if prop_idx % 3 == 0:
            ax.set_ylabel('Value')
        
        ax.grid(alpha=0.2, linestyle='--', axis='y')
    
    plt.tight_layout(pad=1.5, h_pad=2, w_pad=2)
    plt.savefig(os.path.join(f"{filename_base}_violin_plots.png"), dpi=600, bbox_inches='tight')
    plt.close()


def amino_acid_frequency_barplot(dataset_sequences, group_labels, filename_base, selected_labels=None):
    """
    Generate grouped bar plot for amino acid frequency.
    
    Args:
        dataset_sequences (dict): Mapping label -> sequence dictionary
        group_labels (list): Labels for each group
        filename_base (str): Base filename for saving plots
        selected_labels (list, optional): Subset of labels to plot. Defaults to all labels.
    """
    amino_acids = sorted("ACDEFGHIKLMNPQRSTVWY")
    plot_labels = [label for label in (selected_labels or group_labels) if label in group_labels]
    amino_acid_counts = {label: {aa: 0 for aa in amino_acids} for label in plot_labels}

    # Count amino acids
    for label in plot_labels:
        fasta_seqs = dataset_sequences.get(label, {})
        for seq in fasta_seqs.values():
            for aa in seq:
                if aa in amino_acid_counts[label]:
                    amino_acid_counts[label][aa] += 1

    # Normalize to fractions
    amino_acid_fractions = {}
    for label, counts in amino_acid_counts.items():
        total_count = sum(counts.values())
        if total_count > 0:
            amino_acid_fractions[label] = {aa: count / total_count 
                                          for aa, count in counts.items()}
        else:
            amino_acid_fractions[label] = {aa: 0 for aa in amino_acids}

    # Plot — use the canonical GROUP_LABELS index so colors match other plots
    dataset_colors = {label: CONTRAST_COLORS[GROUP_LABELS.index(label) % len(CONTRAST_COLORS)]
                     if label in GROUP_LABELS else CONTRAST_COLORS[i % len(CONTRAST_COLORS)]
                     for i, label in enumerate(plot_labels)}
    x = np.arange(len(amino_acids))

    # Keep total cluster width < 1 so there is visible gap between amino acids.
    n_groups = max(len(plot_labels), 1)
    total_cluster_width = 0.72
    width = total_cluster_width / n_groups

    plt.figure(figsize=UMAP_FIGSIZE, dpi=600)
    for i, label in enumerate(plot_labels):
        fractions = [amino_acid_fractions[label][aa] for aa in amino_acids]
        offset = (i - (n_groups - 1) / 2) * width
        plt.bar(x + offset, fractions, width, label=label,
               color=dataset_colors[label], alpha=0.8)

    plt.xticks(x, amino_acids)
    plt.xticks(x, amino_acids, fontsize=PLOT_FONT_SIZE)
    plt.xlabel('Amino Acids', fontsize=PLOT_FONT_SIZE)
    plt.ylabel('Fraction', fontsize=PLOT_FONT_SIZE)
    plt.tick_params(axis='y', labelsize=PLOT_FONT_SIZE)
    plt.legend(fontsize=PLOT_LEGEND_SIZE)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    suffix = "_" + "_".join(plot_labels) if plot_labels else ""
    plt.savefig(os.path.join(f"{filename_base}_amino_acid_frequency{suffix}.png"), 
               dpi=600, bbox_inches='tight')
    plt.close()


# ==============================================================================
# Main Analysis Function
# ==============================================================================

def comprehensive_analysis(table, output_dir):
    """
    Run comprehensive peptide analysis with multiple methods.
    
    Args:
        table (dict): PC6 conversion table
        output_dir (str): Directory to save output files
    
    Returns:
        np.ndarray: Encoded data
    """
    os.makedirs(output_dir, exist_ok=True)
    filename_base = os.path.join(output_dir, "peptide_analysis")
    
    print("\n" + "="*60)
    print("Starting Comprehensive Peptide Analysis")
    print("="*60 + "\n")
    
    embedded_data, group_sizes, group_labels, dataset_sequences = get_embedded_data(table)
    
    print(f"Dataset Summary:")
    for label, size in zip(group_labels, group_sizes):
        print(f"  {label}: {size} sequences")
    print()
    
    print("Running UMAP analysis...")
    umap_embedding = umap_visualization(embedded_data, group_sizes, group_labels, filename_base)

    print("Running UMAP 3D analysis...")
    umap_visualization_3d(embedded_data, group_sizes, group_labels, filename_base)

    print("Running requested UMAP case comparison: trainPep / AMY(D) / AMP(D)...")
    # Fit UMAP on Case 1 (reference) - 2D
    _, reducer_2d, scaler_2d = umap_case_plot(
        embedded_data,
        group_sizes,
        group_labels,
        filename_base,
        case_labels=["trainPep", "AMY(D)", "AMP(D)"],
        n_components=2,
    )
    # Fit UMAP on Case 1 (reference) - 3D
    _, reducer_3d, scaler_3d = umap_case_plot(
        embedded_data,
        group_sizes,
        group_labels,
        filename_base,
        case_labels=["trainPep", "AMY(D)", "AMP(D)"],
        n_components=3,
    )

    print("Running requested UMAP case comparison: trainPep / amyAMP / randPep...")
    print("  (1) Using fitted UMAP from Case 1 (trainPep distribution matches reference)...")
    # Use Case 1 fit for Case 2 to keep trainPep distribution consistent - 2D
    umap_case_plot(
        embedded_data,
        group_sizes,
        group_labels,
        filename_base,
        case_labels=["amyAMP", "trainPep", "randPep"],
        n_components=2,
        fitted_reducer=reducer_2d,
        fitted_scaler=scaler_2d,
        filename_suffix="_fitted",
    )
    # Use Case 1 fit for Case 2 to keep trainPep distribution consistent - 3D
    umap_case_plot(
        embedded_data,
        group_sizes,
        group_labels,
        filename_base,
        case_labels=["amyAMP", "trainPep", "randPep"],
        n_components=3,
        fitted_reducer=reducer_3d,
        fitted_scaler=scaler_3d,
        filename_suffix="_fitted",
    )
    
    print("  (2) Generating independent UMAP (original behavior)...")
    # Also generate independent UMAP for Case 2 (original behavior) - 2D
    umap_case_plot(
        embedded_data,
        group_sizes,
        group_labels,
        filename_base,
        case_labels=["amyAMP", "trainPep", "randPep"],
        n_components=2,
        filename_suffix="_independent",
    )
    # Also generate independent UMAP for Case 2 (original behavior) - 3D
    umap_case_plot(
        embedded_data,
        group_sizes,
        group_labels,
        filename_base,
        case_labels=["amyAMP", "trainPep", "randPep"],
        n_components=3,
        filename_suffix="_independent",
    )

    print("Creating UMAP density contours...")
    umap_density_contours(umap_embedding, group_sizes, group_labels, filename_base)
    
    print("Creating violin plots...")
    violin_plots(embedded_data, group_sizes, group_labels, filename_base)
    
    print("Creating amino acid frequency plots...")
    amino_acid_frequency_barplot(
        dataset_sequences,
        group_labels,
        filename_base,
        selected_labels=["amyAMP", "trainPep", "randPep"],
    )

    print("Creating feature correlation heatmap...")
    feature_correlation_heatmap(embedded_data, filename_base)
    
    print("\n" + "="*60)
    print("Analysis Complete! All plots saved to:", output_dir)
    print("="*60 + "\n")

    return embedded_data


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    path_result = os.path.join(PATH_ROOT, "results")
    path_pc6 = os.path.join(PATH_ROOT, "data_master", "physical_chemical_6.txt")
    
    # Create results directory if it doesn't exist
    os.makedirs(path_result, exist_ok=True)
    
    # Load PC6 conversion table
    table = util.get_conversion_table(path_pc6)
    
    # Run comprehensive analysis
    embedded_data = comprehensive_analysis(table, path_result)

