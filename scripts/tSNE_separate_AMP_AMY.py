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
CONTRAST_COLORS = ['#2ECC40', '#FF4136', '#0074D9', '#FFD60A']  # Green (amyAMP), Red (AMP), Blue (AMY), Yellow (RandPep)
MARKERS = ['o', 'o', 'o', 'o']  # Same marker for all datasets
GROUP_LABELS = ["amyAMP", "AMP", "AMY", "RandPep"]

# Analysis parameters
BATCH_GENERATE = 1000
N_PCA_COMPONENTS = 50
T_SNE_PERPLEXITIES = [30]
DBSCAN_EPS = 3.0
DBSCAN_MIN_SAMPLES = 10


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


def get_embedded_data(fastafiles, table):
    """
    Load and embed data from multiple FASTA files.
    
    Args:
        fastafiles (list): List of FASTA file paths
        table (dict): PC6 conversion table
    
    Returns:
        tuple: (combined_data, group_sizes, group_labels, encoded_data)
    """
    all_data = []
    group_sizes = []
    
    # Reordered indices: AmyAmp (2), AMP (0), AMY (1), Random (3)
    reordered_indices = [2, 0, 1, 3]

    for idx in reordered_indices:
        if idx >= len(fastafiles):
            group_sizes.append(0)
            continue
            
        fastafile = fastafiles[idx]
        if not os.path.exists(fastafile):
            print(f"Warning: File not found: {fastafile}")
            group_sizes.append(0)
            continue
            
        fasta_seqs = util.read_fasta(fastafile)
        selected_seqs = {id_: seq for id_, seq in fasta_seqs.items() if len(seq) < 30}
        
        if len(selected_seqs) == 0:
            print(f"Warning: No sequences < 30 residues in {os.path.basename(fastafile)}")
            group_sizes.append(0)
            continue
        
        encoded = get_encoded_seqs(selected_seqs, table)
        print(f"Loaded {len(selected_seqs)} sequences from {os.path.basename(fastafile)}")
        
        # Reshape: (n_samples, 1, 30, 6) -> (n_samples, 30, 6)
        arr_squeezed = np.squeeze(encoded, axis=1)
        # Average along sequence length: (n_samples, 30, 6) -> (n_samples, 6)
        encoded_flat = np.mean(arr_squeezed, axis=1)
        
        all_data.append(encoded_flat)
        group_sizes.append(len(encoded_flat))
    
    combined_data = np.vstack(all_data) if all_data else np.empty((0, 6))
    
    return combined_data, group_sizes, GROUP_LABELS, encoded


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
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5), dpi=600)
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
    fig2, ax2 = plt.subplots(figsize=(3.5, 3.5), dpi=600)
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
                            figsize=(3.5, 3.5), dpi=600)
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
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(f't-SNE 2D (perplexity={actual_perplexity})')
        ax.legend(loc='best')
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
    
    plt.figure(figsize=(3.5, 3.5), dpi=600)
    
    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue
            
        end_idx = start_idx + group_size
        group_data = embedding[start_idx:end_idx]
        
        plt.scatter(group_data[:, 0], group_data[:, 1], 
                   c=CONTRAST_COLORS[i % len(CONTRAST_COLORS)],
                   marker=MARKERS[i % len(MARKERS)], label=label, 
                   alpha=0.7, s=10, edgecolors='black', linewidth=0.25)
        start_idx = end_idx
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(fontsize=7)
    plt.grid(alpha=0.2, linestyle='--')
    plt.savefig(os.path.join(f"{filename_base}_umap.png"), dpi=600, bbox_inches='tight')
    plt.close()
    
    return embedding


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


def amino_acid_frequency_barplot(fastafiles, group_labels, filename_base):
    """
    Generate grouped bar plot for amino acid frequency.
    
    Args:
        fastafiles (list): List of FASTA file paths
        group_labels (list): Labels for each group
        filename_base (str): Base filename for saving plots
    """
    amino_acids = sorted("ACDEFGHIKLMNPQRSTVWY")
    amino_acid_counts = {label: {aa: 0 for aa in amino_acids} for label in group_labels}

    # Count amino acids
    for fastafile, label in zip(fastafiles, group_labels):
        if not os.path.exists(fastafile):
            continue
        fasta_seqs = util.read_fasta(fastafile)
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

    # Plot
    dataset_colors = {label: CONTRAST_COLORS[i % len(CONTRAST_COLORS)] 
                     for i, label in enumerate(group_labels)}
    x = np.arange(len(amino_acids))
    width = 0.2

    plt.figure(figsize=(3.5, 3.5), dpi=600)
    for i, label in enumerate(group_labels):
        fractions = [amino_acid_fractions[label][aa] for aa in amino_acids]
        plt.bar(x + i * width, fractions, width, label=label, 
               color=dataset_colors[label], alpha=0.8)

    plt.xticks(x + width, amino_acids)
    plt.xlabel('Amino Acids')
    plt.ylabel('Fraction')
    plt.legend(fontsize=7)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(os.path.join(f"{filename_base}_amino_acid_frequency.png"), 
               dpi=600, bbox_inches='tight')
    plt.close()


# ==============================================================================
# Main Analysis Function
# ==============================================================================

def comprehensive_analysis(fastafiles, table, output_dir):
    """
    Run comprehensive peptide analysis with multiple methods.
    
    Args:
        fastafiles (list): List of FASTA file paths
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
    
    embedded_data, group_sizes, group_labels, encoded_data = get_embedded_data(fastafiles, table)
    
    print(f"Dataset Summary:")
    for label, size in zip(group_labels, group_sizes):
        print(f"  {label}: {size} sequences")
    print()
    
    print("Running UMAP analysis...")
    umap_visualization(embedded_data, group_sizes, group_labels, filename_base)
    
    print("Creating violin plots...")
    violin_plots(embedded_data, group_sizes, group_labels, filename_base)
    
    print("Creating amino acid frequency plots...")
    amino_acid_frequency_barplot(fastafiles, group_labels, filename_base)
    
    print("\n" + "="*60)
    print("Analysis Complete! All plots saved to:", output_dir)
    print("="*60 + "\n")

    return encoded_data


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Define paths (relative to project root)
    path_data = os.path.join(PATH_ROOT, "data_master")
    path_result = os.path.join(PATH_ROOT, "results")
    path_pc6 = os.path.join(PATH_ROOT, "data_master", "physical_chemical_6.txt")
    
    # Create results directory if it doesn't exist
    os.makedirs(path_result, exist_ok=True)
    
    # Load PC6 conversion table
    table = util.get_conversion_table(path_pc6)
    
    # Define FASTA files
    l_fasta = [
        os.path.join(path_result, "sequence", f"seqs_realAMPs{BATCH_GENERATE}.fasta"),
        os.path.join(path_result, "sequence", f"seqs_realAMYs{BATCH_GENERATE}.fasta"),
        os.path.join(path_result, "sequence", f"seqs_generated{BATCH_GENERATE}.fasta"),
        os.path.join(path_data, "random_peptides_1000.fasta")
    ]
    
    # Run comprehensive analysis
    encoded_data = comprehensive_analysis(l_fasta, table, path_result)

