#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.expanduser("~/workspace/amyAMP/"))
from scripts import util, plotStyle
plotStyle.setPlotStyle()

# Define high-contrast color palette
CONTRAST_COLORS = ['#1f77b4', '#d62728', '#FFD60A']  # Blue, Red, Yellow
MARKERS = ['o', 'o', 'o', 'o']  # Same marker for all datasets

def get_encoded_seqs(selected_seqs, table):
    """
    Encode sequences using the conversion table
    """
    try:
        return util.get_encoded_seqs(selected_seqs, table)
    except AttributeError:
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
    Get embedded data from multiple FASTA files.
    Each dataset in `fastafiles` is treated as a separate group.
    """
    all_data = []
    group_sizes = []
    group_labels = ["amyAMP", "trainPep", "randPep"]  # Labels for the three datasets

    for fastafile, label in zip(fastafiles, group_labels):
        if not os.path.exists(fastafile):
            group_sizes.append(0)
            continue
        
        fasta_seqs = util.read_fasta(fastafile)
        selected_seqs = {id_: seq for id_, seq in fasta_seqs.items() if len(seq) < 30}
        
        if len(selected_seqs) == 0:
            group_sizes.append(0)
            continue
        
        encoded = get_encoded_seqs(selected_seqs, table)
        arr_squeezed = np.squeeze(encoded, axis=1)  # Shape becomes (N, 30, 6)
        encoded_flat = np.mean(arr_squeezed, axis=1)  # Shape: (N, 6)
        all_data.append(encoded_flat)
        group_sizes.append(len(encoded_flat))
    
    combined_data = np.vstack(all_data) if all_data else np.empty((0, 180))
    
    return combined_data, group_sizes, group_labels, all_data




def pca_analysis_3D(embedded_data, group_sizes, group_labels, filename_base, reference_vector=None):
    """
    3D PCA analysis with separate plots for variance, 2D, and 3D projections
    
    Parameters:
    -----------
    embedded_data : array-like
        Input data to analyze
    group_sizes : list
        Sizes of each group
    group_labels : list
        Labels for each group
    filename_base : str
        Base filename for saving plots
    reference_vector : array-like, optional
        Reference vector to plot
    
    Returns:
    --------
    pca_result : array
        PCA-transformed data
    explained : array
        Explained variance ratio
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(embedded_data)
    
    # Perform PCA
    pca = PCA(n_components=min(50, data_scaled.shape[1]))
    pca_result = pca.fit_transform(data_scaled)
    explained = pca.explained_variance_ratio_
    
    # Calculate variance percentages
    pc1_var = explained[0] * 100
    pc2_var = explained[1] * 100
    pc3_var = explained[2] * 100 if len(explained) > 2 else 0
    total_var_2d = np.sum(explained[:2]) * 100
    total_var_3d = np.sum(explained[:3]) * 100
    
    # Plot 1: Explained variance (scree plot) - Separate figure
    fig1 = plt.figure(figsize=(3.5, 3.5), dpi=600)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(range(1, min(51, len(explained) + 1)), 
            np.cumsum(explained[:50]), 'o-', color='#E63946', 
            linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance')
    ax1.set_title('PCA Variance Explained')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    textstr = f'PC1: {pc1_var:.1f}%\nPC2: {pc2_var:.1f}%\nPC3: {pc3_var:.1f}%\n' \
              f'PC1+PC2: {total_var_2d:.1f}%\nPC1+PC2+PC3: {total_var_3d:.1f}%'
    ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_pca_variance.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    # Plot 2: 2D PCA (PC1 vs PC2) - Separate figure
    fig2 = plt.figure(figsize=(3.5, 3.5), dpi=600)
    ax2 = fig2.add_subplot(1, 1, 1)
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
    #ax2.set_title('PCA 2D Projection')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_pca_2d.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    # Plot 3: 3D PCA - Separate figure with single view
    fig3 = plt.figure(figsize=(10, 8), dpi=600)
    ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
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
    # Default view - can be changed interactively
    ax3.view_init(elev=40, azim=60)
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_pca_3d.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    # Print variance information
    print(f"\n{'='*60}")
    print(f"PCA Variance Analysis:")
    print(f"{'='*60}")
    print(f"PC1 variance: {pc1_var:.2f}%")
    print(f"PC2 variance: {pc2_var:.2f}%")
    print(f"PC3 variance: {pc3_var:.2f}%")
    print(f"Cumulative (PC1+PC2): {total_var_2d:.2f}%")
    print(f"Cumulative (PC1+PC2+PC3): {total_var_3d:.2f}%")
    print(f"{'='*60}\n")
    
    return pca_result, explained


def pca_analysis(embedded_data, group_sizes, group_labels, filename_base, reference_vector=None, nonlinear=False):
    """
    Original 2D PCA analysis with contrasting colors and same markers
    """
    assert sum(group_sizes) == embedded_data.shape[0], "Group sizes must sum to total samples!"

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(embedded_data)

    if nonlinear:
        from sklearn.decomposition import KernelPCA
        pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.01)
        pca_result = pca.fit_transform(data_scaled)
        explained = None
    else:
        pca = PCA()
        pca_result = pca.fit_transform(data_scaled)
        explained = pca.explained_variance_ratio_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=600)

    # Explained variance
    if explained is not None:
        ax1.plot(range(1, min(51, len(explained) + 1)), 
                np.cumsum(explained[:50]), 'o-', color='#E63946', 
                linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Cumulative Explained Variance')
        ax1.set_title('PCA Variance Explained')
        ax1.grid(True, alpha=0.3, linestyle='--')
        total_var = np.sum(explained[:2]) * 100
        ax1.text(0.95, 0.05, f'PC1+PC2: {total_var:.1f}%', 
                transform=ax1.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Group scatter
    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue
        end_idx = start_idx + group_size
        group_data = pca_result[start_idx:end_idx]
        
        ax2.scatter(group_data[:, 0], group_data[:, 1],
                   c=CONTRAST_COLORS[i % len(CONTRAST_COLORS)],
                   marker=MARKERS[i % len(MARKERS)], label=label, 
                   alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        start_idx = end_idx

    # Reference marker
    if reference_vector is not None:
        ref_pca = pca.transform(scaler.transform(reference_vector.reshape(1, -1)))
        ax2.scatter(ref_pca[0, 0], ref_pca[0, 1], color='black', 
                   s=200, marker='*', label='Reference', edgecolors='white', linewidth=2)

    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('PCA Projection')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_pca.png", dpi=600, bbox_inches='tight')
    plt.show()

    return pca_result, explained


def improved_tSNE_3D(embedded_data, group_sizes, group_labels, filename_base):
    """
    Perform t-SNE analysis with improved preprocessing for high-dimensional data
    """
    # Combine all data
    combined_data = np.vstack(embedded_data)
    n_samples, n_features = combined_data.shape
    
    print(f"Data shape: {combined_data.shape}")
    
    # Standardize the data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    
    # Define max_components based on sample size
    max_components = min(50, n_samples - 1)  # Add this line
    
    # Apply PCA for dimensionality reduction if needed
    if n_features > 50:
        # Reduce to 50 dimensions first
        n_pca_components = max_components
        pca = PCA(n_components=n_pca_components, random_state=42)
        pca_data = pca.fit_transform(combined_data_scaled)
        print(f"PCA reduced from {n_features} to {n_pca_components} dimensions")
    elif n_features > 3:
        # Use fewer PCA components if features are between 3 and 50
        n_pca_components = min(max_components - 1, n_features)
        pca = PCA(n_components=n_pca_components, random_state=42)
        pca_data = pca.fit_transform(combined_data_scaled)
        print(f"PCA reduced from {n_features} to {n_pca_components} dimensions")
    else:
        pca_data = combined_data_scaled
    
    perplexities = [10, 30, 50]
    fig = plt.figure(figsize=(18, 6), dpi=600)
    
    for idx, perplexity in enumerate(perplexities, 1):
        # Adjust perplexity if it's too large for the dataset
        adjusted_perplexity = min(perplexity, (n_samples - 1) // 3)
        if adjusted_perplexity != perplexity:
            print(f"Adjusted perplexity from {perplexity} to {adjusted_perplexity} due to small sample size")
        
        tsne = TSNE(
            n_components=3,
            perplexity=adjusted_perplexity,
            max_iter=1000,  # Changed from n_iter
            random_state=42,
            n_jobs=-1
        )
        
        tsne_result = tsne.fit_transform(pca_data)
        
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        
        start_idx = 0
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (size, label) in enumerate(zip(group_sizes, group_labels)):
            end_idx = start_idx + size
            ax.scatter(
                tsne_result[start_idx:end_idx, 0],
                tsne_result[start_idx:end_idx, 1],
                tsne_result[start_idx:end_idx, 2],
                c=colors[i],
                label=label,
                alpha=0.6,
                s=50
            )
            start_idx = end_idx
        
        ax.set_title(f'Perplexity = {adjusted_perplexity}', fontsize=12, pad=10)
        ax.set_xlabel('t-SNE 1', fontsize=10)
        ax.set_ylabel('t-SNE 2', fontsize=10)
        ax.set_zlabel('t-SNE 3', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('3D t-SNE Visualization with Different Perplexities', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(filename_base + '_tsne_3D.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"3D t-SNE plot saved to {filename_base}_tsne_3D.png")


def improved_tSNE2D(embedded_data, group_sizes, group_labels, filename_base):
    """
    Improved 2D t-SNE with contrasting colors and same markers.
    Saves a separate plot for perplexity 30.
    """
    # Scale the data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(embedded_data)
    
    # PCA preprocessing
    n_features = combined_data_scaled.shape[1]
    pca = PCA(n_components=min(50, n_features), random_state=42)
    pca_data = pca.fit_transform(combined_data_scaled)
    
    perplexities = [10, 30, 70]
    fig, axes = plt.subplots(1, len(perplexities), figsize=(18, 5), dpi=600)
    if len(perplexities) == 1:
        axes = [axes]
    
    transformed_dict = {}
    
    for idx, perplexity in enumerate(perplexities):
        # Ensure perplexity is within a reasonable range
        actual_perplexity = min(perplexity, len(combined_data_scaled) // 4)
        if actual_perplexity < 5:
            actual_perplexity = 5
            
        tsne = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            max_iter=1000,  # Updated for scikit-learn >=0.22
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
                label="trainPep" if label == "training data" else label,  # Update label
                alpha=0.7,
                s=40,
                edgecolors='black' if MARKERS[i % len(MARKERS)] in ['o','s','^','D'] else 'none',
                linewidth=0.5
            )
            start_idx = end_idx
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(f't-SNE 2D (perplexity={actual_perplexity})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Save a separate plot for perplexity 30
        if perplexity == 30:
            plt.figure(figsize=(8, 6), dpi=600)
            
            # Create a color array for all points based on their group
            color_array = np.zeros(len(transformed_data), dtype=object)
            start_idx = 0
            for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
                end_idx = start_idx + group_size
                color_array[start_idx:end_idx] = CONTRAST_COLORS[i % len(CONTRAST_COLORS)]
                start_idx = end_idx
            
            plt.scatter(
                transformed_data[:, 0], transformed_data[:, 1],
                c=color_array,  # Assign colors to each point
                alpha=0.7, s=40, edgecolors='black', linewidth=0.5
            )
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('t-SNE 2D (Perplexity=30)')
            plt.grid(True, alpha=0.2, linestyle='--')
            plt.savefig(f"{filename_base}_tsne_2d_perplexity_30.png", dpi=600, bbox_inches='tight')
            plt.show()
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_tsne_2d.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    return transformed_dict  # returns a dict: perplexity -> t-SNE coordinates


def umap_visualization(embedded_data, group_sizes, group_labels, filename_base):
    """
    UMAP visualization with contrasting colors and same markers
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
    #plt.title('UMAP Projection')
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.savefig(f"{filename_base}_umap.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    return embedding


def violin_plots(embedded_data, group_sizes, group_labels, filename_base):
    """
    Create violin plots for physicochemical properties with reordered plots.
    """
    # Reorder property names to flip (a) and (d)
    property_names = ["Pl", "V", "P1", "H1", "PKa", "NCl"]  # Swapped "H1" and "Pl"

    # Rearrange the columns of embedded_data to match the new property order
    property_order = [3, 1, 2, 0, 4, 5]  # Indices corresponding to the new order
    avg_features = embedded_data[:, property_order]  # Rearrange columns

    # Create DataFrame for violin plots
    data_list = []
    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue

        end_idx = start_idx + group_size
        group_data = avg_features[start_idx:end_idx]  # Shape: (group_size, 6)

        for prop_idx, prop_name in enumerate(property_names):
            for value in group_data[:, prop_idx]:  # Extract column for this property
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

    # Updated subplot labels to match the new order
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for prop_idx, prop_name in enumerate(property_names):
        ax = axes[prop_idx]

        # Filter data for this property
        prop_data = df[df['Property'] == prop_name]

        # Get groups in order (AmyAmp is already first in group_labels)
        groups_present = [label for label in group_labels if label in prop_data['Group'].values]

        # Create violin plot with contrasting colors
        parts = ax.violinplot(
            [prop_data[prop_data['Group'] == label]['Value'].values
             for label in groups_present],
            positions=range(len(groups_present)),
            showmeans=True,
            showmedians=True,
            widths=0.7
        )

        # Color the violins (AmyAmp gets first color) with thinner edges
        for pc, color in zip(parts['bodies'], CONTRAST_COLORS):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        # Style the other elements with thinner lines
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor('black')
                vp.set_linewidth(0.8)

        # Bold subtitle
        ax.set_title(f'{subplot_labels[prop_idx]} {prop_name}', loc='left', fontweight='bold')

        # Set x-tick labels
        ax.set_xticks(range(len(groups_present)))
        ax.set_xticklabels(groups_present, rotation=0)

        # Only show y-label in leftmost subplots
        if prop_idx % 3 == 0:
            ax.set_ylabel('Value')

        ax.grid(True, alpha=0.2, linestyle='--', axis='y')

    plt.tight_layout(pad=1.5, h_pad=2, w_pad=2)
    plt.savefig(f"{filename_base}_violin_plots.png", dpi=600, bbox_inches='tight')
    plt.show()


def dbscan_clustering(embedded_data, group_sizes, group_labels, filename_base):
    """
    DBSCAN clustering with contrasting colors and same markers
    """
    from sklearn.cluster import DBSCAN
    
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(embedded_data)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=50, random_state=42)
    pca_data = pca.fit_transform(combined_data_scaled)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=3.0, min_samples=10)
    cluster_labels = dbscan.fit_predict(pca_data)
    
    # Apply t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
               learning_rate=200, max_iter=2000)
    tsne_data = tsne.fit_transform(pca_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=600)
    
    # Left plot: Original groups
    ax1.set_title('Original Groups')
    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue
            
        end_idx = start_idx + group_size
        group_data = tsne_data[start_idx:end_idx]
        
        ax1.scatter(group_data[:, 0], group_data[:, 1], 
                   c=CONTRAST_COLORS[i % len(CONTRAST_COLORS)],
                   marker=MARKERS[i % len(MARKERS)], label=label, 
                   alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        start_idx = end_idx
    
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend()
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # Right plot: DBSCAN clusters
    ax2.set_title('DBSCAN Clusters')
    
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Generate distinct colors for clusters
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    
    for k in unique_labels:
        if k == -1:
            col = '#808080'  # Gray for noise
            marker = 'x'
            label = 'Noise'
            alpha = 0.3
            size = 20
        else:
            col = cluster_colors[k % len(cluster_colors)]
            marker = 'o'
            label = f'Cluster {k}'
            alpha = 0.7
            size = 40
            
        class_member_mask = (cluster_labels == k)
        xy = tsne_data[class_member_mask]
        
        ax2.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                   label=label, alpha=alpha, s=size, 
                   edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(ncol=2)
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_dbscan.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    n_noise = list(cluster_labels).count(-1)
    print(f"\n{'='*60}")
    print(f"DBSCAN Clustering Results:")
    print(f"{'='*60}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    if n_clusters > 1:
        print(f"Silhouette score: {silhouette_score(pca_data, cluster_labels):.3f}")
    print(f"{'='*60}\n")
    
    return cluster_labels, tsne_data



def amino_acid_frequency_barplot(fastafiles, group_labels, filename_base):
    """
    Generate a grouped bar plot for amino acid frequency (fractions) across all datasets.
    """
    amino_acids = sorted("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acids in alphabetical order
    amino_acid_counts = {label: {aa: 0 for aa in amino_acids} for label in group_labels}

    # Count amino acids for each dataset
    for fastafile, label in zip(fastafiles, group_labels):
        if not os.path.exists(fastafile):
            continue
        fasta_seqs = util.read_fasta(fastafile)
        for seq in fasta_seqs.values():
            for aa in seq:
                if aa in amino_acid_counts[label]:
                    amino_acid_counts[label][aa] += 1

    # Normalize counts to fractions
    amino_acid_fractions = {}
    for label, counts in amino_acid_counts.items():
        total_count = sum(counts.values())
        if total_count > 0:
            amino_acid_fractions[label] = {aa: count / total_count for aa, count in counts.items()}
        else:
            amino_acid_fractions[label] = {aa: 0 for aa in amino_acids}

    # Prepare data for plotting
    dataset_colors = {
        "amyAMP": "#d62728",         # Red
        "trainPep": "#1f77b4",       # Blue
        "randPep": "#FFD60A"         # Yellow
    }
    x = np.arange(len(amino_acids))  # X-axis positions for amino acids
    width = 0.25  # Width of each bar

    plt.figure(figsize=(3.5, 3), dpi=600)
    for i, label in enumerate(group_labels):
        fractions = [amino_acid_fractions[label][aa] for aa in amino_acids]
        plt.bar(x + i * width, fractions, width, label=label, color=dataset_colors[label], edgecolor='black', alpha=0.8)

    # Customize plot
    plt.xticks(x + width, amino_acids)
    plt.xlabel('Amino Acids')
    plt.ylabel('Fraction')
    #plt.title('Amino Acid Frequency Comparison (Fraction)', fontsize=14)
    plt.legend(fontsize=7)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"{filename_base}_amino_acid_frequency_comparison_fraction.png", dpi=600, bbox_inches='tight')
    plt.show()



def comprehensive_analysis(fastafiles, table, output_dir):
    """
    Run all analysis methods with enhanced visualizations
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
    
    # print("Running 3D PCA analysis...")
    # pca_analysis_3D(embedded_data, group_sizes, group_labels, filename_base)
    
    # print("Running 2D t-SNE analysis...")
    # improved_tSNE2D(embedded_data, group_sizes, group_labels, filename_base)
    
    print("Running UMAP analysis...")
    umap_visualization(embedded_data, group_sizes, group_labels, filename_base)
    
    # print("Running 2D PCA analysis...")
    # pca_analysis(embedded_data, group_sizes, group_labels, filename_base)
    
    print("Creating violin plots...")
    violin_plots(embedded_data, group_sizes, group_labels, filename_base)
    print("Creating amino acid frequency bar plot...")
    amino_acid_frequency_barplot(fastafiles, group_labels, filename_base)
    print("\n" + "="*60)
    print("Analysis Complete! All plots saved to:", output_dir)
    print("="*60 + "\n")

    return encoded_data


if __name__ == "__main__":
    batch_generate = 1000

    path_data = os.path.expanduser("~/workspace/amyAMP/data_master/")
    path_result = os.path.expanduser("~/workspace/amyAMP/results/")
    if not os.path.exists(path_result):
        os.makedirs(path_result)
        
    table = util.get_conversion_table(path_data+"physical_chemical_6.txt")
    l_fasta = [
        path_result+"sequence/"+"seqs_generated_postprocessed.fasta",
        path_result+"sequence/"+"seqs_realAMPs_realAMYs1000.fasta",
        path_result+"sequence/"+"random_peptides_1000.fasta"
    ]

    encoded_data = comprehensive_analysis(l_fasta, table, path_result)

