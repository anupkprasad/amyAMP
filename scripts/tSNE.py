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
sys.path.append(os.path.expanduser("~/workspace/amyAMP"))
from scripts import util, plotStyle
plotStyle.setPlotStyle()

def get_encoded_seqs(selected_seqs, table):
    """
    Encode sequences using the conversion table
    This is a placeholder - update based on your actual encoding logic
    """
    # If you have an existing implementation in util, use it:
    try:
        return util.get_encoded_seqs(selected_seqs, table)
    except AttributeError:
        # Fallback implementation
        sequences = list(selected_seqs.values())
        max_len = 30
        n_features = 6
        
        encoded_seqs = []
        for seq in sequences:
            # Pad or truncate to max_len
            seq_padded = seq[:max_len].ljust(max_len, 'X')
            
            # Simple encoding - replace with your actual encoding logic
            encoded_seq = np.zeros((max_len, n_features))
            for i, aa in enumerate(seq_padded):
                if aa in table:
                    encoded_seq[i] = table[aa]
                else:
                    encoded_seq[i] = np.zeros(n_features)  # Unknown amino acid
            
            encoded_seqs.append(encoded_seq)
        group_labels
        return np.array(encoded_seqs)


def get_embedded_data(fastafiles, table):
    """
    Get embedded data from multiple FASTA files
    Returns: (combined_data, group_sizes, group_labels)
    """
    all_data = []
    group_sizes = []
    group_labels = ["AMP", "AMY", "AmyAmp", "Random"]

    for i, fastafile in enumerate(fastafiles):
        if not os.path.exists(fastafile):
            group_sizes.append(0)
            continue
            
        fasta_seqs = util.read_fasta(fastafile)
        selected_seqs = {id_: seq for id_, seq in fasta_seqs.items() if len(seq) < 30}
        
        if len(selected_seqs) == 0:
            group_sizes.append(0)
            continue
        
        encoded = get_encoded_seqs(selected_seqs, table)
        encoded_flat = encoded.reshape(-1, 30 * 6)
        
        encoded_flat = encoded_flat
        all_data.append(encoded_flat)
        group_sizes.append(len(encoded_flat))
    
    combined_data = np.vstack(all_data) if all_data else np.empty((0, 180))
    
    # # ---- PCA to 100 dimensions ----
    # if combined_data.shape[0] > 0:
    #     pca = PCA(n_components=100, random_state=42)
    #     combined_data_100D = pca.fit_transform(combined_data)
    # else:
    #     combined_data_100D = combined_data

    return combined_data, group_sizes, group_labels


def improved_tSNE2D(embedded_data, group_sizes, group_labels, filename_base):
    """
    Improved t-SNE with better parameters and multiple perplexity values
    """
    colors = ['blue', 'orange', 'red', 'cyan']
    markers = ['o', 's', '^', 'D']
    
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(embedded_data)
    
    perplexities = [10, 30, 70]
    fig, axes = plt.subplots(1, len(perplexities), figsize=(10.5, 3.5), dpi=600)
    if len(perplexities) == 1:
        axes = [axes]
    
    for idx, perplexity in enumerate(perplexities):
        actual_perplexity = min(perplexity, len(combined_data_scaled) // 4)
        if actual_perplexity < 5:
            actual_perplexity = 5
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=actual_perplexity, 
                   learning_rate=200, max_iter=1000, n_iter_without_progress=300)
        
        transformed_data = tsne.fit_transform(combined_data_scaled)
        ax = axes[idx]
        
        start_idx = 0
        for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
            if group_size == 0:
                continue
                
            end_idx = start_idx + group_size
            group_data = transformed_data[start_idx:end_idx]
            
            ax.scatter(group_data[:, 0], group_data[:, 1], 
                      facecolors='none', edgecolors=colors[i % len(colors)], 
                      marker=markers[i % len(markers)], label=label, alpha=0.5, 
                      s=5, linewidth=0.8)
            start_idx = end_idx
        
        #ax.set_title(f't-SNE (perplexity={actual_perplexity})')
        ax.legend(fontsize=7)
        # ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_tsne_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return transformed_data

def umap_visualization(embedded_data, group_sizes, group_labels, filename_base):
    """
    UMAP visualization - often better than t-SNE for biological data
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None
    
    colors = ['blue', 'orange', 'red', 'cyan']
    
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
                   facecolors='none', edgecolors=colors[i % len(colors)], 
                   label=label, alpha=0.5, s=5, linewidth=0.8)
        start_idx = end_idx
    
    #plt.title('UMAP Projection of Peptide Sequences')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(fontsize=7)
    # plt.grid(True, alpha=0.3)
    plt.savefig(f"{filename_base}_umap.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    return embedding

def pca_analysis(embedded_data, group_sizes, group_labels, filename_base):
    """
    PCA analysis with variance explanation
    """
    colors = ['blue', 'orange', 'red', 'cyan']
    
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(embedded_data)
    
    pca = PCA()
    pca_result = pca.fit_transform(combined_data_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=600)
    
    ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance Ratio')
    #ax1.set_title('PCA Explained Variance')
    # ax1.grid(True, alpha=0.3)
    
    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue
            
        end_idx = start_idx + group_size
        group_data = pca_result[start_idx:end_idx]
        
        ax2.scatter(group_data[:, 0], group_data[:, 1], 
                   facecolors='none', edgecolors=colors[i % len(colors)], 
                   label=label, alpha=0.5, s=5, linewidth=0.8)
        start_idx = end_idx
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    #ax2.set_title('PCA Projection')
    ax2.legend(fontsize=7)
    # ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_pca.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    return pca_result, pca.explained_variance_ratio_

def dbscan_clustering(embedded_data, group_sizes, group_labels, filename_base):
    """
    DBSCAN clustering visualization
    """
    from sklearn.cluster import DBSCAN
    
    colors = ['blue', 'orange', 'red', 'cyan']
    
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(embedded_data)
    
    # Apply PCA for dimensionality reduction before DBSCAN
    pca = PCA(n_components=50, random_state=42)
    pca_data = pca.fit_transform(combined_data_scaled)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(pca_data)
    
    # Apply t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
               learning_rate=200, max_iter=1000)
    tsne_data = tsne.fit_transform(pca_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=600)
    
    # Left plot: Original groups
    ax1.set_title('Original Groups', fontweight='bold', fontsize=12)
    start_idx = 0
    for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
        if group_size == 0:
            continue
            
        end_idx = start_idx + group_size
        group_data = tsne_data[start_idx:end_idx]
        
        ax1.scatter(group_data[:, 0], group_data[:, 1], 
                   facecolors='none', edgecolors=colors[i % len(colors)], 
                   label=label, alpha=0.7, s=8, linewidth=0.8)
        start_idx = end_idx
    
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(fontsize=8)
    
    # Right plot: DBSCAN clusters
    ax2.set_title('DBSCAN Clusters', fontweight='bold', fontsize=12)
    
    unique_labels = set(cluster_labels)
    cluster_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, cluster_colors):
        if k == -1:
            # Black used for noise
            col = 'black'
            marker = 'x'
            label = 'Noise'
            alpha = 0.3
        else:
            marker = 'o'
            label = f'Cluster {k}'
            alpha = 0.7
            
        class_member_mask = (cluster_labels == k)
        xy = tsne_data[class_member_mask]
        
        ax2.scatter(xy[:, 0], xy[:, 1], facecolors='none', edgecolors=col,
                   marker=marker, label=label, alpha=alpha, s=8, linewidth=0.8)
    
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_dbscan_clustering.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    # Print clustering statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"DBSCAN Clustering Results:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"Silhouette score: {silhouette_score(pca_data, cluster_labels) if n_clusters > 1 else 'N/A (need >1 cluster)'}")
    
    return cluster_labels, tsne_data

def density_plots(embedded_data, group_sizes, group_labels, filename_base):
    """
    Create density plots for each physicochemical property
    """
    property_names = ["H1", "V", "P1", "Pl", "PKa", "NCl"]
    colors = ['blue', 'orange', 'red', 'cyan']
    
    # Reshape data to get individual amino acid features
    reshaped_data = embedded_data.reshape(-1, 30 , 6)
    avg_features = np.mean(reshaped_data, axis=1)  # Average over sequence length
    
    fig, axes = plt.subplots(2, 3, figsize=(7, 5), dpi=600)
    axes = axes.flatten()
    
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    for prop_idx in range(6):
        ax = axes[prop_idx]
        
        start_idx = 0
        for i, (group_size, label) in enumerate(zip(group_sizes, group_labels)):
            if group_size == 0:
                continue
                
            end_idx = start_idx + group_size
            group_data = avg_features[start_idx:end_idx]
            
            property_values = group_data[:, prop_idx]
            ax.hist(property_values, bins=30, alpha=0.6, 
                   label=label, density=True, color=colors[i % len(colors)])
            start_idx = end_idx
        
        ax.set_title(f'{subplot_labels[prop_idx]} Distribution of {property_names[prop_idx]}', 
                    loc='left', fontweight='bold', fontsize=10)
        ax.set_xlabel(property_names[prop_idx])
        
        # Only show y-label in subplots (a) and (d)
        if prop_idx == 0 or prop_idx == 3:
            ax.set_ylabel('Density')
        
        # Only show legend in the first subplot (a) with two columns
        if prop_idx == 0:
            ax.legend(fontsize=7)
        
        # ax.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(f"{filename_base}_distributions.png", dpi=600, bbox_inches='tight')
    plt.show()

def comprehensive_analysis(fastafiles, table, output_dir):
    """
    Run all analysis methods
    """
    os.makedirs(output_dir, exist_ok=True)
    filename_base = os.path.join(output_dir, "peptide_analysis")
    
    embedded_data, group_sizes, group_labels = get_embedded_data(fastafiles, table)
    
    improved_tSNE2D(embedded_data, group_sizes, group_labels, filename_base)
    umap_visualization(embedded_data, group_sizes, group_labels, filename_base)
    pca_analysis(embedded_data, group_sizes, group_labels, filename_base)
    dbscan_clustering(embedded_data, group_sizes, group_labels, filename_base)
    density_plots(embedded_data, group_sizes, group_labels, filename_base)



#### Enhanced visualization analysis
if __name__ == "__main__":
    run_num = 10
    batch_generate = 1000

    path_data = os.path.expanduser("~/workspace/amyAMP/data_master/")
    path_result = os.path.expanduser("~/workspace/amyAMP/results/")
    if not os.path.exists(path_result):
        os.makedirs(path_result)
        
    table = util.get_conversion_table(path_data+"physical_chemical_6.txt")
    l_fasta = [path_result+"sequence/"+"seqs_realAMPs"+str(batch_generate)+".fasta",
               path_result+"sequence/"+"seqs_realAMYs"+str(batch_generate)+".fasta",
               path_result+ "sequence/"+"seqs_generated"+str(batch_generate)+".fasta",
               path_data+ "random_pep_uni.fasta"]

    comprehensive_analysis(l_fasta, table, path_result)

