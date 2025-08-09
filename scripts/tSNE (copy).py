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
import pandas as pd
sys.path.append(os.path.expanduser("~/workspace/amyAMP"))
from scripts import util

def get_encoded_seqs(selected_seqs, table):
    """Helper function - you may need to implement this based on your util module"""
    # This should be implemented based on your existing encoding logic
    pass

def improved_tSNE2D(fastafiles, table, filename_base):
    """
    Improved t-SNE with better parameters and multiple perplexity values
    """
    labels = ["BiGAN-peps", "AMPs", "AMYs", "Random-peps"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Better color palette
    markers = ['o', 's', '^', 'D']
    
    # Collect all data first
    all_data = []
    all_labels = []
    
    for i, fastafile in enumerate(fastafiles):
        print(f"Processing {fastafile}")
        fasta_seqs = util.read_fasta(fastafile)
        
        selected_seqs = {id_: seq for id_, seq in fasta_seqs.items() if len(seq) < 30}
        print(f"Selected sequences: {len(selected_seqs)}")
        
        if len(selected_seqs) == 0:
            continue
            
        encoded = get_encoded_seqs(selected_seqs, table)
        encoded = encoded.reshape(-1, 30, 6)
        encoded_flat = encoded.reshape(-1, 6)
        
        all_data.append(encoded_flat)
        all_labels.extend([labels[i]] * len(encoded_flat))
    
    # Combine all data
    combined_data = np.vstack(all_data)
    
    # Standardize the data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    
    # Try different perplexity values
    perplexities = [30, 50, 100]
    
    fig, axes = plt.subplots(1, len(perplexities), figsize=(15, 5), dpi=300)
    
    for idx, perplexity in enumerate(perplexities):
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                   learning_rate=200, max_iter=1000, n_iter_without_progress=300)
        
        transformed_data = tsne.fit_transform(combined_data_scaled)
        
        ax = axes[idx]
        
        # Plot each group
        start_idx = 0
        for i, data in enumerate(all_data):
            end_idx = start_idx + len(data)
            group_data = transformed_data[start_idx:end_idx]
            
            ax.scatter(group_data[:, 0], group_data[:, 1], 
                      c=colors[i], marker=markers[i], label=labels[i], 
                      alpha=0.7, s=20, edgecolors='black', linewidth=0.5)
            start_idx = end_idx
        
        ax.set_title(f't-SNE (perplexity={perplexity})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_tsne_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return transformed_data

def umap_visualization(fastafiles, table, filename_base):
    """
    UMAP visualization - often better than t-SNE for biological data
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None
    
    labels = ["BiGAN-peps", "AMPs", "AMYs", "Random-peps"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Collect all data
    all_data = []
    all_labels = []
    
    for i, fastafile in enumerate(fastafiles):
        fasta_seqs = util.read_fasta(fastafile)
        selected_seqs = {id_: seq for id_, seq in fasta_seqs.items() if len(seq) < 30}
        
        if len(selected_seqs) == 0:
            continue
            
        encoded = get_encoded_seqs(selected_seqs, table)
        encoded_flat = encoded.reshape(-1, 6)
        
        all_data.append(encoded_flat)
        all_labels.extend([labels[i]] * len(encoded_flat))
    
    combined_data = np.vstack(all_data)
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    
    # UMAP with different parameters
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(combined_data_scaled)
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    start_idx = 0
    for i, data in enumerate(all_data):
        end_idx = start_idx + len(data)
        group_data = embedding[start_idx:end_idx]
        
        plt.scatter(group_data[:, 0], group_data[:, 1], 
                   c=colors[i], label=labels[i], alpha=0.7, s=20)
        start_idx = end_idx
    
    plt.title('UMAP Projection of Peptide Sequences')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{filename_base}_umap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return embedding

def pca_analysis(fastafiles, table, filename_base):
    """
    PCA analysis with variance explanation
    """
    labels = ["BiGAN-peps", "AMPs", "AMYs", "Random-peps"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Collect all data
    all_data = []
    all_labels = []
    
    for i, fastafile in enumerate(fastafiles):
        fasta_seqs = util.read_fasta(fastafile)
        selected_seqs = {id_: seq for id_, seq in fasta_seqs.items() if len(seq) < 30}
        
        if len(selected_seqs) == 0:
            continue
            
        encoded = get_encoded_seqs(selected_seqs, table)
        encoded_flat = encoded.reshape(-1, 6)
        
        all_data.append(encoded_flat)
        all_labels.extend([labels[i]] * len(encoded_flat))
    
    combined_data = np.vstack(all_data)
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    
    # PCA
    pca = PCA()
    pca_result = pca.fit_transform(combined_data_scaled)
    
    # Plot explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Explained variance plot
    ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance Ratio')
    ax1.set_title('PCA Explained Variance')
    ax1.grid(True, alpha=0.3)
    
    # PCA scatter plot
    start_idx = 0
    for i, data in enumerate(all_data):
        end_idx = start_idx + len(data)
        group_data = pca_result[start_idx:end_idx]
        
        ax2.scatter(group_data[:, 0], group_data[:, 1], 
                   c=colors[i], label=labels[i], alpha=0.7, s=20)
        start_idx = end_idx
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.set_title('PCA Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_pca.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca_result, pca.explained_variance_ratio_

def density_plots(fastafiles, table, filename_base):
    """
    Create density plots for each physicochemical property
    """
    labels = ["BiGAN-peps", "AMPs", "AMYs", "Random-peps"]
    property_names = ["H1", "V", "P1", "Pl", "PKa", "NCl"]  # Update based on your properties
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    axes = axes.flatten()
    
    for prop_idx in range(6):
        ax = axes[prop_idx]
        
        for i, fastafile in enumerate(fastafiles):
            fasta_seqs = util.read_fasta(fastafile)
            selected_seqs = {id_: seq for id_, seq in fasta_seqs.items() if len(seq) < 30}
            
            if len(selected_seqs) == 0:
                continue
                
            encoded = get_encoded_seqs(selected_seqs, table)
            encoded_flat = encoded.reshape(-1, 6)
            
            # Plot density for this property
            property_values = encoded_flat[:, prop_idx]
            ax.hist(property_values, bins=30, alpha=0.6, label=labels[i], density=True)
        
        ax.set_title(f'Distribution of {property_names[prop_idx]}')
        ax.set_xlabel(property_names[prop_idx])
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()

def comprehensive_analysis(fastafiles, table, output_dir):
    """
    Run all analysis methods
    """
    print("Running comprehensive peptide sequence analysis...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    filename_base = os.path.join(output_dir, "peptide_analysis")
    
    print("1. Running improved t-SNE analysis...")
    improved_tSNE2D(fastafiles, table, filename_base)
    
    print("2. Running UMAP analysis...")
    umap_visualization(fastafiles, table, filename_base)
    
    print("3. Running PCA analysis...")
    pca_analysis(fastafiles, table, filename_base)
    
    print("4. Creating distribution plots...")
    density_plots(fastafiles, table, filename_base)
    
    print("Analysis complete! Check the output directory for results.")

# ...existing code...

