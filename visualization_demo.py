#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Peptide Visualization Demo

This script demonstrates better visualization techniques for separating
overlapping peptide groups in embedding space.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def create_sample_data():
    """
    Create sample peptide data for demonstration
    """
    np.random.seed(42)
    
    # Simulate different peptide groups with some overlap
    group1 = np.random.multivariate_normal([2, 3, 1, 4, 2, 1], np.eye(6) * 0.5, 200)  # BiGAN-peps
    group2 = np.random.multivariate_normal([1, 2, 3, 2, 3, 4], np.eye(6) * 0.3, 200)  # AMPs  
    group3 = np.random.multivariate_normal([3, 1, 2, 1, 4, 3], np.eye(6) * 0.4, 200)  # AMYs
    group4 = np.random.multivariate_normal([2, 2, 2, 2, 2, 2], np.eye(6) * 0.8, 200)  # Random-peps
    
    data = np.vstack([group1, group2, group3, group4])
    labels = ['BiGAN-peps'] * 200 + ['AMPs'] * 200 + ['AMYs'] * 200 + ['Random-peps'] * 200
    
    return data, labels

def improved_tsne_comparison(data, labels, output_dir):
    """
    Compare t-SNE with different parameters
    """
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    unique_labels = ['BiGAN-peps', 'AMPs', 'AMYs', 'Random-peps']
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Different t-SNE parameters to try
    params = [
        {'perplexity': 30, 'learning_rate': 200, 'n_iter': 1000},
        {'perplexity': 50, 'learning_rate': 300, 'n_iter': 1500},
        {'perplexity': 100, 'learning_rate': 400, 'n_iter': 2000}
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    for idx, param in enumerate(params):
        tsne = TSNE(n_components=2, random_state=42, **param)
        embedding = tsne.fit_transform(data_scaled)
        
        ax = axes[idx]
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=colors[i], label=label, alpha=0.7, s=20,
                      edgecolors='black', linewidth=0.1)
        
        ax.set_title(f't-SNE (perp={param["perplexity"]}, lr={param["learning_rate"]})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    
    plt.suptitle('t-SNE Parameter Comparison for Peptide Separation', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tsne_parameter_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def pca_vs_tsne_comparison(data, labels, output_dir):
    """
    Compare PCA vs t-SNE
    """
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    unique_labels = ['BiGAN-peps', 'AMPs', 'AMYs', 'Random-peps']
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    
    ax = axes[0]
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                  c=colors[i], label=label, alpha=0.7, s=20,
                  edgecolors='black', linewidth=0.1)
    
    ax.set_title(f'PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=300, n_iter=1500)
    tsne_result = tsne.fit_transform(data_scaled)
    
    ax = axes[1]
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                  c=colors[i], label=label, alpha=0.7, s=20,
                  edgecolors='black', linewidth=0.1)
    
    ax.set_title('t-SNE (Optimized Parameters)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('PCA vs t-SNE for Peptide Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_vs_tsne.png", dpi=300, bbox_inches='tight')
    plt.show()

def density_based_visualization(data, labels, output_dir):
    """
    Create density plots and contour plots
    """
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    unique_labels = ['BiGAN-peps', 'AMPs', 'AMYs', 'Random-peps']
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=300, n_iter=1500)
    embedding = tsne.fit_transform(data_scaled)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Regular scatter plot
    ax = axes[0, 0]
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                  c=colors[i], label=label, alpha=0.7, s=20)
    ax.set_title('Standard Scatter Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Density plot with contours
    ax = axes[0, 1]
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        group_data = embedding[mask]
        
        # Create density plot
        ax.scatter(group_data[:, 0], group_data[:, 1], 
                  c=colors[i], label=label, alpha=0.4, s=15)
        
        # Add contour lines
        try:
            sns.kdeplot(x=group_data[:, 0], y=group_data[:, 1], 
                       ax=ax, color=colors[i], alpha=0.8, levels=3)
        except:
            pass  # Skip if not enough data points
    
    ax.set_title('Density Plot with Contours')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hexbin plot
    ax = axes[1, 0]
    # Use all data for hexbin background
    hb = ax.hexbin(embedding[:, 0], embedding[:, 1], gridsize=30, alpha=0.3, cmap='Greys')
    
    # Overlay scatter points
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                  c=colors[i], label=label, alpha=0.8, s=15)
    
    ax.set_title('Hexbin Density + Scatter')
    ax.legend()
    
    # Separate subplots for each group
    ax = axes[1, 1]
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        group_data = embedding[mask]
        
        # Large points with edge
        ax.scatter(group_data[:, 0], group_data[:, 1], 
                  c=colors[i], label=label, alpha=0.8, s=30,
                  edgecolors='white', linewidth=1)
    
    ax.set_title('Enhanced Scatter (Larger Points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Different Visualization Techniques for Peptide Separation', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/density_visualizations.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create output directory
    output_dir = os.path.expanduser("~/workspace/amyAMP/visualization_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating sample peptide data...")
    data, labels = create_sample_data()
    
    print("Running visualization comparisons...")
    
    print("1. t-SNE parameter comparison...")
    improved_tsne_comparison(data, labels, output_dir)
    
    print("2. PCA vs t-SNE comparison...")
    pca_vs_tsne_comparison(data, labels, output_dir)
    
    print("3. Density-based visualizations...")
    density_based_visualization(data, labels, output_dir)
    
    print(f"\nVisualization demo complete! Check {output_dir} for results.")
    print("\nRecommendations for your peptide data:")
    print("1. Try different t-SNE perplexity values (30, 50, 100)")
    print("2. Increase learning rate (200-400) and iterations (1000-2000)")
    print("3. Use standardized data")
    print("4. Consider UMAP as an alternative to t-SNE")
    print("5. Add density contours to show group boundaries")
    print("6. Use better color palettes and point styling")
