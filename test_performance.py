#!/usr/bin/env python3
"""
Performance comparison for different visualization methods
"""
import time
import os
import sys
sys.path.append(os.path.expanduser("~/workspace/amyAMP"))

from scripts.tSNE import (
    fast_tsne_analysis, 
    quick_pca_visualization, 
    hierarchical_clustering_visualization,
    improved_tSNE2D
)
from scripts import util

def time_function(func, *args, **kwargs):
    """Time a function execution"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def main():
    # Setup paths
    path_data = os.path.expanduser("~/workspace/amyAMP/data_master/")
    path_result = os.path.expanduser("~/workspace/amyAMP/results/")
    os.makedirs(path_result, exist_ok=True)
    
    # Check if files exist
    test_files = [
        path_data + "random_pep_uni.fasta",
        path_data + "uperin.fasta"
    ]
    
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_files:
        print("No test files found. Creating sample data...")
        # Create minimal test data
        sample_fasta = path_result + "sample_test.fasta"
        with open(sample_fasta, 'w') as f:
            f.write(">seq1\nACDEFGHIKLMNPQRSTVWY\n")
            f.write(">seq2\nFGHIKLMNPQRSTVWYACD\n")
            f.write(">seq3\nKLMNPQRSTVWYACDEFGH\n")
            f.write(">seq4\nSTVWYACDEFGHIKLMNPQR\n")
        existing_files = [sample_fasta] * 4  # Use same file 4 times for test
    
    # Load conversion table
    table_file = path_data + "physical_chemical_6.txt"
    if os.path.exists(table_file):
        table = util.get_conversion_table(table_file)
    else:
        print("Warning: Conversion table not found. Creating dummy table...")
        # Create a simple dummy table for testing
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        table = {aa: [i/20, (i*2)%7/6, (i*3)%5/4, (i*4)%3/2, (i*5)%11/10, (i*6)%13/12] 
                for i, aa in enumerate(amino_acids)}
    
    filename_base = path_result + "performance_test"
    
    print("=== PERFORMANCE COMPARISON ===\n")
    
    # Test 1: Quick PCA (should be fastest)
    print("1. Testing Quick PCA...")
    try:
        _, pca_time = time_function(quick_pca_visualization, existing_files, table, filename_base, 100)
        print(f"   Quick PCA time: {pca_time:.2f} seconds\n")
    except Exception as e:
        print(f"   Quick PCA failed: {e}\n")
    
    # Test 2: Fast t-SNE
    print("2. Testing Fast t-SNE...")
    try:
        _, fast_tsne_time = time_function(fast_tsne_analysis, existing_files, table, filename_base, 100)
        print(f"   Fast t-SNE time: {fast_tsne_time:.2f} seconds\n")
    except Exception as e:
        print(f"   Fast t-SNE failed: {e}\n")
    
    # Test 3: Hierarchical clustering
    print("3. Testing Hierarchical Clustering...")
    try:
        _, cluster_time = time_function(hierarchical_clustering_visualization, existing_files, table, filename_base, 50)
        print(f"   Hierarchical clustering time: {cluster_time:.2f} seconds\n")
    except Exception as e:
        print(f"   Hierarchical clustering failed: {e}\n")
    
    # Test 4: Original t-SNE (for comparison, with small dataset)
    print("4. Testing Original t-SNE (small sample)...")
    try:
        _, orig_tsne_time = time_function(improved_tSNE2D, existing_files[:2], table, filename_base)
        print(f"   Original t-SNE time: {orig_tsne_time:.2f} seconds\n")
    except Exception as e:
        print(f"   Original t-SNE failed: {e}\n")
    
    print("=== PERFORMANCE TIPS ===")
    print("1. Use Quick PCA for initial exploration (fastest)")
    print("2. Use Fast t-SNE with sampling for detailed analysis")
    print("3. Reduce max_samples parameter for faster computation")
    print("4. Use PCA preprocessing before t-SNE")
    print("5. Consider UMAP as alternative to t-SNE")

if __name__ == "__main__":
    main()
