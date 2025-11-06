import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def calculate_aromatic_content(sequence):
    """Calculate the fraction of aromatic amino acids (F, W, Y) in a sequence."""
    aromatic = set("FWY")
    return sum(1 for aa in sequence if aa in aromatic) / len(sequence) if sequence else 0

def calculate_hydrophobic_content(sequence):
    """Calculate the fraction of hydrophobic amino acids in a sequence."""
    hydrophobic = set("AILMFPWV")
    return sum(1 for aa in sequence if aa in hydrophobic) / len(sequence) if sequence else 0

def generate_random_peptides(n_peptides=1000, min_length=15, max_length=25, output_file="random_peptides.fasta"):
    """
    Generate random peptides and save them in FASTA format.
    
    Parameters:
    - n_peptides: Number of peptides to generate
    - min_length: Minimum peptide length
    - max_length: Maximum peptide length  
    - output_file: Output FASTA file name
    """
    # Standard amino acids (exclude unusual ones)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    
    # Generate random lengths
    lengths = np.random.randint(min_length, max_length + 1, n_peptides)
    
    # Generate peptides
    peptides = ["".join(random.choices(aas, k=l)) for l in lengths]
    
    # Remove duplicates (if any) and ensure we have exactly n_peptides
    peptides = list(set(peptides))
    
    # If we lost peptides due to duplicates, generate more
    while len(peptides) < n_peptides:
        additional_needed = n_peptides - len(peptides)
        additional_lengths = np.random.randint(min_length, max_length + 1, additional_needed * 2)  # Generate extra to account for duplicates
        additional_peptides = ["".join(random.choices(aas, k=l)) for l in additional_lengths]
        peptides.extend(additional_peptides)
        peptides = list(set(peptides))  # Remove duplicates again
    
    # Take exactly n_peptides
    peptides = peptides[:n_peptides]
    
    # Save to FASTA format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_file, 'w') as f:
        for i, peptide in enumerate(peptides, 1):
            f.write(f">random_peptide_{i:04d} | length={len(peptide)}\n")
            f.write(f"{peptide}\n")
    
    print(f"Generated {len(peptides)} random peptides and saved to {output_file}")
    
    return peptides

def analyze_peptide_properties(peptides):
    """Analyze and visualize properties of generated peptides."""
    # Calculate properties
    lengths = [len(p) for p in peptides]
    aromatic_contents = [calculate_aromatic_content(p) for p in peptides]
    hydrophobic_contents = [calculate_hydrophobic_content(p) for p in peptides]
    
    # Create summary statistics
    print("\n=== Peptide Property Summary ===")
    print(f"Total peptides: {len(peptides)}")
    print(f"Length - Mean: {np.mean(lengths):.2f}, Std: {np.std(lengths):.2f}, Range: {min(lengths)}-{max(lengths)}")
    print(f"Aromatic content - Mean: {np.mean(aromatic_contents):.3f}, Std: {np.std(aromatic_contents):.3f}")
    print(f"Hydrophobic content - Mean: {np.mean(hydrophobic_contents):.3f}, Std: {np.std(hydrophobic_contents):.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Random Peptide Properties Analysis', fontsize=16, fontweight='bold')
    
    # Length distribution
    axes[0, 0].hist(lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Peptide Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Length Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Aromatic content distribution
    axes[0, 1].hist(aromatic_contents, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Aromatic Content (fraction)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Aromatic Content Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hydrophobic content distribution
    axes[1, 0].hist(hydrophobic_contents, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Hydrophobic Content (fraction)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Hydrophobic Content Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Length vs aromatic content scatter plot
    scatter = axes[1, 1].scatter(lengths, aromatic_contents, alpha=0.6, c=hydrophobic_contents, 
                                cmap='viridis', s=30)
    axes[1, 1].set_xlabel('Peptide Length')
    axes[1, 1].set_ylabel('Aromatic Content')
    axes[1, 1].set_title('Length vs Aromatic Content\n(colored by Hydrophobic Content)')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Hydrophobic Content')
    
    plt.tight_layout()
    plt.savefig('random_peptides_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return lengths, aromatic_contents, hydrophobic_contents

if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    # random.seed(42)
    # np.random.seed(42)
    
    # Generate random peptides and save to FASTA
    peptides = generate_random_peptides(n_peptides=1000, min_length=3, max_length=30, 
                                       output_file="random_peptides_1000_2.fasta")
    
    # Analyze properties
    lengths, aromatic_contents, hydrophobic_contents = analyze_peptide_properties(peptides)
    
    print("\n=== First 10 generated peptides ===")
    for i, peptide in enumerate(peptides[:10], 1):
        print(f"{i:2d}. {peptide} (length: {len(peptide)}, aromatic: {calculate_aromatic_content(peptide):.3f}, hydrophobic: {calculate_hydrophobic_content(peptide):.3f})")

