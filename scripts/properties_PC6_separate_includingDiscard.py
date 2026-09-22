"""
properties_PC6_separate_includingDiscard.py
============================================
Analysis and visualization of PC6 (six physicochemical properties) for peptide datasets.

PC6 properties (doi: 10.1128/mSystems.00299-21):
1. H1 - Hydrophobicity
2. V - Volume of side chain
3. P1 - Polarity
4. Pl - pH at isoelectric point
5. pKa - Dissociation constant for -COOH group
6. NCI - Net charge index of side chain

Note: J = amidated, Z = non-amidated
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from Bio import SeqIO

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

# Paths (relative to project root)
PATH_RESULT = os.path.join(PATH_ROOT, "results")
PATH_SEQUENCE = os.path.join(PATH_RESULT, "sequence")
PATH_PC6 = os.path.abspath(os.path.join(PATH_ROOT, "data_master", "physical_chemical_6.txt"))

# Dataset configuration
FASTA_FILES = [
    "seqs_generated_postprocessed.fasta",  # amyAMP
    None,                                   # trainPep - combined AMP+AMY (special case)
    "dbaasp_non_APR.fasta",                # AMP(D)
    "AMY_nonAntibacterial.fasta",          # AMY(D)
    "random_peptides_1000.fasta"           # randPep
]
DATASET_LABELS = ["amyAMP", "trainPep", "AMP(D)", "AMY(D)", "randPep"]

# Color scheme based on functional categories:
# amyAMP (green), trainPep (purple), AMP(D) (light red), AMY(D) (light blue), randPep (yellow)
DATASET_COLORS = [
    '#2ECC40',  # amyAMP - Green (dual functional)
    '#B10DC9',  # trainPep - Purple (training set: AMP + AMY combined)
    '#FF6B6B',  # AMP(D) - Light Red (antimicrobial database)
    '#4DA6FF',  # AMY(D) - Light Blue (amyloid database)
    '#FFD60A'   # randPep - Yellow (random/control)
]

# PC6 property names (based on physical_chemical_6.txt file format)
PC6_PROPERTIES = [
    'H1',      # Hydrophobicity
    'V',       # Volume of side chain
    'P1',      # Polarity
    'Pl',      # pH at isoelectric point
    'pKa',     # Dissociation constant for -COOH group
    'NCI'      # Net charge index of side chain
]

PC6_TITLES = [
    'Hydrophobicity (H1)',
    'Side Chain Volume (V)',
    'Polarity (P1)',
    'Isoelectric Point (Pl)',
    'pKa (-COOH)',
    'Net Charge Index (NCI)'
]

# ==============================================================================
# PC6 Scale Loading
# ==============================================================================

def load_pc6_scales():
    """
    Load PC6 physicochemical scales from physical_chemical_6.txt.
    
    File format:
    AA H1 V P1 Pl pKa NCI
    
    Returns:
        dict: Dictionary of PC6 scales, each mapping amino acids to values
    """
    print("=" * 80)
    print("Loading PC6 Scales")
    print("=" * 80)
    print(f"Reading from: {PATH_PC6}")
    
    pc6_scales = {prop: {} for prop in PC6_PROPERTIES}
    
    with open(PATH_PC6, 'r') as f:
        lines = f.readlines()
    
    # Parse the file - format: AA H1 V P1 Pl pKa NCI
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) < 7:  # AA + 6 properties
            continue
        
        aa = parts[0]
        try:
            pc6_scales['H1'][aa] = float(parts[1])
            pc6_scales['V'][aa] = float(parts[2])
            pc6_scales['P1'][aa] = float(parts[3])
            pc6_scales['Pl'][aa] = float(parts[4])
            pc6_scales['pKa'][aa] = float(parts[5])
            pc6_scales['NCI'][aa] = float(parts[6])
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse line: {line}")
            continue
    
    print(f"✓ Loaded PC6 scales for {len(pc6_scales['H1'])} amino acids")
    
    # Show sample values
    print("\nSample PC6 values:")
    for aa in ['A', 'K', 'F']:
        if aa in pc6_scales['H1']:
            print(f"  {aa}: H1={pc6_scales['H1'][aa]:.2f}, V={pc6_scales['V'][aa]:.1f}, "
                  f"P1={pc6_scales['P1'][aa]:.1f}, Pl={pc6_scales['Pl'][aa]:.2f}, "
                  f"pKa={pc6_scales['pKa'][aa]:.2f}, NCI={pc6_scales['NCI'][aa]:.6f}")
    
    return pc6_scales


# ==============================================================================
# PC6 Property Calculation
# ==============================================================================

def calculate_pc6_properties(seq, pc6_scales):
    """
    Calculate PC6 properties for a peptide sequence.
    
    Args:
        seq (str): Amino acid sequence
        pc6_scales (dict): PC6 scale dictionaries
    
    Returns:
        dict: Dictionary of PC6 property values (averaged over sequence)
    """
    seq_str = str(seq).upper()
    length = len(seq_str)
    
    if length == 0:
        return None
    
    properties = {}
    
    # Calculate average for each PC6 property
    for prop_name in PC6_PROPERTIES:
        scale = pc6_scales[prop_name]
        values = []
        
        for aa in seq_str:
            if aa in scale:
                values.append(scale[aa])
            else:
                # Handle unknown amino acids (use neutral value or skip)
                print(f"Warning: Unknown amino acid '{aa}' in sequence")
                values.append(0.0)
        
        properties[prop_name] = np.mean(values) if values else 0.0
    
    properties['length'] = length
    
    return properties


# ==============================================================================
# Data Loading and Processing
# ==============================================================================

def load_and_analyze_sequences(pc6_scales):
    """
    Load sequences from FASTA files and calculate PC6 properties.
    Includes trainPep as combined AMP + AMY training set.
    
    Args:
        pc6_scales (dict): PC6 scale dictionaries
        
    Returns:
        pd.DataFrame: DataFrame containing all sequences and their PC6 properties
    """
    print("\n" + "=" * 80)
    print("Loading and Analyzing Sequences")
    print("=" * 80)
    
    all_data = []
    
    for fasta_file, label in zip(FASTA_FILES, DATASET_LABELS):
        # Handle special case: trainPep is combination of AMP and AMY
        if label == "trainPep":
            print(f"Processing {label} (combination of AMP + AMY sequences used for model training)...")

            # Get AMP sequences from training set
            amp_file_path = os.path.join(PATH_SEQUENCE, "dbaasp_APR_processed_removedJZ.fasta")
            if os.path.exists(amp_file_path):
                amp_records = list(SeqIO.parse(amp_file_path, "fasta"))
                for record in amp_records:
                    properties = calculate_pc6_properties(record.seq, pc6_scales)
                    if properties:
                        properties['dataset'] = label
                        properties['sequence_id'] = record.id
                        all_data.append(properties)

            # Get AMY sequences from training set
            amy_file_path = os.path.join(PATH_SEQUENCE, "amys_uniqueAI4AMP_normalized.fasta")
            if os.path.exists(amy_file_path):
                amy_records = list(SeqIO.parse(amy_file_path, "fasta"))
                for record in amy_records:
                    properties = calculate_pc6_properties(record.seq, pc6_scales)
                    if properties:
                        properties['dataset'] = label
                        properties['sequence_id'] = record.id
                        all_data.append(properties)
            continue

        # Handle datasets in different locations
        if label == "AMP(D)":
            file_path = os.path.join(PATH_ROOT, "data_master", "amps", "dbaasp", fasta_file)
        elif label == "AMY(D)":
            file_path = os.path.join(PATH_ROOT, "data_master", "amyloid", fasta_file)
        else:
            file_path = os.path.join(PATH_SEQUENCE, fasta_file)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Processing {label}...")
        seq_records = list(SeqIO.parse(file_path, "fasta"))
        
        for record in seq_records:
            properties = calculate_pc6_properties(record.seq, pc6_scales)
            if properties:
                properties['dataset'] = label
                properties['sequence_id'] = record.id
                all_data.append(properties)
    
    df = pd.DataFrame(all_data)
    print(f"\nTotal sequences analyzed: {len(df)}")
    print(f"Datasets: {df['dataset'].value_counts().to_dict()}")
    
    return df


def save_summary_statistics(df, output_path):
    """
    Calculate and save summary statistics for PC6 properties.
    
    Args:
        df (pd.DataFrame): DataFrame with PC6 properties
        output_path (str): Path to save summary statistics
    """
    print("\n" + "=" * 80)
    print("Summary Statistics (PC6)")
    print("=" * 80)
    
    summary_stats = df.groupby('dataset')[PC6_PROPERTIES].agg(['mean', 'std', 'min', 'max'])
    print(summary_stats.round(4))
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_pc6_distributions(df, save_path):
    """
    Create 2x3 violin plots for PC6 properties across 5 datasets.
    
    Args:
        df (pd.DataFrame): DataFrame with PC6 properties
        save_path (str): Path to save the plot
    """
    print("\n" + "=" * 80)
    print("Creating PC6 Distribution Plots")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=600)
    axes = axes.flatten()
    
    for idx, (prop, title) in enumerate(zip(PC6_PROPERTIES, PC6_TITLES)):
        ax = axes[idx]
        
        # Prepare data
        dataset_data = [df[df['dataset'] == label][prop].values for label in DATASET_LABELS]
        
        # Create violin plot
        parts = ax.violinplot(dataset_data, positions=range(1, len(DATASET_LABELS) + 1),
                              showmeans=True, showmedians=False, widths=0.7)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(DATASET_COLORS[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Style mean lines
        parts['cmeans'].set_edgecolor('black')
        parts['cmeans'].set_linewidth(2)
        
        # Add statistical annotations
        for i, label in enumerate(DATASET_LABELS):
            data = df[df['dataset'] == label][prop].values
            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data)
                
                # Position text box at top of violin
                y_pos = np.max(data) if len(data) > 0 else 0
                
                ax.text(i + 1, y_pos * 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                       ha='center', va='top', fontsize=6, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                               edgecolor=DATASET_COLORS[i], linewidth=1.5, alpha=0.95))
        
        # Formatting
        subplot_letter = chr(ord('a') + idx)
        ax.set_title(f'({subplot_letter}) {title}', fontweight='bold', loc='left', fontsize=10)
        ax.set_xticks(range(1, len(DATASET_LABELS) + 1))
        ax.set_xticklabels(DATASET_LABELS, fontsize=8, rotation=20, ha='right')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_ylabel('PC6 Value', fontsize=8)
        ax.set_xlabel('')
        
        # Add horizontal line at zero if data crosses zero
        y_min, y_max = ax.get_ylim()
        if y_min < 0 < y_max:
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"PC6 distribution plots saved to: {save_path}")


# ==============================================================================
# Main Analysis Function
# ==============================================================================

def analyze_pc6_properties():
    """
    Main function to run PC6 properties analysis.
    """
    print("\n" + "=" * 80)
    print("PC6 Properties Analysis")
    print("=" * 80 + "\n")
    
    # Load PC6 scales
    pc6_scales = load_pc6_scales()
    
    # Load and analyze sequences
    df = load_and_analyze_sequences(pc6_scales)
    
    # Save summary statistics
    summary_path = os.path.join(PATH_RESULT, "pc6_properties_analysis_incDiscard.csv")
    save_summary_statistics(df, summary_path)
    
    # Create visualization
    plot_path = os.path.join(PATH_RESULT, "pc6_properties_distributions_incDiscard.png")
    plot_pc6_distributions(df, plot_path)
    
    print("\n" + "=" * 80)
    print("PC6 Analysis Complete!")
    print("=" * 80)


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    analyze_pc6_properties()