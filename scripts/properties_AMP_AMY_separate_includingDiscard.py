"""
properties_AMP_AMY_seperate.py
==============================
Comprehensive physicochemical and biological properties analysis for peptide datasets.

This script analyzes and visualizes key properties relevant to antimicrobial
and amyloidogenic peptides, including:
1. Hydrophobic moment and GRAVY scores
2. Beta-sheet propensity and instability index
3. Charge distributions and isoelectric points
4. Amino acid composition and entropy
5. Correlation matrices between properties

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

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

# Property acronym mapping for visualizations
PROPERTY_ACRONYMS = {
    'length': 'LEN',
    'charge': 'CHG',
    'net_charge': 'NET',
    'charge_density': 'CDD',
    'hydrophobic_fraction': 'HYD',
    'aromatic_fraction': 'ARO',
    'positive_fraction': 'POS',
    'negative_fraction': 'NEG',
    'polar_fraction': 'POL',
    'gravy': 'GRV',
    'aromaticity': 'ART',
    'instability_index': 'INS',
    'isoelectric_point': 'pI',
    'entropy': 'ENT',
    'charge_hydrophobic_ratio': 'CHR',
    'aromatic_hydrophobic_ratio': 'AHR',
    'beta_propensity': 'BET',
    'amp_favorable_fraction': 'AMP',
    'hydrophobic_moment': 'HMO',
    'amphipathicity': 'APH'
}

# ==============================================================================
# Property Calculation Functions
# ==============================================================================

def calculate_hydrophobic_moment(seq):
    """
    Calculate hydrophobic moment using Kyte-Doolittle hydropathy scale.
    
    Args:
        seq (str): Amino acid sequence
    
    Returns:
        float: Maximum hydrophobic moment across sliding windows
    """
    kd_scale = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    if len(seq) < 3:
        return 0.0
    
    max_moment = 0.0
    window_size = min(len(seq), 11)
    
    for i in range(len(seq) - window_size + 1):
        window = seq[i:i + window_size]
        sum_cos = 0.0
        sum_sin = 0.0
        
        for j, aa in enumerate(window):
            if aa in kd_scale:
                angle = (j * 2 * np.pi) / window_size
                hydrophobicity = kd_scale[aa]
                sum_cos += hydrophobicity * np.cos(angle)
                sum_sin += hydrophobicity * np.sin(angle)
        
        moment = np.sqrt(sum_cos**2 + sum_sin**2) / window_size
        max_moment = max(max_moment, moment)
    
    return max_moment


def calculate_amyloid_properties(seq):
    """
    Calculate comprehensive amyloid-relevant properties for a sequence.
    
    Args:
        seq (str): Amino acid sequence
    
    Returns:
        dict: Dictionary of calculated properties or None if error
    """
    try:
        seq_str = str(seq).upper()
        p = ProteinAnalysis(seq_str)
        
        # Define amino acid groups
        hydrophobic_residues = set("AILFWVY")
        aromatic_residues = set("FYWH")
        charged_positive = set("KR")
        charged_negative = set("DE")
        polar_residues = set("NQSTC")
        beta_sheet_residues = set("VIFYL")
        amp_favorable_residues = set("KRFWC")
        
        length = len(seq_str)
        
        # Basic physicochemical properties
        charge = p.charge_at_pH(7.0)
        gravy = p.gravy()
        aromaticity = p.aromaticity()
        instability = p.instability_index()
        isoelectric_point = p.isoelectric_point()
        
        # Amino acid composition fractions
        hydrophobic_fraction = sum([seq_str.count(r) for r in hydrophobic_residues]) / length
        aromatic_fraction = sum([seq_str.count(r) for r in aromatic_residues]) / length
        positive_fraction = sum([seq_str.count(r) for r in charged_positive]) / length
        negative_fraction = sum([seq_str.count(r) for r in charged_negative]) / length
        polar_fraction = sum([seq_str.count(r) for r in polar_residues]) / length
        
        # Derived properties
        net_charge = (positive_fraction - negative_fraction) * length
        charge_density = abs(charge) / length
        
        # Shannon entropy
        aa_counts = Counter(seq_str)
        entropy = -sum([(count/length) * np.log2(count/length) for count in aa_counts.values()])
        
        # Ratios and propensities
        charge_hydrophobic_ratio = abs(charge) / (hydrophobic_fraction + 1e-6)
        aromatic_hydrophobic_ratio = aromatic_fraction / (hydrophobic_fraction + 1e-6)
        beta_propensity = sum([seq_str.count(r) for r in beta_sheet_residues]) / length
        amp_favorable_fraction = sum([seq_str.count(r) for r in amp_favorable_residues]) / length
        
        # Advanced properties
        hydrophobic_moment = calculate_hydrophobic_moment(seq_str)
        amphipathicity = (positive_fraction + negative_fraction) * hydrophobic_fraction
        
        return {
            "length": length,
            "charge": charge,
            "net_charge": net_charge,
            "charge_density": charge_density,
            "hydrophobic_fraction": hydrophobic_fraction,
            "aromatic_fraction": aromatic_fraction,
            "positive_fraction": positive_fraction,
            "negative_fraction": negative_fraction,
            "polar_fraction": polar_fraction,
            "gravy": gravy,
            "aromaticity": aromaticity,
            "instability_index": instability,
            "isoelectric_point": isoelectric_point,
            "entropy": entropy,
            "charge_hydrophobic_ratio": charge_hydrophobic_ratio,
            "aromatic_hydrophobic_ratio": aromatic_hydrophobic_ratio,
            "beta_propensity": beta_propensity,
            "amp_favorable_fraction": amp_favorable_fraction,
            "hydrophobic_moment": hydrophobic_moment,
            "amphipathicity": amphipathicity
        }
    except Exception as e:
        print(f"Error processing sequence: {seq_str[:20]}... - {e}")
        return None


# ==============================================================================
# Data Loading and Processing
# ==============================================================================

def load_and_analyze_sequences():
    """
    Load sequences from FASTA files and calculate properties.
    Includes trainPep as combined AMP + AMY training set.
    
    Returns:
        pd.DataFrame: DataFrame containing all sequences and their properties
    """
    print("=" * 80)
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
                    properties = calculate_amyloid_properties(record.seq)
                    if properties:
                        properties['dataset'] = label
                        properties['sequence_id'] = record.id
                        all_data.append(properties)
            
            # Get AMY sequences from training set
            amy_file_path = os.path.join(PATH_SEQUENCE, "amys_uniqueAI4AMP_normalized.fasta")
            if os.path.exists(amy_file_path):
                amy_records = list(SeqIO.parse(amy_file_path, "fasta"))
                for record in amy_records:
                    properties = calculate_amyloid_properties(record.seq)
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
            properties = calculate_amyloid_properties(record.seq)
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
    Calculate and save summary statistics.
    
    Args:
        df (pd.DataFrame): DataFrame with properties
        output_path (str): Path to save summary statistics
    """
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    key_properties = [
        'length', 'charge', 'isoelectric_point', 'hydrophobic_moment',
        'instability_index', 'gravy', 'entropy', 'beta_propensity',
        'amp_favorable_fraction', 'amphipathicity'
    ]
    
    summary_stats = df.groupby('dataset')[key_properties].agg(['mean', 'std', 'min', 'max'])
    print(summary_stats.round(3))
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_property_distributions(df, save_path):
    """
    Create 3x3 violin plots for key properties with 5 datasets.
    
    Args:
        df (pd.DataFrame): DataFrame with properties
        save_path (str): Path to save the plot
    """
    print("\n" + "=" * 80)
    print("Creating Property Distribution Plots")
    print("=" * 80)
    
    viz_properties = [
        ('net_charge', 'Net Charge Distribution'),
        ('amp_favorable_fraction', 'AMP-Favorable Residues'),
        ('isoelectric_point', 'Isoelectric Point'),
        ('beta_propensity', 'Beta-sheet Propensity'),
        ('hydrophobic_moment', 'Hydrophobic Moment'),
        ('instability_index', 'Instability Index'),
        ('gravy', 'GRAVY Score'),
        ('amphipathicity', 'Amphipathicity Index'),
        ('entropy', 'Shannon Entropy')
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 7), dpi=600)
    axes = axes.flatten()
    
    for idx, (prop, title) in enumerate(viz_properties):
        ax = axes[idx]
        
        # Prepare data
        dataset_data = [df[df['dataset'] == label][prop].values for label in DATASET_LABELS]
        
        # Create violin plot
        parts = ax.violinplot(dataset_data, positions=range(1, len(DATASET_LABELS) + 1),
                              showmeans=True, showmedians=False)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(DATASET_COLORS[i])
            pc.set_alpha(0.7)
        
        # Add statistical annotations
        for i, label in enumerate(DATASET_LABELS):
            data = df[df['dataset'] == label][prop].values
            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data)
                # Position annotation in lower part of plot to avoid covering violins
                y_pos = np.min(data) + (np.max(data) - np.min(data)) * 0.08
                ax.text(i + 1, y_pos, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                       ha='center', va='bottom', fontsize=5, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                               edgecolor=DATASET_COLORS[i], linewidth=1.5, alpha=1.0))
        
        # Formatting
        subplot_letter = chr(ord('a') + idx)
        ax.set_title(f'({subplot_letter}) {title}', fontweight='bold', loc='left', fontsize=9)
        ax.set_xticks(range(1, len(DATASET_LABELS) + 1))
        ax.set_xticklabels(DATASET_LABELS, fontsize=6.5, rotation=20, ha='right')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Property distribution plots saved to: {save_path}")


def plot_charge_distributions(df, save_path):
    """
    Create 1x5 subplot layout for positive charge fraction probability distributions (5 datasets).
    
    Args:
        df (pd.DataFrame): DataFrame with properties
        save_path (str): Path to save the plot
    """
    print("\n" + "=" * 80)
    print("Creating Charge Distribution Plots")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), dpi=600)
    axes = axes.flatten()
    
    # Set consistent axis limits
    x_min, x_max = 0, 1
    y_min, y_max = 0, 5
    
    for idx, (label, color) in enumerate(zip(DATASET_LABELS, DATASET_COLORS)):
        ax = axes[idx]
        data = df[df['dataset'] == label]['positive_fraction'].values
        
        if len(data) > 0:
            # Create histogram
            n, bins, patches = ax.hist(data, bins=20, density=True, alpha=0.7,
                                      color=color, edgecolor='black', linewidth=0.5)
            
            # Calculate statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Add text box
            stats_text = f'μ = {mean_val:.3f}\nσ = {std_val:.3f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   ha='right', va='top', fontsize=6, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor=color, linewidth=1.5, alpha=0.9))
        else:
            ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes,
                   ha='center', va='center', fontsize=6, fontweight='bold')
        
        # Formatting
        subplot_letter = chr(ord('a') + idx)
        ax.set_title(f'({subplot_letter}) {label}', fontweight='bold', loc='left', fontsize=9)
        ax.set_xlabel('Positive Charge Fraction (K, R)', fontsize=8)
        ax.set_ylabel('Probability Density', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Charge distribution plots (1x5) saved to: {save_path}")


def plot_correlation_matrices(df, save_path):
    """
    Create 2x3 subplot layout for correlation matrices (5 datasets).
    
    Args:
        df (pd.DataFrame): DataFrame with properties
        save_path (str): Path to save the plot
    """
    print("\n" + "=" * 80)
    print("Creating Correlation Matrices")
    print("=" * 80)
    
    # Get numeric columns
    numeric_cols = [col for col in df.columns
                   if col not in ['dataset', 'sequence_id'] and df[col].dtype in ['float64', 'int64']]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=600)
    axes = axes.flatten()
    
    # Create shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    
    for idx, (label, color) in enumerate(zip(DATASET_LABELS, DATASET_COLORS)):
        ax = axes[idx]
        dataset_df = df[df['dataset'] == label]
        
        if len(dataset_df) > 0:
            # Calculate correlation matrix
            correlation_matrix = dataset_df[numeric_cols].corr()
            
            # Rename to acronyms
            acronym_cols = [PROPERTY_ACRONYMS.get(col, col) for col in correlation_matrix.columns]
            correlation_matrix.columns = acronym_cols
            correlation_matrix.index = acronym_cols
            
            # Create heatmap
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar=(idx == 4),
                       cbar_ax=cbar_ax if idx == 4 else None, ax=ax)
            
            # Formatting
            subplot_letter = chr(ord('a') + idx)
            ax.set_title(f'({subplot_letter}) {label}', fontweight='bold', loc='left', fontsize=9)
            
            # Set ticks in the middle of boxes
            tick_positions = np.arange(len(acronym_cols)) + 0.5
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(acronym_cols, fontsize=8, rotation=45, ha='center')
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(acronym_cols, fontsize=8)
        else:
            ax.set_title(f'{label} - No Data', fontweight='bold', fontsize=9)
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation matrices (2x3) saved to: {save_path}")
# ==============================================================================
# Main Analysis Function
# ==============================================================================

def analyze_peptide_properties():
    """
    Main function to run comprehensive peptide properties analysis.
    """
    print("\n" + "=" * 80)
    print("Peptide Properties Analysis")
    print("=" * 80 + "\n")
    
    # Load and analyze sequences
    df = load_and_analyze_sequences()
    
    # Save summary statistics
    summary_path = os.path.join(PATH_RESULT, "amyloid_properties_analysis_incDiscard.csv")
    save_summary_statistics(df, summary_path)
    
    # Create visualizations
    dist_plot_path = os.path.join(PATH_RESULT, "amyloid_properties_distributions_incDiscard.png")
    plot_property_distributions(df, dist_plot_path)
    
    charge_plot_path = os.path.join(PATH_RESULT, "positive_charge_probability_distributions_IncDiscard.png")
    plot_charge_distributions(df, charge_plot_path)
    
    corr_plot_path = os.path.join(PATH_RESULT, "dataset_specific_correlations_incDiscard.png")
    plot_correlation_matrices(df, corr_plot_path)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    analyze_peptide_properties()