from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import sys
sys.path.append(os.path.expanduser("~/workspace/amyAMP"))
from scripts import util, plotStyle
plotStyle.setPlotStyle()



# Load sequences from the four files
path = os.path.expanduser("~/workspace/amyAMP/results/")
fasta_files = ["seqs_generated_postprocessed.fasta", "seqs_realAMPs1000.fasta", "seqs_realAMYs1000.fasta", "random_peptides_1000.fasta"]
file_labels = ["amyAMP", "AMP", "AMY", "randPep"]  # Updated to match four datasets
colors = ['#E63946', '#06FFA5', "#754DE3", '#FFD60A']  # Updated to match four datasets

def calculate_hydrophobic_moment(seq):
    """Calculate hydrophobic moment using Kyte-Doolittle hydropathy scale"""
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

def amyloid_properties(seq):
    """Calculate comprehensive amyloid-relevant properties for a sequence"""
    try:
        seq_str = str(seq).upper()
        p = ProteinAnalysis(seq_str)
        hydrophobic_residues = set("AILFWVY")
        aromatic_residues = set("FYWH")
        charged_positive = set("KR")
        charged_negative = set("DE")
        polar_residues = set("NQSTC")
        length = len(seq_str)
        charge = p.charge_at_pH(7.0)
        gravy = p.gravy()
        aromaticity = p.aromaticity()
        instability = p.instability_index()
        isoelectric_point = p.isoelectric_point()
        hydrophobic_fraction = sum([seq_str.count(r) for r in hydrophobic_residues]) / length
        aromatic_fraction = sum([seq_str.count(r) for r in aromatic_residues]) / length
        positive_fraction = sum([seq_str.count(r) for r in charged_positive]) / length
        negative_fraction = sum([seq_str.count(r) for r in charged_negative]) / length
        polar_fraction = sum([seq_str.count(r) for r in polar_residues]) / length
        net_charge = positive_fraction - negative_fraction
        charge_density = abs(charge) / length
        aa_counts = Counter(seq_str)
        entropy = -sum([(count/length) * np.log2(count/length) for count in aa_counts.values()])
        charge_hydrophobic_ratio = abs(charge) / (hydrophobic_fraction + 1e-6)
        aromatic_hydrophobic_ratio = aromatic_fraction / (hydrophobic_fraction + 1e-6)
        beta_sheet_residues = set("VIFYL")
        beta_propensity = sum([seq_str.count(r) for r in beta_sheet_residues]) / length
        amp_favorable_residues = set("KRFWC")
        amp_favorable_fraction = sum([seq_str.count(r) for r in amp_favorable_residues]) / length
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

# Load and analyze all datasets
all_data = []
dataset_names = []

print("Loading and analyzing sequences...")
for i, fasta_file in enumerate(fasta_files):
    print(f"Processing {file_labels[i]}...")
    seq_records = list(SeqIO.parse(path + "sequence/"+ fasta_file, "fasta"))
    for record in seq_records:
        properties = amyloid_properties(record.seq)
        if properties:
            properties['dataset'] = file_labels[i]
            properties['sequence_id'] = record.id
            all_data.append(properties)

# Create DataFrame
df = pd.DataFrame(all_data)
print(f"\nTotal sequences analyzed: {len(df)}")
print(f"Datasets: {df['dataset'].value_counts().to_dict()}")

# Display summary statistics
print("\n" + "="*80)
print("AMYLOID AND ANTIMICROBIAL PROPERTIES SUMMARY STATISTICS")
print("="*80)

# Select key amyloid and antimicrobial-relevant properties for summary
key_properties = [
    'length', 'charge', 'isoelectric_point', 'hydrophobic_moment', 'instability_index', 
    'gravy', 'entropy', 'beta_propensity',
    'amp_favorable_fraction', 'amphipathicity'
]

summary_stats = df.groupby('dataset')[key_properties].agg(['mean', 'std', 'min', 'max'])
print(summary_stats.round(3))

# Save detailed results
df.to_csv(path + "amyloid_properties_analysis.csv", index=False)
print(f"\nDetailed results saved to: amyloid_properties_analysis.csv")

# Create visualizations
plt.style.use('default')

# Properties to visualize (amyloid and antimicrobial relevant)
viz_properties = [
    ('positive_fraction', 'Positive Charge Fraction'),
    ('amp_favorable_fraction', 'AMP-Favorable Residues'),
    ('isoelectric_point', 'Isoelectric Point'),
    ('beta_propensity', 'Beta-sheet Propensity'),
    ('hydrophobic_moment', 'Hydrophobic Moment'),
    ('instability_index', 'Instability Index'),
    ('gravy', 'GRAVY Score'),
    ('amphipathicity', 'Amphipathicity Index'),
    ('entropy', 'Shannon Entropy')
]

# Create 3x3 violin plots
fig, axes = plt.subplots(3, 3, figsize=(10, 8))  # Adjusted figure size for clarity

for idx, (prop, title) in enumerate(viz_properties):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Create violin plot for all 4 datasets
    dataset_data = [df[df['dataset'] == label][prop].values for label in file_labels]
    parts = ax.violinplot(dataset_data, positions=range(1, len(file_labels) + 1), showmeans=True, showmedians=False)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add statistical annotations
    for i, label in enumerate(file_labels):
        data = df[df['dataset'] == label][prop].values
        if len(data) > 0:  # Check if data exists for this dataset
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.text(i + 1, max(data) * 0.9, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                    ha='center', va='top', fontsize=6, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=colors[i], 
                             linewidth=1.5, alpha=1.0))
    
    # Add subplot numbering (a), (b), etc. with left alignment
    subplot_letter = chr(ord('a') + idx)
    ax.set_title(f'({subplot_letter}) {title}', fontweight='bold', loc='left', fontsize=8)  # Title font size = 8
    ax.set_xticks(range(1, len(file_labels) + 1))
    ax.set_xticklabels(file_labels, fontsize=7)  # x-tick labels font size = 7
    ax.tick_params(axis='y', labelsize=7)  # y-tick labels font size = 7
    ax.grid(axis='y', alpha=0.3)

    # Remove y-labels for clarity
    ax.set_ylabel('')  # No y-label

plt.tight_layout()
plt.savefig(path + "amyloid_properties_distributions.png", dpi=600, bbox_inches='tight')
plt.show()

# Create a 2x2 subplot layout for positive charge fraction probability distributions
print("\n" + "="*60)
print("POSITIVE CHARGE FRACTION PROBABILITY DISTRIBUTIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns

# Set consistent x and y limits for all subplots
x_min, x_max = 0, 1  # Adjust based on your data range
y_min, y_max = 0, 5  # Adjust based on your data range

for idx, (label, color) in enumerate(zip(file_labels, colors)):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Get positive charge fraction data for current dataset
    data = df[df['dataset'] == label]['positive_fraction'].values
    
    if len(data) > 0:
        # Create histogram (probability distribution) - bars only
        n, bins, patches = ax.hist(data, bins=20, density=True, alpha=0.7, 
                                   color=color, edgecolor='black', linewidth=0.5)
        
        # Add statistical information (text box only)
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Add text box with statistics
        stats_text = f'μ = {mean_val:.3f}\nσ = {std_val:.3f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                ha='right', va='top', fontsize=6, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor=color, linewidth=1.5, alpha=0.9))
        
        ax.set_xlabel('Positive Charge Fraction (K, R)', fontsize=7)  # x-label font size = 7
        ax.set_ylabel('Probability Density', fontsize=7)  # y-label font size = 7
        
    else:
        ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=6, fontweight='bold')  # Annotation font size = 6
        ax.set_xlabel('Positive Charge Fraction (K, R)', fontsize=7)  # x-label font size = 7
        ax.set_ylabel('Probability Density', fontsize=7)  # y-label font size = 7
    
    # Add subplot numbering (a), (b), etc. with left alignment
    subplot_letter = chr(ord('a') + idx)
    ax.set_title(f'({subplot_letter}) {label}', fontweight='bold', loc='left', fontsize=8)  # Title font size = 8
    ax.grid(axis='y', alpha=0.3)

    # Set consistent x and y limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# Adjust layout and save the combined figure
plt.tight_layout()
plt.savefig(path + "positive_charge_probability_distributions_2x2.png", dpi=600, bbox_inches='tight')
plt.show()

numeric_cols = [col for col in df.columns if col not in ['dataset', 'sequence_id'] and df[col].dtype in ['float64', 'int64']]

# Create acronym mapping for better visualization
acronym_mapping = {
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
# Create a 2x2 subplot layout for correlation matrices with a single color bar
fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [1, 1]})  # 2 rows, 2 columns

# Create a shared color bar axis
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # Position for the shared color bar

for idx, (label, color) in enumerate(zip(file_labels, colors)):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Filter data for current dataset
    dataset_df = df[df['dataset'] == label]
    
    if len(dataset_df) > 0:
        # Calculate correlation matrix for this dataset
        correlation_matrix = dataset_df[numeric_cols].corr()
        
        # Rename columns and index to acronyms
        acronym_cols = [acronym_mapping.get(col, col) for col in correlation_matrix.columns]
        correlation_matrix.columns = acronym_cols
        correlation_matrix.index = acronym_cols
        
        # Create heatmap without annotations
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', cbar=(idx == 3), cbar_ax=cbar_ax if idx == 3 else None, ax=ax)
        
        # Add subplot numbering and title
        subplot_letter = chr(ord('a') + idx)
        ax.set_title(f'({subplot_letter}) {label}', fontweight='bold', loc='left', fontsize=8)  # Title font size = 8
        
        # Ensure all x-tick and y-tick labels appear in the middle of the boxes
        tick_positions = np.arange(len(acronym_cols)) + 0.5  # Center of each box
        ax.set_xticks(tick_positions)  # Set x-ticks explicitly
        ax.set_xticklabels(acronym_cols, fontsize=7, rotation=45, ha='center')  # Ensure all x-tick labels appear
        ax.set_yticks(tick_positions)  # Set y-ticks explicitly
        ax.set_yticklabels(acronym_cols, fontsize=7)  # Ensure all y-tick labels appear
    else:
        ax.set_title(f'{label} - No Data', fontweight='bold', fontsize=8)  # Title font size = 8
        ax.axis('off')

# Adjust layout and save the combined figure
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
plt.savefig(path + "dataset_specific_correlations_2x2.png", dpi=600, bbox_inches='tight')
plt.show()