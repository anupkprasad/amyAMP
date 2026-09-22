import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.expanduser("~/workspace/amyAMP"))
from scripts import util, plotStyle
plotStyle.setPlotStyle()

path_result = os.path.expanduser("~/workspace/amyAMP/results/")
amylogram_dir = os.path.join(path_result, "seq_validation/amylogram")
waltz_file = os.path.join(path_result, "seq_validation/WaltzJob_cutoff_0to100/WaltzJob_1778293937.txt")

# === Helper functions ===
def parse_waltz_file(path):
    """Parse Waltz output into a DataFrame."""
    rows = []
    current_id = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:]  # Remove '>' character
            elif re.match(r"^\d+-\d+\t[A-Z]+\t[0-9]*\.?[0-9]+$", line):
                positions, sequence, score = line.split("\t")
                rows.append({
                    "seq_id": current_id,
                    "positions": positions,
                    "sequence": sequence,
                    "score": float(score)
                })
    return pd.DataFrame(rows)

def parse_amylogram_dir(dir_path):
    """Parse all AmyloGram*.csv files."""
    rows = []
    for fname in os.listdir(dir_path):
        if fname.startswith("AmyloGram") and fname.endswith(".csv"):
            fpath = os.path.join(dir_path, fname)
            df_tmp = pd.read_csv(fpath)
            rows.append(df_tmp[["Input name", "Amyloid probability", "Is amyloid?"]].rename(
                columns={"Input name": "seq_id", "Amyloid probability": "probability", "Is amyloid?": "is_amyloid"}))
    return pd.concat(rows, ignore_index=True)

# === Load data ===
df_ai4amp = pd.read_csv(path_result + 'seq_validation/AI4AMP_seqs_generated_postprocess.csv', sep=',')
df_ai4amp['AMP Score'] = pd.to_numeric(df_ai4amp['AMP Score'], errors='coerce')
df_ai4amp = df_ai4amp.dropna(subset=['AMP Score'])

df_scanner = pd.read_csv(path_result + 'seq_validation/AMPScanner/1763847117927_Prediction_Summary.csv', sep=',')

# === Load simulated sequences from GenSeq.ods (100 sequences selected for MD simulation) ===
df_genseq = pd.read_excel(path_result + 'seq_validation/GenSeq.ods', engine='odf', header=0)
sim_seqs = set(df_genseq['sequence'].dropna().str.strip())
# Match sequences to AMPScanner SeqIDs
sim_ids = set(df_scanner.loc[df_scanner['Sequence'].isin(sim_seqs), 'SeqID'])
print(f"   Simulated sequences identified: {len(sim_ids)}")
df_scanner['Scanner Score'] = pd.to_numeric(df_scanner['Prediction_Probability'], errors='coerce')
df_scanner = df_scanner.dropna(subset=['Scanner Score'])

df_waltz = parse_waltz_file(waltz_file)
# Convert Waltz score to fraction (0-1 range) - assuming original range is 0-100
df_waltz['score'] = df_waltz['score'] / 100.0

df_amylogram = parse_amylogram_dir(amylogram_dir)

# === Merge AI4AMP and Waltz data for scatter plot ===
# Get maximum Waltz score per peptide (already normalized to 0-1)
df_waltz_max = df_waltz.groupby('seq_id')['score'].max().reset_index()
df_waltz_max.rename(columns={'score': 'AMY_Score'}, inplace=True)  # Just rename, don't divide again

# Merge on peptide ID — left join keeps all generated sequences (those without
# a Waltz result will have NaN AMY_Score and won't appear as scatter dots)
df_merged = pd.merge(
    df_ai4amp[['Peptide', 'AMP Score']],
    df_waltz_max,
    left_on='Peptide',
    right_on='seq_id',
    how='left'
)

print(f"\n📊 Merged Data Statistics:")
print(f"   AI4AMP sequences: {len(df_ai4amp)}")
print(f"   Waltz sequences: {len(df_waltz_max)}")
print(f"   Merged sequences: {len(df_merged)}")
print(f"\n   Sample merged data:")
print(df_merged.head())

# === Flag simulated (MD simulation) sequences in merged data ===
df_merged['is_simulated'] = df_merged['Peptide'].isin(sim_ids)
df_plottable = df_merged.dropna(subset=['AMY_Score'])
print(f"   Simulated sequences flagged: {df_merged['is_simulated'].sum()}")
print(f"   Simulated sequences plottable (have Waltz score): {df_plottable['is_simulated'].sum()}")

# === Define bins and labels ===
bins = [0, 0.5, 0.7, 0.9, 1.0]
labels = ['Low (0-0.5)', 'Medium (0.5-0.7)', 'High (0.7-0.9)', 'Very High (0.9-1.0)']
colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF']

# === Create pie charts ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# AI4AMP Pie Chart
df_ai4amp['Category'] = pd.cut(df_ai4amp['AMP Score'], bins=bins, labels=labels, include_lowest=True)
category_counts_ai4amp = df_ai4amp['Category'].value_counts().reindex(labels)
axes[0, 0].pie(category_counts_ai4amp, labels=category_counts_ai4amp.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 10})
axes[0, 0].set_title('(a) AI4AMP', fontsize=12, fontweight='bold', loc='left')

# AMPScanner Pie Chart
df_scanner['Category'] = pd.cut(df_scanner['Scanner Score'], bins=bins, labels=labels, include_lowest=True)
category_counts_scanner = df_scanner['Category'].value_counts().reindex(labels)
axes[0, 1].pie(category_counts_scanner, labels=category_counts_scanner.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 10})
axes[0, 1].set_title('(b) AMPScanner', fontsize=12, fontweight='bold', loc='left')

# Waltz Pie Chart - with custom autopct to avoid overlap
df_waltz['Category'] = pd.cut(df_waltz['score'], bins=bins, labels=labels, include_lowest=True)
category_counts_waltz = df_waltz['Category'].value_counts().reindex(labels)

def autopct_format(pct):
    return f'{pct:.1f}%' if pct > 0 else ''

# Only show labels for categories with values > 0
waltz_labels = [label if count > 0 else '' for label, count in zip(labels, category_counts_waltz)]
axes[1, 0].pie(category_counts_waltz, labels=waltz_labels, autopct=autopct_format, 
               colors=colors, startangle=45, textprops={'fontsize': 10}, labeldistance=1.1, pctdistance=0.85)
axes[1, 0].set_title('(c) Waltz', fontsize=12, fontweight='bold', loc='left')

# AmyloGram Pie Chart
df_amylogram['Category'] = pd.cut(df_amylogram['probability'], bins=bins, labels=labels, include_lowest=True)
category_counts_amylogram = df_amylogram['Category'].value_counts().reindex(labels)
axes[1, 1].pie(category_counts_amylogram, labels=category_counts_amylogram.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 10})
axes[1, 1].set_title('(d) AmyloGram', fontsize=12, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(path_result + 'seq_validation/combined_pie_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# === Create probability distribution plots ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Common histogram bins for all plots
hist_bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1

# AI4AMP Probability Distribution
axes[0, 0].hist(df_ai4amp['AMP Score'], bins=hist_bins, edgecolor='black', alpha=0.7, color='#4ECDC4')
mean_ai4amp = df_ai4amp['AMP Score'].mean()
median_ai4amp = df_ai4amp['AMP Score'].median()
axes[0, 0].axvline(mean_ai4amp, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_ai4amp:.2f}')
axes[0, 0].axvline(median_ai4amp, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_ai4amp:.2f}')
axes[0, 0].set_title('(a) AI4AMP', fontsize=12, fontweight='bold', loc='left')
axes[0, 0].set_xlabel('Score', fontsize=10)
axes[0, 0].set_ylabel('Peptides count', fontsize=10)
axes[0, 0].set_xlim(0, 1)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# AMPScanner Probability Distribution
axes[0, 1].hist(df_scanner['Scanner Score'], bins=hist_bins, edgecolor='black', alpha=0.7, color='#FF6B6B')
mean_scanner = df_scanner['Scanner Score'].mean()
median_scanner = df_scanner['Scanner Score'].median()
axes[0, 1].axvline(mean_scanner, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_scanner:.2f}')
axes[0, 1].axvline(median_scanner, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_scanner:.2f}')
axes[0, 1].set_title('(b) AMPScanner', fontsize=12, fontweight='bold', loc='left')
axes[0, 1].set_xlabel('Score', fontsize=10)
axes[0, 1].set_ylabel('Peptides count', fontsize=10)
axes[0, 1].set_xlim(0, 1)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# Waltz Probability Distribution (now fractional)
axes[1, 0].hist(df_waltz['score'], bins=hist_bins, edgecolor='black', alpha=0.7, color='#6BCB77')
mean_waltz = df_waltz['score'].mean()
median_waltz = df_waltz['score'].median()
axes[1, 0].axvline(mean_waltz, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_waltz:.2f}')
axes[1, 0].axvline(median_waltz, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_waltz:.2f}')
axes[1, 0].set_title('(c) Waltz', fontsize=12, fontweight='bold', loc='left')
axes[1, 0].set_xlabel('Score', fontsize=10)
axes[1, 0].set_ylabel('Peptides count', fontsize=10)
axes[1, 0].set_xlim(0, 1)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)

# AmyloGram Probability Distribution
axes[1, 1].hist(df_amylogram['probability'], bins=hist_bins, edgecolor='black', alpha=0.7, color='#FFD93D')
mean_amylogram = df_amylogram['probability'].mean()
median_amylogram = df_amylogram['probability'].median()
axes[1, 1].axvline(mean_amylogram, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_amylogram:.2f}')
axes[1, 1].axvline(median_amylogram, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_amylogram:.2f}')
axes[1, 1].set_title('(d) AmyloGram', fontsize=12, fontweight='bold', loc='left')
axes[1, 1].set_xlabel('Score', fontsize=10)
axes[1, 1].set_ylabel('Peptides count', fontsize=10)
axes[1, 1].set_xlim(0, 1)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(path_result + 'seq_validation/combined_probability_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# === Create scatter plot: AMP Score vs AMY Score ===
fig, ax = plt.subplots(figsize=(4, 4))

def quadrant_label(count, total):
    percentage = (count / total * 100.0) if total > 0 else 0.0
    return f"n={count}\n{percentage:.1f}%"

# Create scatter plot — non-simulated (green) first, simulated (orange) on top
# Use df_plottable (rows with valid AMY_Score) for actual dots
df_other = df_plottable[~df_plottable['is_simulated']]
df_sim = df_plottable[df_plottable['is_simulated']]

ax.scatter(df_other['AMP Score'], df_other['AMY_Score'],
           alpha=0.5, s=30, c='#2ECC40', edgecolors='black', linewidth=0.4,
           label=f'Generated (n={len(df_merged)})')
ax.scatter(df_sim['AMP Score'], df_sim['AMY_Score'],
           alpha=0.85, s=45, c='#B10DC9', edgecolors='black', linewidth=0.6,
           label=f'Simulated (n={df_merged["is_simulated"].sum()})', zorder=5)

# Add quadrant lines at 0.5
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

# Calculate correlation (only plottable rows)
correlation = df_plottable['AMP Score'].corr(df_plottable['AMY_Score'])

# Count peptides in each quadrant (only plottable rows)
high_amp_high_amy = len(df_plottable[(df_plottable['AMP Score'] >= 0.5) & (df_plottable['AMY_Score'] >= 0.5)])
high_amp_low_amy = len(df_plottable[(df_plottable['AMP Score'] >= 0.5) & (df_plottable['AMY_Score'] < 0.5)])
low_amp_high_amy = len(df_plottable[(df_plottable['AMP Score'] < 0.5) & (df_plottable['AMY_Score'] >= 0.5)])
low_amp_low_amy = len(df_plottable[(df_plottable['AMP Score'] < 0.5) & (df_plottable['AMY_Score'] < 0.5)])
total_quadrant = len(df_plottable)

# Add statistics
stats_text = f'N = {len(df_plottable)}'
ax.text(0.05, 1.16, stats_text, transform=ax.transAxes,
        fontsize=9, fontweight='bold', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=1.0),
        clip_on=False, zorder=20)

# Add quadrant labels with percentages
ax.text(0.25, 0.95, f'High AMY\nLow AMP\n{quadrant_label(high_amp_low_amy, total_quadrant)}',
    transform=ax.transAxes, fontsize=7, ha='center', va='top', color='gray', style='italic',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=1.0),
    clip_on=False, zorder=20)
ax.text(0.75, 0.95, f'High AMY\nHigh AMP\n{quadrant_label(high_amp_high_amy, total_quadrant)}',
    transform=ax.transAxes, fontsize=7, ha='center', va='top', color='gray', style='italic',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=1.0),
    clip_on=False, zorder=20)
ax.text(0.25, 0.05, f'Low AMY\nLow AMP\n{quadrant_label(low_amp_low_amy, total_quadrant)}',
    transform=ax.transAxes, fontsize=7, ha='center', va='bottom', color='gray', style='italic',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=1.0),
    clip_on=False, zorder=20)
ax.text(0.75, 0.05, f'Low AMY\nHigh AMP\n{quadrant_label(low_amp_high_amy, total_quadrant)}',
    transform=ax.transAxes, fontsize=7, ha='center', va='bottom', color='gray', style='italic',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=1.0),
    clip_on=False, zorder=20)

# Add marginal probability distributions on the panel spines
divider = make_axes_locatable(ax)
ax_histx = divider.append_axes("top", size="18%", pad=0.08, sharex=ax)
ax_histy = divider.append_axes("right", size="18%", pad=0.08, sharey=ax)

hist_color = '#2ECC40'
ax_histx.hist(df_plottable['AMP Score'], bins=np.linspace(0, 1, 21), density=False,
          color=hist_color, edgecolor='black', alpha=0.65)
ax_histy.hist(df_plottable['AMY_Score'], bins=np.linspace(0, 1, 21), density=False,
          orientation='horizontal', color=hist_color, edgecolor='black', alpha=0.65)

ax_histx.set_ylabel('Count', fontsize=8)
ax_histy.set_xlabel('Count', fontsize=8)

# Clean marginal axes to look like spines
ax_histx.tick_params(axis='x', labelbottom=False)
ax_histx.tick_params(axis='y', labelsize=7)
ax_histy.tick_params(axis='y', labelleft=False)
ax_histy.tick_params(axis='x', labelsize=7)

for spine_ax in (ax_histx, ax_histy):
    spine_ax.grid(alpha=0.15, linestyle=':')
    spine_ax.spines['top'].set_visible(False)
    spine_ax.spines['right'].set_visible(False)

# Formatting
ax.set_xlabel('AMP Score (AI4AMP)', fontsize=11)
ax.set_ylabel('AMY Score (Waltz)', fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
leg = ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.35),
                ncol=2, frameon=False)
leg.set_zorder(30)

# Equal aspect ratio for better visualization
ax.set_aspect('equal', adjustable='box')
fig.subplots_adjust(top=0.82, bottom=0.12, right=0.88)
ax_histx.set_title('')
plt.tight_layout()
plt.savefig(path_result + 'seq_validation/amp_vs_amy_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# === Print correlation summary ===
print("\n" + "="*70)
print("AMP vs AMY Score Correlation Analysis")
print("="*70)
print(f"Total peptides analyzed: {len(df_merged)}")
print(f"Pearson correlation coefficient: {correlation:.4f}")
print(f"\nScore statistics:")
print(f"  AMP Score (AI4AMP):")
print(f"    Mean: {df_merged['AMP Score'].mean():.3f}")
print(f"    Std:  {df_merged['AMP Score'].std():.3f}")
print(f"  AMY Score (Waltz):")
print(f"    Mean: {df_merged['AMY_Score'].mean():.3f}")
print(f"    Std:  {df_merged['AMY_Score'].std():.3f}")

print(f"\nQuadrant distribution (threshold=0.5):")
print(f"  High AMP & High AMY: {high_amp_high_amy} ({high_amp_high_amy/len(df_merged)*100:.1f}%)")
print(f"  High AMP & Low AMY:  {high_amp_low_amy} ({high_amp_low_amy/len(df_merged)*100:.1f}%)")
print(f"  Low AMP & High AMY:  {low_amp_high_amy} ({low_amp_high_amy/len(df_merged)*100:.1f}%)")
print(f"  Low AMP & Low AMY:   {low_amp_low_amy} ({low_amp_low_amy/len(df_merged)*100:.1f}%)")
print("="*70)

# === Amino acid frequency comparison: generated vs simulated ===
amino_acids = sorted("ACDEFGHIKLMNPQRSTVWY")

def aa_fractions(sequences):
    counts = {aa: 0 for aa in amino_acids}
    for seq in sequences:
        for aa in str(seq):
            if aa in counts:
                counts[aa] += 1
    total = sum(counts.values())
    return [counts[aa] / total if total > 0 else 0 for aa in amino_acids]

gen_seqs = df_scanner['Sequence'].dropna().tolist()
sim_seqs_list = df_genseq['sequence'].dropna().str.strip().tolist()

gen_fracs = aa_fractions(gen_seqs)
sim_fracs = aa_fractions(sim_seqs_list)

x = np.arange(len(amino_acids))
width = 0.38

fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(x - width/2, gen_fracs, width, label=f'Generated (n={len(gen_seqs)})',
       color='#2ECC40', edgecolor='black', alpha=0.8, linewidth=0.5)
ax.bar(x + width/2, sim_fracs, width, label=f'Simulated (n={len(sim_seqs_list)})',
       color='#B10DC9', edgecolor='black', alpha=0.8, linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(amino_acids, fontsize=10)
ax.set_xlabel('Amino Acid', fontsize=11)
ax.set_ylabel('Fraction', fontsize=11)
ax.tick_params(axis='y', labelsize=10)
ax.legend(fontsize=8, ncol=1)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(path_result + 'seq_validation/amino_acid_frequency_gen_vs_sim.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nAmino acid frequency plot saved.")