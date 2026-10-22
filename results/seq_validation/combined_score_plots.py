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
# Convert Waltz score to fraction (0-1 range) - WaltzJob cutoff 0-100
df_waltz['score'] = df_waltz['score'] / 100.0

df_amylogram = parse_amylogram_dir(amylogram_dir)

# === Canonical peptide set for Figure 4: AI4AMP ∩ Waltz (one score per peptide) ===
# Waltz can return multiple APR windows per peptide; use the maximum score per peptide
# so pie charts and the scatter plot share the same statistic and the same n.
df_waltz_max = df_waltz.groupby('seq_id')['score'].max().reset_index()
df_waltz_max.rename(columns={'score': 'AMY_Score'}, inplace=True)

df_merged = pd.merge(
    df_ai4amp[['Peptide', 'AMP Score']],
    df_waltz_max,
    left_on='Peptide',
    right_on='seq_id',
    how='left'
)

# Analysis set used for ALL Figure 4 panels (a-e): peptides with both scores
df_plottable = df_merged.dropna(subset=['AMY_Score']).copy()
canonical_ids = set(df_plottable['Peptide'])

# Restrict AMPScanner / AmyloGram to the same peptide IDs
df_scanner = df_scanner[df_scanner['SeqID'].isin(canonical_ids)].copy()
df_amylogram = df_amylogram[df_amylogram['seq_id'].isin(canonical_ids)].copy()

# One-row-per-peptide frames for pies (aligned with scatter)
df_ai4amp_fig = df_plottable[['Peptide', 'AMP Score']].copy()
df_waltz_fig = df_plottable[['Peptide', 'AMY_Score']].rename(columns={'AMY_Score': 'score'}).copy()

print(f"\nMerged Data Statistics (Figure 4 canonical set):")
print(f"   AI4AMP sequences (raw): {len(df_ai4amp)}")
print(f"   Waltz unique peptides (max score): {len(df_waltz_max)}")
print(f"   Canonical set (AI4AMP ∩ Waltz): {len(df_plottable)}")
print(f"   Excluded (AI4AMP without Waltz): {int(df_merged['AMY_Score'].isna().sum())}")
print(f"   AMPScanner after filter: {len(df_scanner)}")
print(f"   AmyloGram after filter: {len(df_amylogram)}")

# === Flag simulated (MD simulation) sequences in merged data ===
df_plottable['is_simulated'] = df_plottable['Peptide'].isin(sim_ids)
print(f"   Simulated sequences in canonical set: {df_plottable['is_simulated'].sum()}")

# === Define bins and labels ===
bins = [0, 0.5, 0.7, 0.9, 1.0]
labels = ['Low (0-0.5)', 'Medium (0.5-0.7)', 'High (0.7-0.9)', 'Very High (0.9-1.0)']
colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF']

# === Create pie charts (all panels: same n = canonical AI4AMP ∩ Waltz set) ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
n_fig = len(df_plottable)

# AI4AMP Pie Chart
df_ai4amp_fig['Category'] = pd.cut(df_ai4amp_fig['AMP Score'], bins=bins, labels=labels, include_lowest=True)
category_counts_ai4amp = df_ai4amp_fig['Category'].value_counts().reindex(labels)
axes[0, 0].pie(category_counts_ai4amp, labels=category_counts_ai4amp.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 10})
axes[0, 0].set_title(f'(a) AI4AMP (n={n_fig})', fontsize=12, fontweight='bold', loc='left')

# AMPScanner Pie Chart
df_scanner['Category'] = pd.cut(df_scanner['Scanner Score'], bins=bins, labels=labels, include_lowest=True)
category_counts_scanner = df_scanner['Category'].value_counts().reindex(labels)
axes[0, 1].pie(category_counts_scanner, labels=category_counts_scanner.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 10})
axes[0, 1].set_title(f'(b) AMPScanner (n={n_fig})', fontsize=12, fontweight='bold', loc='left')

# Waltz Pie Chart — one max score per peptide (same as scatter)
df_waltz_fig['Category'] = pd.cut(df_waltz_fig['score'], bins=bins, labels=labels, include_lowest=True)
category_counts_waltz = df_waltz_fig['Category'].value_counts().reindex(labels)

def autopct_format(pct):
    return f'{pct:.1f}%' if pct > 0 else ''

waltz_labels = [label if count > 0 else '' for label, count in zip(labels, category_counts_waltz)]
axes[1, 0].pie(category_counts_waltz, labels=waltz_labels, autopct=autopct_format, 
               colors=colors, startangle=45, textprops={'fontsize': 10}, labeldistance=1.1, pctdistance=0.85)
axes[1, 0].set_title(f'(c) Waltz (n={n_fig})', fontsize=12, fontweight='bold', loc='left')

# AmyloGram Pie Chart
df_amylogram['Category'] = pd.cut(df_amylogram['probability'], bins=bins, labels=labels, include_lowest=True)
category_counts_amylogram = df_amylogram['Category'].value_counts().reindex(labels)
axes[1, 1].pie(category_counts_amylogram, labels=category_counts_amylogram.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 10})
axes[1, 1].set_title(f'(d) AmyloGram (n={n_fig})', fontsize=12, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(path_result + 'seq_validation/combined_pie_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# Print pie fractions for manuscript / response
print("\n=== Figure 4 pie category % (canonical n) ===")
for name, counts in [
    ('AI4AMP', category_counts_ai4amp),
    ('AMPScanner', category_counts_scanner),
    ('Waltz', category_counts_waltz),
    ('AmyloGram', category_counts_amylogram),
]:
    pct = (counts / counts.sum() * 100).round(1)
    print(name, pct.to_dict())

# === Create probability distribution plots ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Common histogram bins for all plots
hist_bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1

# AI4AMP Probability Distribution (canonical set)
axes[0, 0].hist(df_ai4amp_fig['AMP Score'], bins=hist_bins, edgecolor='black', alpha=0.7, color='#4ECDC4')
mean_ai4amp = df_ai4amp_fig['AMP Score'].mean()
median_ai4amp = df_ai4amp_fig['AMP Score'].median()
axes[0, 0].axvline(mean_ai4amp, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_ai4amp:.2f}')
axes[0, 0].axvline(median_ai4amp, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_ai4amp:.2f}')
axes[0, 0].set_title(f'(a) AI4AMP (n={n_fig})', fontsize=12, fontweight='bold', loc='left')
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
axes[0, 1].set_title(f'(b) AMPScanner (n={n_fig})', fontsize=12, fontweight='bold', loc='left')
axes[0, 1].set_xlabel('Score', fontsize=10)
axes[0, 1].set_ylabel('Peptides count', fontsize=10)
axes[0, 1].set_xlim(0, 1)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# Waltz Probability Distribution — max score per peptide (canonical set)
axes[1, 0].hist(df_waltz_fig['score'], bins=hist_bins, edgecolor='black', alpha=0.7, color='#6BCB77')
mean_waltz = df_waltz_fig['score'].mean()
median_waltz = df_waltz_fig['score'].median()
axes[1, 0].axvline(mean_waltz, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_waltz:.2f}')
axes[1, 0].axvline(median_waltz, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_waltz:.2f}')
axes[1, 0].set_title(f'(c) Waltz (n={n_fig})', fontsize=12, fontweight='bold', loc='left')
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
axes[1, 1].set_title(f'(d) AmyloGram (n={n_fig})', fontsize=12, fontweight='bold', loc='left')
axes[1, 1].set_xlabel('Score', fontsize=10)
axes[1, 1].set_ylabel('Peptides count', fontsize=10)
axes[1, 1].set_xlim(0, 1)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(path_result + 'seq_validation/combined_probability_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# === Create scatter plot: AMP Score vs AMY Score ===
fig, ax = plt.subplots(figsize=(4.2, 4.2))

def quadrant_label(count, total):
    percentage = (count / total * 100.0) if total > 0 else 0.0
    return f"n={count}\n{percentage:.1f}%"

# Create scatter plot — non-simulated (green) first, simulated (purple) on top
# Use df_plottable (rows with valid AMY_Score) for green + most simulated points.
# One GenSeq peptide lacks a Waltz window; include it on the scatter with AMY_Score=0
# so the simulated overlay has all n=100 selected MD peptides.
df_other = df_plottable[~df_plottable['is_simulated']]
df_sim = df_plottable[df_plottable['is_simulated']].copy()
n_sim_selected = len(sim_ids)  # GenSeq MD selection size (100)

missing_sim_ids = sorted(sim_ids - set(df_plottable.loc[df_plottable['is_simulated'], 'Peptide']))
if missing_sim_ids:
    df_miss = df_ai4amp[df_ai4amp['Peptide'].isin(missing_sim_ids)][['Peptide', 'AMP Score']].copy()
    df_miss['AMY_Score'] = 0.0  # no Waltz APR window returned → treat as score 0 for display
    df_miss['is_simulated'] = True
    df_sim = pd.concat([df_sim, df_miss], ignore_index=True)
    print(
        f"Note: {len(missing_sim_ids)} simulated peptide(s) had no Waltz hit; "
        f"plotted with AMY_Score=0 so Simulated n={n_sim_selected}."
    )

ax.scatter(df_other['AMP Score'], df_other['AMY_Score'],
           alpha=0.5, s=30, c='#2ECC40', edgecolors='black', linewidth=0.4,
           label=f'Generated (n={len(df_plottable)})')
ax.scatter(df_sim['AMP Score'], df_sim['AMY_Score'],
           alpha=0.85, s=45, c='#B10DC9', edgecolors='black', linewidth=0.6,
           label=f'Simulated (n={n_sim_selected})', zorder=5)

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
ax.text(0.25, 0.95, f'High AMY\nLow AMP\n{quadrant_label(low_amp_high_amy, total_quadrant)}',
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
ax.text(0.75, 0.05, f'Low AMY\nHigh AMP\n{quadrant_label(high_amp_low_amy, total_quadrant)}',
    transform=ax.transAxes, fontsize=7, ha='center', va='bottom', color='gray', style='italic',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=1.0),
    clip_on=False, zorder=20)

# Add marginal probability distributions on the panel spines
divider = make_axes_locatable(ax)
ax_histx = divider.append_axes("top", size="18%", pad=0.12, sharex=ax)
ax_histy = divider.append_axes("right", size="18%", pad=0.18, sharey=ax)

hist_color = '#9E9E9E'  # gray marginals (avoid clashing with green scatter)
ax_histx.hist(df_plottable['AMP Score'], bins=np.linspace(0, 1, 21), density=False,
          color=hist_color, edgecolor='black', alpha=0.7)
ax_histy.hist(df_plottable['AMY_Score'], bins=np.linspace(0, 1, 21), density=False,
          orientation='horizontal', color=hist_color, edgecolor='black', alpha=0.7)

ax_histx.set_ylabel('Count', fontsize=8)
ax_histy.set_xlabel('Count', fontsize=8, labelpad=6)

# Clean marginal axes to look like spines
ax_histx.tick_params(axis='x', labelbottom=False)
ax_histx.tick_params(axis='y', labelsize=7)
ax_histy.tick_params(axis='y', labelleft=False)
ax_histy.tick_params(axis='x', labelsize=7, pad=2)
# Keep count-axis ticks including 0; spacing/pad avoids clash with scatter x=1.0
ax_histy.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True, min_n_ticks=3))

for spine_ax in (ax_histx, ax_histy):
    spine_ax.grid(alpha=0.15, linestyle=':')
    spine_ax.spines['top'].set_visible(False)
    spine_ax.spines['right'].set_visible(False)

# Formatting
ax.set_xlabel('AMP Score (AI4AMP)', fontsize=11, labelpad=4)
ax.set_ylabel('AMY Score (Waltz)', fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# Shift scatter "1.0" slightly left of the corner so hist count "0" can show
ax.tick_params(axis='x', pad=3)
for label in ax.get_xticklabels():
    if label.get_text() in ('1.0', '1'):
        label.set_horizontalalignment('right')
ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
leg = ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.35),
                ncol=2, frameon=False)
leg.set_zorder(30)

# Equal aspect ratio for better visualization
ax.set_aspect('equal', adjustable='box')
fig.subplots_adjust(top=0.82, bottom=0.14, right=0.84)
ax_histx.set_title('')
plt.tight_layout()
# Extra gap between scatter and right histogram so "1.0" and count "0" do not overlap
pos = ax.get_position()
pos_y = ax_histy.get_position()
gap = 0.012
ax_histy.set_position([pos.x1 + gap, pos_y.y0, pos_y.width, pos_y.height])
plt.savefig(path_result + 'seq_validation/amp_vs_amy_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# === Print correlation summary ===
print("\n" + "="*70)
print("AMP vs AMY Score Correlation Analysis (canonical set)")
print("="*70)
print(f"Total peptides analyzed: {len(df_plottable)}")
print(f"Pearson correlation coefficient: {correlation:.4f}")
print(f"\nScore statistics:")
print(f"  AMP Score (AI4AMP):")
print(f"    Mean: {df_plottable['AMP Score'].mean():.3f}")
print(f"    Median: {df_plottable['AMP Score'].median():.3f}")
print(f"    Std:  {df_plottable['AMP Score'].std():.3f}")
print(f"  AMY Score (Waltz, max per peptide, cutoff 0-100):")
print(f"    Mean: {df_plottable['AMY_Score'].mean():.3f}")
print(f"    Median: {df_plottable['AMY_Score'].median():.3f}")
print(f"    Std:  {df_plottable['AMY_Score'].std():.3f}")

print(f"\nQuadrant distribution (threshold=0.5):")
print(f"  High AMP & High AMY: {high_amp_high_amy} ({high_amp_high_amy/len(df_plottable)*100:.1f}%)")
print(f"  High AMP & Low AMY:  {high_amp_low_amy} ({high_amp_low_amy/len(df_plottable)*100:.1f}%)")
print(f"  Low AMP & High AMY:  {low_amp_high_amy} ({low_amp_high_amy/len(df_plottable)*100:.1f}%)")
print(f"  Low AMP & Low AMY:   {low_amp_low_amy} ({low_amp_low_amy/len(df_plottable)*100:.1f}%)")
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
