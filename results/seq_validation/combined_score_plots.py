import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

sys.path.append(os.path.expanduser("~/workspace/amyAMP"))
from scripts import util, plotStyle
plotStyle.setPlotStyle()

path_result = os.path.expanduser("~/workspace/amyAMP/results/")
amylogram_dir = os.path.join(path_result, "seq_validation/amylogram")
waltz_file = os.path.join(path_result, "seq_validation/WaltzJob_cutoff_50to100/WaltzJob_1763851079.txt")

# === Helper functions ===
def parse_waltz_file(path):
    """Parse Waltz output into a DataFrame."""
    rows = []
    current_id = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:]
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
df_ai4amp['AMP Score'] = pd.to_numeric(df_ai4amp['AMP Score'], errors='coerce').dropna()

df_scanner = pd.read_csv(path_result + 'seq_validation/AMPScanner/1763847117927_Prediction_Summary.csv', sep=',')
df_scanner['Scanner Score'] = pd.to_numeric(df_scanner['Prediction_Probability'], errors='coerce').dropna()

df_waltz = parse_waltz_file(waltz_file)
df_amylogram = parse_amylogram_dir(amylogram_dir)

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
               colors=colors, startangle=90, textprops={'fontsize': 10})  # Increased font size
axes[0, 0].set_title('(a) AI4AMP', fontsize=12, fontweight='bold', loc='left')

# AMPScanner Pie Chart
df_scanner['Category'] = pd.cut(df_scanner['Scanner Score'], bins=bins, labels=labels, include_lowest=True)
category_counts_scanner = df_scanner['Category'].value_counts().reindex(labels)
axes[0, 1].pie(category_counts_scanner, labels=category_counts_scanner.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 10})  # Increased font size
axes[0, 1].set_title('(b) AMPScanner', fontsize=12, fontweight='bold', loc='left')

# Waltz Pie Chart
waltz_bins = [0, 50, 60, 70, np.inf]
waltz_labels = ['Low (<50)', 'Moderate (50-60)', 'High (60-70)', 'Very High (>=70)']
df_waltz['Category'] = pd.cut(df_waltz['score'], bins=waltz_bins, labels=waltz_labels, include_lowest=True)
category_counts_waltz = df_waltz['Category'].value_counts().reindex(waltz_labels)
axes[1, 0].pie(category_counts_waltz, labels=category_counts_waltz.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 10})  # Increased font size
axes[1, 0].set_title('(c) Waltz', fontsize=12, fontweight='bold', loc='left')

# AmyloGram Pie Chart
amylogram_bins = [0, 0.5, 0.6, 0.7, np.inf]
amylogram_labels = ['Low (<0.5)', 'Moderate (0.5-0.6)', 'High (0.6-0.7)', 'Very High (>=0.7)']
df_amylogram['Category'] = pd.cut(df_amylogram['probability'], bins=amylogram_bins, labels=amylogram_labels, include_lowest=True)
category_counts_amylogram = df_amylogram['Category'].value_counts().reindex(amylogram_labels)
axes[1, 1].pie(category_counts_amylogram, labels=category_counts_amylogram.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 10})  # Increased font size
axes[1, 1].set_title('(d) AmyloGram', fontsize=12, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(path_result + 'seq_validation/combined_pie_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# === Create probability distribution plots ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# AI4AMP Probability Distribution
axes[0, 0].hist(df_ai4amp['AMP Score'], bins=20, edgecolor='black', alpha=0.7, color='#4ECDC4')
mean_ai4amp = df_ai4amp['AMP Score'].mean()
median_ai4amp = df_ai4amp['AMP Score'].median()
axes[0, 0].axvline(mean_ai4amp, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_ai4amp:.2f}')
axes[0, 0].axvline(median_ai4amp, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_ai4amp:.2f}')
axes[0, 0].set_title('(a) AI4AMP', fontsize=12, fontweight='bold', loc='left')
axes[0, 0].set_xlabel('AMP Score', fontsize=10)
axes[0, 0].set_ylabel('Frequency', fontsize=10)
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# AMPScanner Probability Distribution
axes[0, 1].hist(df_scanner['Scanner Score'], bins=20, edgecolor='black', alpha=0.7, color='#FF6B6B')
mean_scanner = df_scanner['Scanner Score'].mean()
median_scanner = df_scanner['Scanner Score'].median()
axes[0, 1].axvline(mean_scanner, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_scanner:.2f}')
axes[0, 1].axvline(median_scanner, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_scanner:.2f}')
axes[0, 1].set_title('(b) AMPScanner', fontsize=12, fontweight='bold', loc='left')
axes[0, 1].set_xlabel('Scanner Score', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Waltz Probability Distribution
axes[1, 0].hist(df_waltz['score'], bins=20, edgecolor='black', alpha=0.7, color='#6BCB77')
mean_waltz = df_waltz['score'].mean()
median_waltz = df_waltz['score'].median()
axes[1, 0].axvline(mean_waltz, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_waltz:.2f}')
axes[1, 0].axvline(median_waltz, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_waltz:.2f}')
axes[1, 0].set_title('(c) Waltz', fontsize=12, fontweight='bold', loc='left')
axes[1, 0].set_xlabel('Waltz Score', fontsize=10)
axes[1, 0].set_ylabel('Frequency', fontsize=10)
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# AmyloGram Probability Distribution
axes[1, 1].hist(df_amylogram['probability'], bins=20, edgecolor='black', alpha=0.7, color='#FFD93D')
mean_amylogram = df_amylogram['probability'].mean()
median_amylogram = df_amylogram['probability'].median()
axes[1, 1].axvline(mean_amylogram, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_amylogram:.2f}')
axes[1, 1].axvline(median_amylogram, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_amylogram:.2f}')
axes[1, 1].set_title('(d) AmyloGram', fontsize=12, fontweight='bold', loc='left')
axes[1, 1].set_xlabel('Amyloid Probability', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(path_result + 'seq_validation/combined_probability_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
