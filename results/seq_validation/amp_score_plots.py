import sys
import os

sys.path.append(os.path.expanduser("~/workspace/amyAMP"))
from scripts import util, plotStyle
plotStyle.setPlotStyle()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_result = os.path.expanduser("~/workspace/amyAMP/results/")

# === Load AI4AMP data ===
df = pd.read_csv(path_result + 'seq_validation/AI4AMP_seqs_generated_postprocess.csv', sep=',')

# === Clean and convert AMP Score column ===
df['AMP Score'] = pd.to_numeric(df['AMP Score'], errors='coerce')
df = df.dropna(subset=['AMP Score'])

# === Load AMPScanner data ===
df_scanner = pd.read_csv(path_result + 'seq_validation/AMPScanner/1763847117927_Prediction_Summary.csv', sep=',')
df_scanner['Scanner Score'] = pd.to_numeric(df_scanner['Prediction_Probability'], errors='coerce')
df_scanner = df_scanner.dropna(subset=['Scanner Score'])

# === Calculate Statistics ===
mean_score = df['AMP Score'].mean()
median_score = df['AMP Score'].median()
std_score = df['AMP Score'].std()
min_score = df['AMP Score'].min()
max_score = df['AMP Score'].max()

print(f"=== AMP Score Statistics ===")
print(f"Mean: {mean_score:.4f}")
print(f"Median: {median_score:.4f}")
print(f"Std Dev: {std_score:.4f}")
print(f"Min: {min_score:.4f}")
print(f"Max: {max_score:.4f}")
print(f"Total Sequences: {len(df)}")

# === Calculate Statistics for AMPScanner ===
mean_score_scanner = df_scanner['Scanner Score'].mean()
median_score_scanner = df_scanner['Scanner Score'].median()
std_score_scanner = df_scanner['Scanner Score'].std()
min_score_scanner = df_scanner['Scanner Score'].min()
max_score_scanner = df_scanner['Scanner Score'].max()

print(f"\n=== AMPScanner Score Statistics ===")
print(f"Mean: {mean_score_scanner:.4f}")
print(f"Median: {median_score_scanner:.4f}")
print(f"Std Dev: {std_score_scanner:.4f}")
print(f"Min: {min_score_scanner:.4f}")
print(f"Max: {max_score_scanner:.4f}")
print(f"Total Sequences: {len(df_scanner)}")

# === Create subplots ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # unified figsize with amy_score_plots

# === 1. Distribution Histogram - AI4AMP ===
ax1 = axes[0, 0]
ax1.hist(df['AMP Score'], bins=20, edgecolor='black', alpha=0.7, color='#4ECDC4')
ax1.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
ax1.axvline(median_score, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_score:.3f}')
ax1.set_title('(a) AI4AMP Score Distribution', fontsize=14, fontweight='bold', loc='left')
ax1.set_xlabel('AMP Score', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

# === 2. Pie Chart - AI4AMP Score Categories ===
ax2 = axes[0, 1]
bins = [0, 0.5, 0.7, 0.9, 1.0]
labels = ['Low (0-0.5)', 'Medium (0.5-0.7)', 'High (0.7-0.9)', 'Very High (0.9-1.0)']
df['Category'] = pd.cut(df['AMP Score'], bins=bins, labels=labels, include_lowest=True)
category_counts = df['Category'].value_counts().reindex(labels)

colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF']
ax2.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax2.set_title('(b) AI4AMP Score Categories', fontsize=14, fontweight='bold', loc='left')

# === 3. Distribution Histogram - AMPScanner ===
ax3 = axes[1, 0]
ax3.hist(df_scanner['Scanner Score'], bins=20, edgecolor='black', alpha=0.7, color='#FF6B6B')
ax3.axvline(mean_score_scanner, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score_scanner:.3f}')
ax3.axvline(median_score_scanner, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_score_scanner:.3f}')
ax3.set_title('(c) AMPScanner Score Distribution', fontsize=14, fontweight='bold', loc='left')
ax3.set_xlabel('AMPScanner Score', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.legend()
ax3.grid(alpha=0.3)

# === 4. Pie Chart - AMPScanner Score Categories ===
ax4 = axes[1, 1]
df_scanner['Category'] = pd.cut(df_scanner['Scanner Score'], bins=bins, labels=labels, include_lowest=True)
category_counts_scanner = df_scanner['Category'].value_counts().reindex(labels)

ax4.pie(category_counts_scanner, labels=category_counts_scanner.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax4.set_title('(d) AMPScanner Score Categories', fontsize=14, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(path_result + 'seq_validation/amp_score_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# === Additional Analysis: Score Distribution Table ===
print("\n=== AI4AMP Score Distribution by Category ===")
print(category_counts)
print(f"\nPercentage of sequences with score > 0.9: {(df['AMP Score'] > 0.9).sum() / len(df) * 100:.2f}%")
print(f"Percentage of sequences with score > 0.7: {(df['AMP Score'] > 0.7).sum() / len(df) * 100:.2f}%")
print(f"Percentage of sequences with score > 0.5: {(df['AMP Score'] > 0.5).sum() / len(df) * 100:.2f}%")

print("\n=== AMPScanner Score Distribution by Category ===")
print(category_counts_scanner)
print(f"\nPercentage of sequences with score > 0.9: {(df_scanner['Scanner Score'] > 0.9).sum() / len(df_scanner) * 100:.2f}%")
print(f"Percentage of sequences with score > 0.7: {(df_scanner['Scanner Score'] > 0.7).sum() / len(df_scanner) * 100:.2f}%")
print(f"Percentage of sequences with score > 0.5: {(df_scanner['Scanner Score'] > 0.5).sum() / len(df_scanner) * 100:.2f}%")
