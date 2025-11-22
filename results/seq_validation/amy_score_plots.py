#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amy_score_plots.py
Parse Waltz amyloid propensity output and generate basic statistics & plots.
Source file assumed: results/seq_validation/WaltzJob_cutoff_50to100/WaltzJob_1763851079.txt
Each block:
>seq_id
Positions\tSequence\tAverage score per residue
start-end\tSEQUENCE\tSCORE
Potentially multiple position/sequence/score lines under one seq id (take all).
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional plot style if available
try:
    import sys
    sys.path.append(os.path.expanduser("~/workspace/amyAMP"))
    from scripts import plotStyle
    plotStyle.setPlotStyle()
except Exception:
    pass

PATH_RESULT = os.path.expanduser("~/workspace/amyAMP/results/")
WALTZ_FILE = os.path.join(PATH_RESULT, "seq_validation/WaltzJob_cutoff_50to100/WaltzJob_1763851079.txt")
AMYLOGRAM_DIR = os.path.join(PATH_RESULT, "seq_validation/amylogram")

HEADER_PATTERN = re.compile(r"^Positions\tSequence\tAverage score per residue$")
DATA_LINE_PATTERN = re.compile(r"^(\S+)\t([A-Z]+)\t([0-9]*\.?[0-9]+)")
SEQ_ID_PATTERN = re.compile(r"^>(\S+)")


def parse_waltz_file(path: str) -> pd.DataFrame:
    """Parse waltz output into DataFrame with columns:
    seq_id, positions, sequence, score
    """
    rows = []
    current_id = None
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m_id = SEQ_ID_PATTERN.match(line)
            if m_id:
                current_id = m_id.group(1)
                continue
            if HEADER_PATTERN.match(line):
                # header line inside block; skip
                continue
            m_data = DATA_LINE_PATTERN.match(line)
            if m_data and current_id is not None:
                positions, sequence, score_str = m_data.groups()
                try:
                    score = float(score_str)
                except ValueError:
                    continue
                rows.append({
                    'seq_id': current_id,
                    'positions': positions,
                    'sequence': sequence,
                    'score': score,
                    'length': len(sequence)
                })
    if not rows:
        raise ValueError("No data parsed from Waltz file.")
    return pd.DataFrame(rows)


def categorize_scores(scores: pd.Series) -> pd.Series:
    """Categorize scores into bins."""
    bins = [0, 50, 60, 70, np.inf]
    labels = ["<50 Low", "50-60 Moderate", "60-70 High", ">=70 Very High"]
    return pd.cut(scores, bins=bins, labels=labels, include_lowest=True, right=False)


def parse_amylogram_dir(dir_path: str) -> pd.DataFrame:
    """Parse all AmyloGram*.csv files, collecting amyloid probability.
    Returns DataFrame with columns: input_name, probability, is_amyloid (yes/no)
    Expects header: "Input name","Amyloid probability","Is amyloid?"
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"AmyloGram directory not found: {dir_path}")
    rows = []
    for fname in sorted(os.listdir(dir_path)):
        if not fname.startswith("AmyloGram") or not fname.endswith(".csv"):
            continue
        fpath = os.path.join(dir_path, fname)
        try:
            df_tmp = pd.read_csv(fpath)
        except Exception as e:
            print(f"Warning: failed to read {fname}: {e}")
            continue
        # Normalize expected columns
        cols_lower = {c.lower(): c for c in df_tmp.columns}
        name_col = cols_lower.get("input name")
        prob_col = cols_lower.get("amyloid probability")
        amy_col = cols_lower.get("is amyloid?")
        if not (name_col and prob_col and amy_col):
            print(f"Warning: missing expected columns in {fname}")
            continue
        df_sel = df_tmp[[name_col, prob_col, amy_col]].copy()
        df_sel.columns = ["input_name", "probability", "is_amyloid"]
        # Clean probability
        df_sel['probability'] = pd.to_numeric(df_sel['probability'], errors='coerce')
        df_sel = df_sel.dropna(subset=['probability'])
        rows.append(df_sel)
    if not rows:
        raise ValueError("No AmyloGram data parsed.")
    return pd.concat(rows, ignore_index=True)


def categorize_probabilities(probs: pd.Series) -> pd.Series:
    """Categorize probabilities into bins comparable to Waltz bins (scaled)."""
    bins = [0, 0.5, 0.6, 0.7, np.inf]
    labels = ["<0.5 Low", "0.5-0.6 Moderate", "0.6-0.7 High", ">=0.7 Very High"]
    return pd.cut(probs, bins=bins, labels=labels, include_lowest=True, right=False)


def main():
    df = parse_waltz_file(WALTZ_FILE)
    # Parse AmyloGram probabilities
    try:
        df_amy = parse_amylogram_dir(AMYLOGRAM_DIR)
    except Exception as e:
        print(f"AmyloGram parsing failed: {e}")
        df_amy = None

    # Basic stats
    mean_score = df['score'].mean()
    median_score = df['score'].median()
    std_score = df['score'].std()
    min_score = df['score'].min()
    max_score = df['score'].max()

    print("=== Amyloid (Waltz) Score Statistics ===")
    print(f"Count: {len(df)}")
    print(f"Mean: {mean_score:.4f}")
    print(f"Median: {median_score:.4f}")
    print(f"Std Dev: {std_score:.4f}")
    print(f"Min: {min_score:.4f}")
    print(f"Max: {max_score:.4f}")

    # Categorize
    df['Category'] = categorize_scores(df['score'])
    category_counts = df['Category'].value_counts().reindex(["<50 Low", "50-60 Moderate", "60-70 High", ">=70 Very High"])  # preserve order

    # Prepare AmyloGram stats if available
    if df_amy is not None:
        mean_prob = df_amy['probability'].mean()
        median_prob = df_amy['probability'].median()
        std_prob = df_amy['probability'].std()
        min_prob = df_amy['probability'].min()
        max_prob = df_amy['probability'].max()
        df_amy['Category'] = categorize_probabilities(df_amy['probability'])
        prob_category_counts = df_amy['Category'].value_counts().reindex(["<0.5 Low", "0.5-0.6 Moderate", "0.6-0.7 High", ">=0.7 Very High"])

        print("\n=== AmyloGram Probability Statistics ===")
        print(f"Count: {len(df_amy)}")
        print(f"Mean: {mean_prob:.4f}")
        print(f"Median: {median_prob:.4f}")
        print(f"Std Dev: {std_prob:.4f}")
        print(f"Min: {min_prob:.4f}")
        print(f"Max: {max_prob:.4f}")
    else:
        mean_prob = median_prob = std_prob = min_prob = max_prob = None
        prob_category_counts = None

    # Plot layout: if AmyloGram available -> 2x2, else 1x2
    if df_amy is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # unified figsize
        ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # widen for consistency
        ax1, ax2 = axes[0], axes[1]

    # Histogram
    # Waltz histogram
    ax1.hist(df['score'], bins=20, color='#4ECDC4', edgecolor='black', alpha=0.75)
    ax1.axvline(mean_score, color='red', linestyle='--', linewidth=1.8, label=f'Mean {mean_score:.2f}')
    ax1.axvline(median_score, color='orange', linestyle='--', linewidth=1.8, label=f'Median {median_score:.2f}')
    ax1.set_title('(a) Waltz Score Distribution', fontsize=14, fontweight='bold', loc='left')
    ax1.set_xlabel('Average score per residue', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Pie chart
    # Waltz pie chart
    ax2 = ax2
    colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF']
    ax2.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('(b) Waltz Score Categories', fontsize=14, fontweight='bold', loc='left')

    # AmyloGram plots if available
    if df_amy is not None:
        # Histogram of probabilities
        ax3.hist(df_amy['probability'], bins=20, color='#FF6B6B', edgecolor='black', alpha=0.75)
        ax3.axvline(mean_prob, color='red', linestyle='--', linewidth=1.5, label=f'Mean {mean_prob:.2f}')
        ax3.axvline(median_prob, color='orange', linestyle='--', linewidth=1.5, label=f'Median {median_prob:.2f}')
        ax3.set_title('(c) AmyloGram Score Distribution', fontsize=14, fontweight='bold', loc='left')
        ax3.set_xlabel('Amyloid probability', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.legend()
        ax3.grid(alpha=0.3)

        # Pie chart of probability categories
        colors_prob = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF']
        ax4.pie(prob_category_counts, labels=prob_category_counts.index, autopct='%1.1f%%', startangle=90, colors=colors_prob)
        ax4.set_title('(d) AmyloGram Score Categories', fontsize=14, fontweight='bold', loc='left')

    plt.tight_layout()
    out_path = os.path.join(PATH_RESULT, 'seq_validation', 'amy_score_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Additional summary
    print("\n=== Category Counts ===")
    print(category_counts)
    for thr in [50, 60, 70]:
        pct = (df['score'] >= thr).sum() / len(df) * 100
        print(f"Percentage with score >= {thr}: {pct:.2f}%")

    if df_amy is not None:
        print("\n=== AmyloGram Probability Category Counts ===")
        print(prob_category_counts)
        for thr in [0.5, 0.6, 0.7]:
            pct_prob = (df_amy['probability'] >= thr).sum() / len(df_amy) * 100
            print(f"Percentage with probability >= {thr}: {pct_prob:.2f}%")

        # Save amylogram parsed data
        df_amy_out = os.path.join(PATH_RESULT, 'seq_validation', 'amylogram_probabilities_parsed.csv')
        df_amy.to_csv(df_amy_out, index=False)
        print(f"AmyloGram parsed data saved to: {df_amy_out}")

    # Optional: save parsed data
    df_out = os.path.join(PATH_RESULT, 'seq_validation', 'amy_scores_parsed.csv')
    df.to_csv(df_out, index=False)
    print(f"Parsed data saved to: {df_out}")

if __name__ == '__main__':
    main()
