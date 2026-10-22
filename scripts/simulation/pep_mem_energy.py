#!/usr/bin/env python3
"""Peptide–membrane interaction energy / depth plots (manuscript Figure 5 panel source).

Historical location (Vijay ``aug_24``)::

  .../vijay/aug_24/pep_mem_energy.py
  .../vijay/aug_24/AMP_Data/energy_trajectory_plots.png   # Fig. 5 composite source

Reads
  - ODS summaries: GenSeq.ods, LowMIC.ods, InterMIC.ods, HighMIC.ods
  - Per-run XVG:   AMP_Data/trajectory_data/{1..100}/prot_mem_interaction.xvg
  - Per-run depth: AMP_Data/depth_data/GenSeq/{1..100}/depth.csv
                   AMP_Data/depth_data/{LDB,IDB,HDB}/{cat}{1..49}/depth.csv

Writes (into ``--output-dir``, default = AMP_Data)::
  energy_trajectory_plots.png   # (a) depth, (b) interaction energy, (c) violin
  depth_by_category.png         # Low / Intermediate / High MIC depth panels

Depth CSVs are produced upstream by ``pap_insertion_cal.py`` (same aug_24 folder).
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
import ezodf
import os
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.append(os.path.expanduser("~/workspace/amyAMP"))
try:
    from scripts import util, plotStyle  # noqa: F401

    plotStyle.setPlotStyle()
except Exception:
    pass

# ──────────── Constants ────────────────────────────────────────────────
DEFAULT_AMP_DATA = Path(
    "/media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/"
    "W_BiGAN_GP_monash/model_saved_231206/vijay/aug_24/AMP_Data"
)
COLORS = {"GenSeq": "#009E73", "LDB": "#0072B2", "IDB": "#E69F00", "HDB": "#D55E00"}
DPI = 600


def _ods_paths(base: Path) -> dict[str, Path]:
    return {
        "GenSeq": base / "GenSeq.ods",
        "LDB": base / "LowMIC.ods",
        "IDB": base / "InterMIC.ods",
        "HDB": base / "HighMIC.ods",
    }

# ──────────── Utility Functions ────────────────────────────────────────
def read_ods(path: Path) -> pd.DataFrame:
    """Load .ods file using ezodf."""
    try:
        doc = ezodf.opendoc(str(path))
        if not doc.sheets:
            raise ValueError(f"No sheets found in the .ods file: {path}")
        
        sheet = doc.sheets[0]
        data = []
        for row in sheet.rows():
            values = []
            for cell in row:
                if cell.value_type == "float":
                    values.append(cell.value)
                elif cell.value_type == "string":
                    values.append(cell.value.strip())
                else:
                    values.append("")
            data.append(values)
        
        if not data:
            raise ValueError(f"No data found in the .ods file: {path}")
        
        header, *rows = data
        if not header or all(h == "" for h in header):
            raise ValueError(f"Invalid or empty header in the .ods file: {path}")
        
        rows = [row[:len(header)] for row in rows]
        return pd.DataFrame(rows, columns=header)
    except Exception as e:
        raise ValueError(f"Failed to read ODS file at {path}: {e}")

def find_col(df: pd.DataFrame, keys: set[str]) -> str:
    """Find the first column matching the given keys."""
    for col in df.columns:
        if str(col).strip().lower() in keys:
            return col
    raise KeyError(f"No column matches {keys} in {list(df.columns)}")

def read_xvg_file(file_path: Path) -> pd.DataFrame:
    """Read XVG file and return DataFrame with 3 columns."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('@') or not line:
                continue
            try:
                values = [float(x) for x in line.split()]
                if len(values) >= 3:
                    data.append(values[:3])
            except ValueError:
                continue
    
    if not data:
        raise ValueError(f"No valid data found in {file_path}")
    
    return pd.DataFrame(data, columns=['col1', 'col2', 'col3'])

def read_depth_csv(file_path: Path) -> np.ndarray:
    """Read depth.csv (use 2nd column if available; otherwise fallback)."""
    try:
        try:
            data = pd.read_csv(file_path)
            numeric_cols = []
            for col in data.columns:
                series = pd.to_numeric(data[col], errors="coerce")
                if series.notna().sum() > 0:
                    numeric_cols.append(series)
            if not numeric_cols:
                raise ValueError("No numeric column found")
            series = numeric_cols[1] if len(numeric_cols) >= 2 else numeric_cols[0]
            return series.dropna().to_numpy()
        except Exception:
            data = pd.read_csv(file_path, header=None)
            col_idx = 1 if data.shape[1] >= 2 else 0
            series = pd.to_numeric(data.iloc[:, col_idx], errors="coerce").dropna()
            return series.to_numpy()
    except Exception as e:
        raise ValueError(f"Failed to read depth CSV at {file_path}: {e}")

def load_trajectory_data(base_path: Path) -> dict:
    """Load trajectory data from 100 directories."""
    trajectory_dir = base_path / "trajectory_data"
    trajectory_data = {}
    
    for i in range(1, 101):
        dir_path = trajectory_dir / str(i)
        file_path = dir_path / "prot_mem_interaction.xvg"
        
        if file_path.exists():
            try:
                df = read_xvg_file(file_path)
                trajectory_data[i] = df['col3'].values
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    return trajectory_data

def load_depth_data(base_path: Path) -> dict[int, np.ndarray]:
    """Load depth insertion data from depth_data/GenSeq/{1..100}/depth.csv."""
    depth_dir = base_path / "depth_data/GenSeq"
    depth_data: dict[int, np.ndarray] = {}
    for i in range(1, 101):
        file_path = depth_dir / str(i) / "depth.csv"
        if file_path.exists():
            try:
                arr = read_depth_csv(file_path)
                depth_data[i] = arr
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        else:
            print(f"Warning: File not found: {file_path}")
    return depth_data

def load_depth_data_by_category(base_path: Path, category: str) -> dict[int, np.ndarray]:
    """Load depth insertion data from depth_data/{category}/{category}{1..49}/depth.csv."""
    depth_dir = base_path / "depth_data" / category
    depth_data: dict[int, np.ndarray] = {}
    for i in range(1, 50):
        file_path = depth_dir / f"{category}{i}" / "depth.csv"
        if file_path.exists():
            try:
                arr = read_depth_csv(file_path)
                depth_data[i] = arr
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        else:
            print(f"Warning: File not found: {file_path}")
    return depth_data

def load_series(ods_paths: dict) -> dict[str, np.ndarray]:
    """Load Total_Mean arrays from ODS files."""
    series = {}
    for name, path in ods_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        df = read_ods(path)
        col = find_col(df, {"total_mean", "total", "total_kj", "total_kj/mol"})
        series[name] = pd.to_numeric(df[col], errors="coerce").dropna().values
    return series

# ──────────── Plotting Functions ───────────────────────────────────────
def plot_violin(ax, df_long, colors):
    """Plot violin plot for Mean energy with statistical annotations."""
    order = ["GenSeq", "LDB", "IDB", "HDB"]
    labels = ["amyAMP", "Low\nMIC", "Intermediate\nMIC", "High\nMIC"]
    
    dataset_data = [df_long[df_long['Category'] == cat]['Total_kJ'].values for cat in order]
    
    parts = ax.violinplot(dataset_data, positions=range(1, len(order) + 1), 
                         showmeans=True, showmedians=False, widths=0.5)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[order[i]])
        pc.set_alpha(0.7)
        pc.set_edgecolor('none')
        pc.set_linewidth(0)
    
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    
    for partname in ('cbars', 'cmins', 'cmaxes'):
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(1.5)
    
    for i, cat in enumerate(order):
        data = df_long[df_long['Category'] == cat]['Total_kJ'].values
        if len(data) > 0:
            mean_val = np.mean(data)
            std_val = np.std(data)
            y_pos = max(data) + (max(data) - min(data)) * 0.05
            ax.text(i + 1, y_pos, 
                    f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
                    ha='center', va='bottom', fontsize=5, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                             edgecolor=colors[cat], linewidth=0.5, alpha=0.9))
    
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=6)
    ax.set_xlabel("", fontsize=6)
    ax.set_ylabel("Mean energy (kJ/mol)", fontsize=6)
    ax.tick_params(axis='y', rotation=45, labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

def plot_trajectory_data(ax, trajectory_data):
    """Plot trajectory data for all directories with smoothing."""
    from scipy.ndimage import uniform_filter1d
    
    for dir_num, data in trajectory_data.items():
        x_axis = np.arange(len(data))
        smoothed_data = uniform_filter1d(data, size=10, mode='nearest')
        
        if dir_num == 22:
            ax.plot(x_axis, smoothed_data, color="red", alpha=0.8, linewidth=0.8, zorder=2)
            annotation_points = [0, 5, 750]
            for time_point in annotation_points:
                if time_point < len(smoothed_data):
                    x_pos = time_point
                    y_pos = smoothed_data[time_point]
                    ax.plot(x_pos, y_pos, 'bo', markersize=3, zorder=3)
                    ax.annotate(f'({int(x_pos)}, {y_pos:.1f})', 
                               xy=(x_pos, y_pos), 
                               xytext=(8, 8), 
                               textcoords='offset points',
                               fontsize=5,
                               color='blue',
                               fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", 
                                       facecolor='white', 
                                       edgecolor='blue', 
                                       linewidth=0.5,
                                       alpha=0.8),
                               arrowprops=dict(arrowstyle='->', 
                                             connectionstyle='arc3,rad=0',
                                             color='blue',
                                             alpha=0.7,
                                             lw=0.5))
        else:
            ax.plot(x_axis, smoothed_data, color="gray", alpha=0.4, linewidth=0.3, zorder=1)
    
    ax.set_xlabel("Time (ns)", fontsize=6)
    ax.set_ylabel("Interaction energy (kJ/mol)", fontsize=6)
    ax.tick_params(axis='y', rotation=45, labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlim(0, 1000)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def _correct_pbc_depth(depth_A: np.ndarray, min_insertion_A: float = -7.0) -> np.ndarray:
    """
    Segment-wise PBC correction for depth (Å):
    - Identify contiguous runs where values are negative.
    - If a run contains any value < min_insertion_A (e.g., -7 Å), flip the whole run to positive (abs).
    - Otherwise keep the run as-is (true shallow insertions stay negative).
    """
    vals = np.asarray(depth_A, dtype=float)
    if vals.size == 0:
        return vals

    corrected = vals.copy()
    is_neg = vals < 0.0

    i = 0
    n = len(vals)
    while i < n:
        if not is_neg[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and is_neg[i + 1]:
            i += 1
        end = i
        segment = vals[start:end + 1]
        if np.any(segment < min_insertion_A):
            corrected[start:end + 1] = np.abs(segment)
        i += 1

    return corrected

def plot_depth_data(ax, depth_data: dict[int, np.ndarray]):
    """Plot depth insertion; apply running average; plot PBC-corrected Å values."""
    from scipy.ndimage import uniform_filter1d
    for dir_num, data in depth_data.items():
        x_axis = np.arange(len(data))
        raw_A = np.asarray(data, dtype=float)
        smoothed = uniform_filter1d(raw_A, size=10, mode="nearest") if len(raw_A) >= 10 else raw_A
        corr_A = _correct_pbc_depth(smoothed, min_insertion_A=-7.0)

        if dir_num == 22:
            ax.plot(x_axis, corr_A, color="red", alpha=0.8, linewidth=0.8, zorder=3, label="Peptide 22")
            annotation_points = [0, 5, 750]
            for time_point in annotation_points:
                if time_point < len(corr_A):
                    x_pos = time_point
                    y_pos_A = corr_A[time_point]
                    ax.plot(x_pos, y_pos_A, 'bo', markersize=3, zorder=4)
                    ax.annotate(f'({int(x_pos)}, {y_pos_A:.2f})',
                                xy=(x_pos, y_pos_A),
                                xytext=(8, 8),
                                textcoords='offset points',
                                fontsize=5,
                                color='blue',
                                fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                          edgecolor='blue', linewidth=0.5, alpha=0.8),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                                color='blue', alpha=0.7, lw=0.5))
        else:
            ax.plot(x_axis, corr_A, color="gray", alpha=0.4, linewidth=0.3, zorder=1)

        # Add horizontal dashed line at y=0
        ax.axhline(y=0, color='cyan', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

    ax.set_xlabel("Time (ns)", fontsize=6)
    ax.set_ylabel("Peptide depth in membrane (Å)", fontsize=6)
    ax.tick_params(axis='y', rotation=45, labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlim(0, 1000)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc="upper right", fontsize=6, frameon=False)

def plot_depth_data_category(ax, depth_data: dict[int, np.ndarray], color: str, title: str):
    """Plot depth insertion for a category with a single color."""
    from scipy.ndimage import uniform_filter1d
    for dir_num, data in depth_data.items():
        x_axis = np.arange(len(data))
        raw_A = np.asarray(data, dtype=float)
        smoothed = uniform_filter1d(raw_A, size=10, mode="nearest") if len(raw_A) >= 10 else raw_A
        corr_A = _correct_pbc_depth(smoothed, min_insertion_A=-7.0)
        ax.plot(x_axis, corr_A, color=color, alpha=0.4, linewidth=0.3, zorder=1)

    # Add horizontal dashed line at y=0
    ax.axhline(y=0, color='purple', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

    ax.set_xlabel("Time (ns)", fontsize=6)
    ax.set_ylabel("Peptide depth in membrane (Å)", fontsize=6)
    ax.set_title(title, fontsize=7, fontweight='bold', loc='left')
    ax.tick_params(axis='y', rotation=45, labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlim(0, 1000)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# ──────────── Main Script ──────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Replot Figure 5–style peptide–membrane energy/depth panels."
    )
    p.add_argument(
        "--amp-data-dir",
        type=Path,
        default=DEFAULT_AMP_DATA,
        help="AMP_Data root (ODS + trajectory_data/ + depth_data/)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as --amp-data-dir)",
    )
    p.add_argument("--dpi", type=int, default=DPI)
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    args = p.parse_args(argv)

    base: Path = args.amp_data_dir
    output_dir: Path = args.output_dir or base
    output_dir.mkdir(parents=True, exist_ok=True)
    ods = _ods_paths(base)
    dpi = args.dpi

    # Load data
    series = load_series(ods)

    # Prepare data for violin plot
    records = []
    for cat, path in ods.items():
        df = read_ods(path)
        col = find_col(df, {"total_mean", "total", "total_kj", "total_kj/mol"})
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        records.extend({"Category": cat, "Total_kJ": v} for v in vals)
    df_long = pd.DataFrame.from_records(records)

    # Load trajectory and depth data
    trajectory_data = load_trajectory_data(base)
    print(f"Loaded trajectory data from {len(trajectory_data)} directories")

    depth_data = load_depth_data(base)
    print(f"Loaded depth data from {len(depth_data)} directories")

    # Create figure with 3 subplots (main figure)
    fig, axs = plt.subplots(1, 3, figsize=(6, 2), dpi=dpi, constrained_layout=True)

    # Subplot (a): depth insertion
    plot_depth_data(axs[0], depth_data)

    # Subplot (b): trajectory energy
    plot_trajectory_data(axs[1], trajectory_data)

    # Subplot (c): violin plot
    plot_violin(axs[2], df_long, COLORS)

    # Save and show main figure
    fig.tight_layout()
    out_main = output_dir / "energy_trajectory_plots.png"
    fig.savefig(out_main, dpi=dpi)
    print(f"Wrote {out_main}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    # ──────────── Additional Figure: Depth by Category ────────────────
    ldb_depth = load_depth_data_by_category(base, "LDB")
    idb_depth = load_depth_data_by_category(base, "IDB")
    hdb_depth = load_depth_data_by_category(base, "HDB")

    print(f"Loaded LDB depth data from {len(ldb_depth)} directories")
    print(f"Loaded IDB depth data from {len(idb_depth)} directories")
    print(f"Loaded HDB depth data from {len(hdb_depth)} directories")

    fig2, axs2 = plt.subplots(1, 3, figsize=(6, 2), dpi=dpi, constrained_layout=True)
    plot_depth_data_category(axs2[0], ldb_depth, COLORS["LDB"], "(a) Low MIC")
    plot_depth_data_category(axs2[1], idb_depth, COLORS["IDB"], "(b) Intermediate MIC")
    plot_depth_data_category(axs2[2], hdb_depth, COLORS["HDB"], "(c) High MIC")

    fig2.tight_layout()
    out_cat = output_dir / "depth_by_category.png"
    fig2.savefig(out_cat, dpi=dpi)
    print(f"Wrote {out_cat}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig2)


if __name__ == "__main__":
    main()


