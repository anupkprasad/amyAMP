#!/usr/bin/env python3
"""
Figure 4 replacement: AMPScanner / AmyloGram scores across datasets.

Addresses reviewer concern that WALTZ / AI4AMP are training-selection tools.
This figure rests validation on independent predictors:
  - AMPScanner v2  (AMP)
  - AmyloGram      (amyloid)
and scores amyAMP, trainPep, AMP(D), AMY(D), randPep as baselines.

Outputs (under results/):
  fig4_heldout_validators.png / .pdf
  fig4_heldout_validators_summary.csv
  fig4_heldout_validators_scores.csv  (long-format scores used in the plot)
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

PATH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_ROOT not in sys.path:
    sys.path.append(PATH_ROOT)

from scripts import plotStyle

plotStyle.setPlotStyle()

PATH_RESULT = os.path.join(PATH_ROOT, "results")
PATH_SEQ = os.path.join(PATH_RESULT, "sequence")
PATH_AMPSCAN = os.path.join(PATH_SEQ, "AMPScanner_vr.2")
PATH_AMYLOGRAM = os.path.join(PATH_SEQ, "amylogram")
PATH_GENSEQ = os.path.join(PATH_RESULT, "seq_validation", "GenSeq.ods")
PATH_EXPERIMENTAL = os.path.join(
    PATH_ROOT, "data_master", "dual_amyAMP_experimental", "dual_amyAMP_experimental.fasta"
)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
COLOR_GENERATED = "#2ECC40"
COLOR_SIMULATED = "#B10DC9"
COLOR_EXPERIMENTAL = "#1A5276"  # deep navy — high contrast vs green/purple/yellow
COLOR_RANDPEP = "#FFD60A"  # match Fig. 3 randPep

# Display order / colors (match Fig. 3 property plots)
DATASETS = [
    ("amyAMP", "amyAMP", "#2ECC40"),
    ("trainPep", "trainPep", "#B10DC9"),
    ("AMP(D)", "AMPD", "#FF6B6B"),
    ("AMY(D)", "AMYD", "#4DA6FF"),
    ("randPep", "randPep", "#FFD60A"),
]
LABELS = [d[0] for d in DATASETS]
COLORS = {d[0]: d[2] for d in DATASETS}
SUBDIRS = {d[0]: d[1] for d in DATASETS}

THRESHOLD = 0.5  # probability cutoff for "positive" fraction bars


def load_ampscanner(label: str) -> pd.DataFrame:
    sub = SUBDIRS[label]
    matches = glob.glob(os.path.join(PATH_AMPSCAN, sub, "*Prediction_Summary.csv"))
    if not matches:
        raise FileNotFoundError(f"AMPScanner summary missing for {label} under {PATH_AMPSCAN}/{sub}")
    df = pd.read_csv(matches[0])
    df = df.rename(columns={"SeqID": "seq_id", "Prediction_Probability": "score", "Sequence": "sequence"})
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).copy()
    df["dataset"] = label
    df["predictor"] = "AMPScanner"
    return df[["dataset", "predictor", "seq_id", "score", "sequence"]]


def load_amylogram(label: str) -> pd.DataFrame:
    sub = SUBDIRS[label]
    matches = glob.glob(os.path.join(PATH_AMYLOGRAM, sub, "*_AmyloGram_scores.csv"))
    if not matches:
        raise FileNotFoundError(f"AmyloGram scores missing for {label} under {PATH_AMYLOGRAM}/{sub}")
    df = pd.read_csv(matches[0])
    df = df.rename(columns={"Name": "seq_id", "Probability": "score"})
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).copy()
    df["dataset"] = label
    df["predictor"] = "AmyloGram"
    df["sequence"] = np.nan
    return df[["dataset", "predictor", "seq_id", "score", "sequence"]]


def load_all_scores() -> pd.DataFrame:
    frames = []
    for label, _, _ in DATASETS:
        frames.append(load_ampscanner(label))
        frames.append(load_amylogram(label))
    return pd.concat(frames, ignore_index=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, _, _ in DATASETS:
        for pred in ("AMPScanner", "AmyloGram"):
            s = df[(df["dataset"] == label) & (df["predictor"] == pred)]["score"]
            rows.append(
                {
                    "dataset": label,
                    "predictor": pred,
                    "n": int(s.shape[0]),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                    f"frac_ge_{THRESHOLD}": float((s >= THRESHOLD).mean()),
                }
            )
    return pd.DataFrame(rows)


def _violin_panel(ax, df: pd.DataFrame, predictor: str, letter: str, title: str, show_n_mu: bool = False, show_mean: bool = True):
    data = [df[(df["dataset"] == lab) & (df["predictor"] == predictor)]["score"].values for lab in LABELS]
    positions = np.arange(1, len(LABELS) + 1)

    parts = ax.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.85,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[LABELS[i]])
        pc.set_edgecolor("0.25")
        pc.set_alpha(0.75)
        pc.set_linewidth(0.6)

    # Overlay boxplots for quartiles / median (thin sticks)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.12,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="0.2", linewidth=0.9),
        capprops=dict(color="0.2", linewidth=0.9),
        boxprops=dict(facecolor="white", edgecolor="0.2", linewidth=0.9),
    )

    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        if show_mean:
            ax.scatter(
                positions[i],
                np.mean(vals),
                marker="D",
                s=14,
                color="black",
                zorder=5,
            )
        if show_n_mu:
            ax.text(
                positions[i],
                -0.07,
                f"n={len(vals)}\nμ={np.mean(vals):.2f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=5.5,
                color="0.25",
                clip_on=False,
            )

    ax.axhline(THRESHOLD, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Probability", fontsize=9.5)
    ax.set_title(f"({letter}) {title}", fontweight="bold", loc="left", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_xlim(0.4, len(LABELS) + 0.6)
    return bp


def _fraction_bar_panel(ax, summary: pd.DataFrame, letter: str):
    x = np.arange(len(LABELS))
    width = 0.36
    amp_frac = [
        summary[(summary.dataset == lab) & (summary.predictor == "AMPScanner")][f"frac_ge_{THRESHOLD}"].values[0]
        for lab in LABELS
    ]
    amy_frac = [
        summary[(summary.dataset == lab) & (summary.predictor == "AmyloGram")][f"frac_ge_{THRESHOLD}"].values[0]
        for lab in LABELS
    ]

    bars1 = ax.bar(
        x - width / 2,
        amp_frac,
        width,
        color=[COLORS[l] for l in LABELS],
        edgecolor="0.2",
        linewidth=0.6,
        alpha=0.9,
        hatch="///",
        label=f"AMPScanner ≥ {THRESHOLD:g}",
    )
    bars2 = ax.bar(
        x + width / 2,
        amy_frac,
        width,
        color=[COLORS[l] for l in LABELS],
        edgecolor="0.2",
        linewidth=0.6,
        alpha=0.55,
        label=f"AmyloGram ≥ {THRESHOLD:g}",
    )

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.02,
                f"{100 * h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(f"Fraction with score ≥ {THRESHOLD:g}", fontsize=9.5)
    ax.set_title(f"({letter})", fontweight="bold", loc="left", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right")


def _load_simulated_sequences() -> set[str]:
    """MD-selected peptides from GenSeq.ods (same set as prior Fig. 4 scatter)."""
    genseq = pd.read_excel(PATH_GENSEQ, engine="odf", header=0)
    return set(genseq["sequence"].dropna().str.strip())


def _amyamp_joint(df: pd.DataFrame) -> pd.DataFrame:
    """Merge AMPScanner + AmyloGram scores for amyAMP; flag simulated peptides."""
    amp = df[(df.dataset == "amyAMP") & (df.predictor == "AMPScanner")][
        ["seq_id", "score", "sequence"]
    ].rename(columns={"score": "amp"})
    amy = df[(df.dataset == "amyAMP") & (df.predictor == "AmyloGram")][["seq_id", "score"]].rename(
        columns={"score": "amy"}
    )
    m = amp.merge(amy, on="seq_id", how="inner")
    sim_seqs = _load_simulated_sequences()
    m["is_simulated"] = m["sequence"].astype(str).str.strip().isin(sim_seqs)
    return m


def _scatter_amyamp(fig, host_ax, df: pd.DataFrame, letter: str):
    """
    amyAMP AMPScanner vs AmyloGram scatter matching seq_validation/amp_vs_amy_scatter.png:
    green generated points, purple simulated overlay, quadrant labels, top/right count bars.
    """
    m = _amyamp_joint(df)
    df_other = m[~m["is_simulated"]]
    df_sim = m[m["is_simulated"]]
    n_total = len(m)
    n_sim = int(df_sim.shape[0])

    # Zero-gap nested layout so right/top hists sit against scatter spines
    spec = host_ax.get_subplotspec()
    host_ax.remove()
    gs = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=spec,
        width_ratios=[1.0, 0.22],
        height_ratios=[0.22, 1.0],
        wspace=0.0,
        hspace=0.0,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    def quadrant_label(count, total):
        pct = (count / total * 100.0) if total > 0 else 0.0
        return f"n={count}\n{pct:.1f}%"

    ax.scatter(
        df_other["amp"],
        df_other["amy"],
        alpha=0.5,
        s=18,
        c="#2ECC40",
        edgecolors="black",
        linewidth=0.35,
        label=f"Generated (n={n_total})",
        zorder=2,
    )
    ax.scatter(
        df_sim["amp"],
        df_sim["amy"],
        alpha=0.85,
        s=28,
        c="#B10DC9",
        edgecolors="black",
        linewidth=0.5,
        label=f"Simulated (n={n_sim})",
        zorder=5,
    )

    ax.axhline(y=THRESHOLD, color="gray", linestyle="--", alpha=0.35, linewidth=0.9)
    ax.axvline(x=THRESHOLD, color="gray", linestyle="--", alpha=0.35, linewidth=0.9)

    high_amp_high_amy = int(((m["amp"] >= THRESHOLD) & (m["amy"] >= THRESHOLD)).sum())
    high_amp_low_amy = int(((m["amp"] >= THRESHOLD) & (m["amy"] < THRESHOLD)).sum())
    low_amp_high_amy = int(((m["amp"] < THRESHOLD) & (m["amy"] >= THRESHOLD)).sum())
    low_amp_low_amy = int(((m["amp"] < THRESHOLD) & (m["amy"] < THRESHOLD)).sum())

    q_kw = dict(
        transform=ax.transAxes,
        fontsize=8,
        color="gray",
        style="italic",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.95),
        zorder=20,
    )
    ax.text(
        0.25,
        0.95,
        f"High AMY\nLow AMP\n{quadrant_label(low_amp_high_amy, n_total)}",
        ha="center",
        va="top",
        **q_kw,
    )
    ax.text(
        0.75,
        0.95,
        f"High AMY\nHigh AMP\n{quadrant_label(high_amp_high_amy, n_total)}",
        ha="center",
        va="top",
        **q_kw,
    )
    ax.text(
        0.25,
        0.05,
        f"Low AMY\nLow AMP\n{quadrant_label(low_amp_low_amy, n_total)}",
        ha="center",
        va="bottom",
        **q_kw,
    )
    ax.text(
        0.75,
        0.05,
        f"Low AMY\nHigh AMP\n{quadrant_label(high_amp_low_amy, n_total)}",
        ha="center",
        va="bottom",
        **q_kw,
    )

    bins = np.linspace(0, 1, 21)
    hist_color = "#9E9E9E"
    ax_histx.hist(m["amp"], bins=bins, color=hist_color, edgecolor="black", alpha=0.7)
    ax_histy.hist(
        m["amy"],
        bins=bins,
        orientation="horizontal",
        color=hist_color,
        edgecolor="black",
        alpha=0.7,
    )

    ax_histx.set_ylabel("Count", fontsize=8.5)
    ax_histy.set_xlabel("Count", fontsize=8.5, labelpad=1)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.tick_params(axis="y", labelsize=8)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.tick_params(axis="x", labelsize=8)
    ax_histy.xaxis.set_major_locator(plt.MaxNLocator(nbins=3, integer=True, min_n_ticks=2))
    for spine_ax in (ax_histx, ax_histy):
        spine_ax.grid(alpha=0.15, linestyle=":")
        spine_ax.spines["top"].set_visible(False)
        spine_ax.spines["right"].set_visible(False)

    # Hide shared tick labels that would sit in the zero-gap seam
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    ax.set_xlabel("AMPScanner probability", fontsize=9.5)
    ax.set_ylabel("AmyloGram probability", fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.5)

    ax_histx.set_title(f"({letter}) amyAMP", fontweight="bold", loc="left", fontsize=10.5)
    handles, labels = ax.get_legend_handles_labels()
    ax_histx.legend(
        handles,
        labels,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.38, 1.65),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.3,
        columnspacing=0.9,
    )

    print(f"Panel d: amyAMP n={n_total}, simulated={n_sim}")
    return ax


def _read_fasta_sequences(path: str) -> list[str]:
    seqs: list[str] = []
    cur: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur).upper())
                cur = []
            else:
                cur.append(line)
        if cur:
            seqs.append("".join(cur).upper())
    return seqs


def _aa_fractions(sequences: list[str]) -> np.ndarray:
    counts = {aa: 0 for aa in AMINO_ACIDS}
    for seq in sequences:
        for aa in seq:
            if aa in counts:
                counts[aa] += 1
    total = sum(counts.values())
    if total == 0:
        return np.zeros(len(AMINO_ACIDS))
    return np.array([counts[aa] / total for aa in AMINO_ACIDS])


def _aa_frequency_panel(ax, df: pd.DataFrame, letter: str):
    """Panel (e): AA composition — experimental dual; amyAMP/simulated from High-High quadrant; randPep."""
    exp_seqs = _read_fasta_sequences(PATH_EXPERIMENTAL)

    # Restrict amyAMP + simulated to panel-D High AMY & High AMP quadrant (scores ≥ 0.5)
    m = _amyamp_joint(df)
    high_high = m[(m["amp"] >= THRESHOLD) & (m["amy"] >= THRESHOLD)].copy()
    gen_seqs = (
        high_high["sequence"].dropna().astype(str).str.strip().tolist()
    )
    sim_seqs = (
        high_high.loc[high_high["is_simulated"], "sequence"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    rand_seqs = (
        df[(df.dataset == "randPep") & (df.predictor == "AMPScanner")]["sequence"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    groups = [
        ("Experimental", exp_seqs, COLOR_EXPERIMENTAL),
        ("amyAMP (High/High)", gen_seqs, COLOR_GENERATED),
        ("Simulated (High/High)", sim_seqs, COLOR_SIMULATED),
        ("randPep", rand_seqs, COLOR_RANDPEP),
    ]
    fracs = [_aa_fractions(seqs) for _, seqs, _ in groups]

    x = np.arange(len(AMINO_ACIDS))
    n_grp = len(groups)
    width = 0.8 / n_grp
    offsets = (np.arange(n_grp) - (n_grp - 1) / 2.0) * width

    for (label, seqs, color), frac, off in zip(groups, fracs, offsets):
        ax.bar(
            x + off,
            frac,
            width,
            label=f"{label} (n={len(seqs)})",
            color=color,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(AMINO_ACIDS, fontsize=8.5)
    ax.set_xlabel("Amino acid", fontsize=9.5)
    ax.set_ylabel("Fraction", fontsize=9.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title(f"({letter}) Amino acid frequency", fontweight="bold", loc="left", fontsize=10.5)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_ylim(0, max(f.max() for f in fracs) * 1.28)

    print(
        "Panel e: "
        + ", ".join(f"{label}={len(seqs)}" for label, seqs, _ in groups)
    )
    return ax

def plot_figure(df: pd.DataFrame, summary: pd.DataFrame, out_png: str, out_pdf: str):
    # Row 1: (a)(b)(c); Row 2: (d)(e)
    fig = plt.figure(figsize=(12.0, 8.2), dpi=600)
    gs = fig.add_gridspec(
        2,
        6,
        left=0.06,
        right=0.98,
        bottom=0.08,
        top=0.92,
        wspace=0.55,
        hspace=0.38,
    )
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d_host = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    _violin_panel(
        ax_a,
        df,
        "AMPScanner",
        "a",
        "AMPScanner",
        show_n_mu=False,
        show_mean=False,
    )
    _violin_panel(
        ax_b,
        df,
        "AmyloGram",
        "b",
        "AmyloGram",
        show_n_mu=False,
        show_mean=True,
    )
    _fraction_bar_panel(ax_c, summary, "c")
    _scatter_amyamp(fig, ax_d_host, df, "d")
    _aa_frequency_panel(ax_e, df, "e")

    # Do not use tight_layout — it reopens gaps in the nested panel-d gridspec
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def main():
    print("Loading AMPScanner + AmyloGram scores for 5 datasets...")
    df = load_all_scores()
    summary = summary_table(df)
    print(summary.to_string(index=False))

    scores_csv = os.path.join(PATH_RESULT, "fig4_heldout_validators_scores.csv")
    summary_csv = os.path.join(PATH_RESULT, "fig4_heldout_validators_summary.csv")
    out_png = os.path.join(PATH_RESULT, "fig4_heldout_validators.png")
    out_pdf = os.path.join(PATH_RESULT, "fig4_heldout_validators.pdf")

    df.to_csv(scores_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {scores_csv}")
    print(f"Wrote {summary_csv}")

    plot_figure(df, summary, out_png, out_pdf)


if __name__ == "__main__":
    main()
