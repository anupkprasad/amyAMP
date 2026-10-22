#!/usr/bin/env python3
"""Plot oligomer-size time series and early vs late size distributions.

Modular replacement for the historical ``aggre_plot.py`` used for Figure 6.
Reads precomputed ``analysisoligo-highest-size.dat`` (or ``oligo-highest-size.dat``)
files produced by ``oligomer_analysis.aggregate_analysis``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Allow running as a script from repo root or this folder
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.simulation.oligomer_analysis import (  # noqa: E402
    classify_high_aggregation,
    load_highest_size_library,
)

try:
    from scripts import plotStyle  # noqa: E402

    plotStyle.setPlotStyle()
except Exception:
    pass


def plot_oligomer_size_evolution(
    data_dir: str | Path,
    out_png: str | Path,
    n_frames: int = 1001,
    exclude: Optional[Sequence[int]] = None,
    highlight_id: Optional[int] = 22,
    early_frames: int = 100,
    late_frames: int = 100,
    max_mer: int = 8,
    dpi: int = 600,
) -> Path:
    """Make a 1×2 figure: (a) size vs time, (b) early vs late size fractions."""
    exclude = list(exclude or [])
    all_data, used_ids, _ = load_highest_size_library(
        data_dir, n_frames=n_frames, exclude=exclude
    )
    time_points = np.arange(n_frames)
    oligomer_sizes = list(range(1, max_mer + 1))

    early = all_data[:early_frames, used_ids].ravel()
    late = all_data[-late_frames:, used_ids].ravel()
    early_counts = [int(np.count_nonzero(early == s)) for s in oligomer_sizes]
    late_counts = [int(np.count_nonzero(late == s)) for s in oligomer_sizes]
    early_tot = sum(early_counts) or 1
    late_tot = sum(late_counts) or 1
    early_frac = [c / early_tot for c in early_counts]
    late_frac = [c / late_tot for c in late_counts]

    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)

    for i in used_ids:
        if highlight_id is not None and i == highlight_id:
            axs[0].plot(
                time_points,
                all_data[:, i],
                color="red",
                alpha=0.8,
                linewidth=1.6,
                zorder=2,
            )
            pts = [0, min(60, n_frames - 1), n_frames - 1]
            axs[0].scatter(
                pts,
                [all_data[t, i] for t in pts],
                color="blue",
                s=18,
                zorder=3,
            )
        else:
            axs[0].plot(
                time_points,
                all_data[:, i],
                color="gray",
                alpha=0.4,
                linewidth=0.4,
                zorder=1,
            )

    axs[0].set_xlabel("Time (ns)")
    axs[0].set_ylabel("Oligomer Size")
    axs[0].set_xlim(0, None)
    axs[0].set_ylim(0.8, max_mer + 0.2)
    axs[0].grid(True, alpha=0.3)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)

    x_pos = np.arange(len(oligomer_sizes))
    bar_width = 0.35
    axs[1].bar(
        x_pos - bar_width / 2,
        early_frac,
        bar_width,
        label=f"First {early_frames} ns",
        color="lightblue",
        alpha=0.8,
        edgecolor="black",
    )
    axs[1].bar(
        x_pos + bar_width / 2,
        late_frac,
        bar_width,
        label=f"Last {late_frames} ns",
        color="lightcoral",
        alpha=0.8,
        edgecolor="black",
    )
    axs[1].set_xlabel("Oligomer Size")
    axs[1].set_ylabel("Fraction")
    axs[1].set_xticks(x_pos)
    axs[1].set_xticklabels([f"{s}-mer" for s in oligomer_sizes], rotation=30)
    axs[1].legend(fontsize=7)
    axs[1].grid(True, alpha=0.3, axis="y")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Trajectories plotted: {len(used_ids)}")
    print(f"Early distribution: {dict(zip(oligomer_sizes, early_counts))}")
    print(f"Late distribution:  {dict(zip(oligomer_sizes, late_counts))}")
    print(f"Saved {out_png.resolve()}")
    return out_png


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Plot oligomer size evolution (Figure 6 style).")
    p.add_argument(
        "-d",
        "--data-dir",
        required=True,
        help="Root directory with <id>/analysisoligo-highest-size.dat",
    )
    p.add_argument(
        "-o",
        "--output",
        default="oligomer_size_evolution.png",
        help="Output PNG path",
    )
    p.add_argument("--n-frames", type=int, default=1001)
    p.add_argument("--exclude", type=int, nargs="*", default=[24, 59, 84, 97])
    p.add_argument("--highlight-id", type=int, default=22)
    p.add_argument("--no-highlight", action="store_true")
    p.add_argument(
        "--write-indices",
        default="",
        help="Optional path to write aggregate-size≥5 classification indices",
    )
    p.add_argument("--size-threshold", type=int, default=5)
    p.add_argument("--tail-frames", type=int, default=200)
    p.add_argument("--occupancy", type=float, default=0.5)
    args = p.parse_args(argv)

    highlight = None if args.no_highlight else args.highlight_id
    plot_oligomer_size_evolution(
        data_dir=args.data_dir,
        out_png=args.output,
        n_frames=args.n_frames,
        exclude=args.exclude,
        highlight_id=highlight,
    )

    if args.write_indices:
        all_data, used, _ = load_highest_size_library(
            args.data_dir, n_frames=args.n_frames, exclude=args.exclude
        )
        selected = classify_high_aggregation(
            all_data,
            used,
            size_threshold=args.size_threshold,
            tail_frames=args.tail_frames,
            occupancy=args.occupancy,
        )
        Path(args.write_indices).write_text(
            "\n".join(str(i) for i in selected) + ("\n" if selected else "")
        )
        print(f"Wrote {len(selected)} indices → {args.write_indices}")


if __name__ == "__main__":
    main()
