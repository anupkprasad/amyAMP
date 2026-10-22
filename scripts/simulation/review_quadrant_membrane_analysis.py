#!/usr/bin/env python3
"""Fig. 7–style membrane thickness / curvature for R1Q8 quadrant simulations.

Analyses the stratified membrane-disruption runs deposited under::

  .../oct17/review_sim/CG_Paper_production_data_2026-10-14/

which sample the four Figure 8a classes (Energy ≈ strong membrane interaction /
AMP-like; Aggregate ≈ high oligomer propensity / AMY-like; Both; Neither)
on Gram-negative and Gram-positive bilayers, plus peptide-free Membrane-Only
controls.

For each complete ``step7_production.{tpr,xtc}`` run this script:

  1. Computes last-N-ns thickness and mean-curvature maps
     (``membrane_thickness_curvature.analyze_membrane_properties``).
  2. Writes per-run ``.npz`` heatmaps and a summary CSV.
  3. Builds comparison figures (maps + scalar bars) suitable for the
     reviewer response / SI.

Default figure exemplars (one peptide per quadrant):
  Aggregate→CG94, Both→CG40, Energy→CG24, Neither→CG55.
Replica choice for maps: ``primary`` if present else ``replicate_1``.
Scalar summaries report mean ± s.d. across all available replicates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import MDAnalysis as mda
except ImportError as exc:  # pragma: no cover
    raise ImportError("MDAnalysis is required. pip install MDAnalysis") from exc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

from .membrane_thickness_curvature import (
    DEFAULT_CURVATURE_SMOOTH_SIGMA,
    analyze_membrane_properties,
    smooth_curvature_map,
    write_method_metadata,
)

# Fig. 7 display: signed mean curvature is shown as H × 100 (paper convention)
CURVATURE_DISPLAY_SCALE = 100.0
MAP_SMOOTH_SIGMA = DEFAULT_CURVATURE_SMOOTH_SIGMA  # smooth thickness + curvature maps

DEFAULT_DATA_ROOT = Path(
    "/media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/"
    "W_BiGAN_GP_monash/model_saved_231206/vijay/oct17/review_sim/"
    "CG_Paper_production_data_2026-10-14"
)

# Figure 8a class labels used in folder names → display names
QUADRANT_LABELS = {
    "Energy": "Energy (AMP-like)",
    "Aggregate": "Aggregate (AMY-like)",
    "Both": "Both",
    "Neither": "Neither",
    "Membrane-Only": "Membrane-only control",
}

# Exemplars for map comparison panels (one CG id per Fig. 8a class)
DEFAULT_EXEMPLARS = {
    "Aggregate": "CG94",
    "Both": "CG40",
    "Energy": "CG24",
    "Neither": "CG55",
}

QUADRANT_ORDER = ["Membrane-Only", "Neither", "Energy", "Aggregate", "Both"]
MEMBRANE_ORDER = ["Gram-ve", "Gram+ve"]

# Folder names on disk are sometimes lower-case (e.g. Gram-ve/energy)
QUADRANT_CANON = {
    "energy": "Energy",
    "aggregate": "Aggregate",
    "both": "Both",
    "neither": "Neither",
    "membrane-only": "Membrane-Only",
}


def canonicalize_quadrant(name: str) -> str:
    return QUADRANT_CANON.get(name.strip().lower(), name)


@dataclass
class RunSpec:
    membrane: str  # Gram-ve | Gram+ve | Membrane-Only
    quadrant: str  # Energy | Aggregate | Both | Neither | Membrane-Only
    peptide: str  # CGxx or control
    replicate: str  # primary | replicate_1 | replicate_2
    run_dir: Path
    tpr: Path
    xtc: Path


def discover_runs(data_root: Path) -> List[RunSpec]:
    """Find all complete production runs under the review_sim archive."""
    runs: List[RunSpec] = []
    root = Path(data_root)

    mem_only = root / "Membrane-Only"
    if mem_only.is_dir():
        for run_dir in sorted(mem_only.iterdir()):
            if not run_dir.is_dir():
                continue
            tpr = run_dir / "step7_production.tpr"
            xtc = run_dir / "step7_production.xtc"
            if tpr.is_file() and xtc.is_file():
                runs.append(
                    RunSpec(
                        membrane="Membrane-Only",
                        quadrant="Membrane-Only",
                        peptide="control",
                        replicate=run_dir.name,
                        run_dir=run_dir,
                        tpr=tpr,
                        xtc=xtc,
                    )
                )

    quad = root / "Quadrant"
    if quad.is_dir():
        for membrane_dir in sorted(quad.iterdir()):
            if not membrane_dir.is_dir():
                continue
            membrane = membrane_dir.name
            for quadrant_dir in sorted(membrane_dir.iterdir()):
                if not quadrant_dir.is_dir():
                    continue
                quadrant = canonicalize_quadrant(quadrant_dir.name)
                for pep_dir in sorted(quadrant_dir.iterdir()):
                    if not pep_dir.is_dir():
                        continue
                    peptide = pep_dir.name
                    for run_dir in sorted(pep_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue
                        tpr = run_dir / "step7_production.tpr"
                        xtc = run_dir / "step7_production.xtc"
                        if tpr.is_file() and xtc.is_file():
                            runs.append(
                                RunSpec(
                                    membrane=membrane,
                                    quadrant=quadrant,
                                    peptide=peptide,
                                    replicate=run_dir.name,
                                    run_dir=run_dir,
                                    tpr=tpr,
                                    xtc=xtc,
                                )
                            )
    return runs


def prefer_replicate(replicates: Sequence[str]) -> str:
    """Map figure: primary if present, else replicate_1, else first."""
    if "primary" in replicates:
        return "primary"
    if "replicate_1" in replicates:
        return "replicate_1"
    return sorted(replicates)[0]


def run_key(spec: RunSpec) -> str:
    return f"{spec.membrane}__{spec.quadrant}__{spec.peptide}__{spec.replicate}"


def analyze_one(
    spec: RunSpec,
    out_dir: Path,
    *,
    analysis_time_ns: float,
    time_window: str,
    grid_resolution: int,
    smooth_sigma: float,
    force: bool,
) -> Dict:
    """Run membrane analysis for one trajectory; cache to npz + json."""
    tag = run_key(spec)
    run_out = out_dir / "per_run" / tag
    run_out.mkdir(parents=True, exist_ok=True)
    npz_path = run_out / f"membrane_maps_{time_window}.npz"
    json_path = run_out / f"summary_{time_window}.json"

    if npz_path.is_file() and json_path.is_file() and not force:
        with open(json_path) as fh:
            summary = json.load(fh)
        summary["cached"] = True
        return summary

    u = mda.Universe(str(spec.tpr), str(spec.xtc))
    results, heatmaps = analyze_membrane_properties(
        u,
        grid_resolution=grid_resolution,
        analysis_time_ns=analysis_time_ns,
        time_window=time_window,
        curvature_smooth_sigma=smooth_sigma,
    )
    if results is None or heatmaps is None:
        summary = {
            "tag": tag,
            "membrane": spec.membrane,
            "quadrant": spec.quadrant,
            "peptide": spec.peptide,
            "replicate": spec.replicate,
            "ok": False,
            "error": "leaflet identification failed",
        }
        with open(json_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    np.savez(
        npz_path,
        X=heatmaps["X"],
        Y=heatmaps["Y"],
        thickness=heatmaps["thickness"],
        upper_curvature=heatmaps["upper_curvature"],
        lower_curvature=heatmaps["lower_curvature"],
    )

    # Scalar diagnostics used for bars / CSV
    thick = heatmaps["thickness"]
    hu = heatmaps["upper_curvature"]
    hl = heatmaps["lower_curvature"]
    abs_h = np.nanmean([np.abs(hu), np.abs(hl)], axis=0)

    summary = {
        "tag": tag,
        "membrane": spec.membrane,
        "quadrant": spec.quadrant,
        "peptide": spec.peptide,
        "replicate": spec.replicate,
        "ok": True,
        "cached": False,
        "n_frames_traj": int(len(u.trajectory)),
        "dt_ps": float(u.trajectory.dt),
        "time_window": time_window,
        "analysis_time_ns": analysis_time_ns,
        "thickness_mean_A": float(np.nanmean(thick)),
        "thickness_std_A": float(np.nanstd(thick)),
        "thickness_min_A": float(np.nanmin(thick)),
        "upper_H_mean_per_A": float(np.nanmean(hu)),
        "lower_H_mean_per_A": float(np.nanmean(hl)),
        "abs_H_mean_per_A": float(np.nanmean(abs_h)),
        "abs_H_max_per_A": float(np.nanmax(abs_h)),
        "npz": str(npz_path),
        "run_dir": str(spec.run_dir),
    }
    # drop heavy series from disk json
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def aggregate_replicate_stats(summaries: Sequence[Dict]) -> List[Dict]:
    """Mean ± s.d. across replicates for each (membrane, quadrant, peptide)."""
    groups: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for s in summaries:
        if not s.get("ok"):
            continue
        groups[(s["membrane"], s["quadrant"], s["peptide"])].append(s)

    metrics = [
        "thickness_mean_A",
        "thickness_min_A",
        "abs_H_mean_per_A",
        "abs_H_max_per_A",
        "upper_H_mean_per_A",
        "lower_H_mean_per_A",
    ]
    rows: List[Dict] = []
    for (membrane, quadrant, peptide), items in sorted(groups.items()):
        row: Dict = {
            "membrane": membrane,
            "quadrant": quadrant,
            "peptide": peptide,
            "n_replicates": len(items),
            "replicates": ",".join(sorted(i["replicate"] for i in items)),
        }
        for m in metrics:
            vals = np.asarray([i[m] for i in items], dtype=float)
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(fieldnames) if fieldnames else list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_npz(summary: Dict) -> Optional[Dict[str, np.ndarray]]:
    path = summary.get("npz")
    if not path or not Path(path).is_file():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _smooth_map(arr: np.ndarray, sigma: float = MAP_SMOOTH_SIGMA) -> np.ndarray:
    """Gaussian-smooth a 2-D map (same NaN-aware filter as Fig. 7 curvature)."""
    return smooth_curvature_map(arr, sigma=sigma)


def _draw_fig7_thickness(ax, X, Y, thickness, fill_levels) -> object:
    """Fig. 7 thickness: smooth contourf (viridis) + sparse white labeled contours."""
    im = ax.contourf(
        X, Y, thickness, levels=fill_levels, cmap="viridis", extend="both"
    )
    # Sparse line levels (Fig. 7 uses ~5 labeled contours, not every fill level)
    line_levels = np.linspace(fill_levels[0], fill_levels[-1], 6)[1:-1]
    cs = ax.contour(
        X,
        Y,
        thickness,
        levels=line_levels,
        colors="white",
        alpha=0.6,
        linewidths=0.6,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f", colors="white", inline_spacing=5)
    return im


def _draw_fig7_curvature(ax, X, Y, curv_scaled, fill_levels) -> object:
    """Fig. 7 upper-leaflet curvature: RdBu_r contourf + sparse black contours."""
    im = ax.contourf(
        X, Y, curv_scaled, levels=fill_levels, cmap="RdBu_r", extend="both"
    )
    # ~8 sparse contours across ± range (matches membrane_prop / Fig. 7)
    cmax = float(max(abs(fill_levels[0]), abs(fill_levels[-1])))
    line_levels = np.linspace(-cmax, cmax, 9)
    pos = [lev for lev in line_levels if lev > 1e-12]
    neg = [lev for lev in line_levels if lev < -1e-12]
    zero = [lev for lev in line_levels if abs(lev) <= 1e-12]
    if pos:
        cs_pos = ax.contour(
            X,
            Y,
            curv_scaled,
            levels=pos,
            colors="black",
            alpha=0.6,
            linewidths=0.6,
            linestyles="solid",
        )
        ax.clabel(
            cs_pos, inline=True, fontsize=7, fmt="%.1f", colors="black", inline_spacing=5
        )
    if neg:
        cs_neg = ax.contour(
            X,
            Y,
            curv_scaled,
            levels=neg,
            colors="black",
            alpha=0.6,
            linewidths=0.6,
            linestyles="dashed",
        )
        ax.clabel(
            cs_neg, inline=True, fontsize=7, fmt="%.1f", colors="black", inline_spacing=5
        )
    if zero:
        ax.contour(
            X,
            Y,
            curv_scaled,
            levels=zero,
            colors="black",
            alpha=0.5,
            linewidths=0.7,
            linestyles="solid",
        )
    return im


def _style_map_axes(ax, *, xlabel: bool = True, ylabel: bool = True) -> None:
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    if xlabel:
        ax.set_xlabel("X (Å)", fontsize=8)
    if ylabel:
        ax.set_ylabel("Y (Å)", fontsize=8)


def plot_exemplar_maps(
    summaries: Sequence[Dict],
    out_png: Path,
    *,
    membrane: str,
    exemplars: Dict[str, str],
    control_summaries: Sequence[Dict],
) -> None:
    """Fig. 7–style thickness + upper-leaflet curvature maps (smooth + contours)."""
    panels: List[Tuple[str, Dict]] = []

    ctrl_ok = [s for s in control_summaries if s.get("ok")]
    if ctrl_ok:
        by_rep = {s["replicate"]: s for s in ctrl_ok}
        rep = prefer_replicate(list(by_rep))
        panels.append(("Membrane-only\ncontrol", by_rep[rep]))

    for quad in ("Neither", "Energy", "Aggregate", "Both"):
        pep = exemplars.get(quad)
        if pep is None:
            continue
        cands = [
            s
            for s in summaries
            if s.get("ok")
            and s["membrane"] == membrane
            and s["quadrant"] == quad
            and s["peptide"] == pep
        ]
        if not cands:
            continue
        by_rep = {s["replicate"]: s for s in cands}
        rep = prefer_replicate(list(by_rep))
        label = f"{QUADRANT_LABELS.get(quad, quad)}\n{pep} ({rep})"
        panels.append((label, by_rep[rep]))

    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 5.6), constrained_layout=True)
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)

    loaded: List[Tuple] = []
    thick_s_all, curv_s_all = [], []
    for label, summary in panels:
        hm = _load_npz(summary)
        if hm is None:
            loaded.append((label, None, None, None, None))
            continue
        # Smooth both maps (Fig. 7 look); curvature was already smoothed in analysis
        thick = _smooth_map(hm["thickness"])
        curv = _smooth_map(hm["upper_curvature"]) * CURVATURE_DISPLAY_SCALE
        loaded.append((label, hm["X"], hm["Y"], thick, curv))
        thick_s_all.append(thick)
        curv_s_all.append(curv)

    if not thick_s_all:
        plt.close(fig)
        return

    t_lo = float(np.nanpercentile(np.concatenate([t.ravel() for t in thick_s_all]), 2))
    t_hi = float(np.nanpercentile(np.concatenate([t.ravel() for t in thick_s_all]), 98))
    if t_hi <= t_lo:
        t_lo, t_hi = t_lo - 0.5, t_hi + 0.5
    # Round to 0.1 Å like Fig. 7 colorbars
    t_lo, t_hi = np.floor(t_lo * 10) / 10, np.ceil(t_hi * 10) / 10
    thickness_levels = np.linspace(t_lo, t_hi, 30)

    c_abs = float(
        np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in curv_s_all])), 98)
    )
    c_abs = max(c_abs, 0.5)
    c_abs = float(np.ceil(c_abs * 2) / 2)  # round up to 0.5
    curvature_levels = np.linspace(-c_abs, c_abs, 30)

    im0 = im1 = None
    for col, item in enumerate(loaded):
        label, X, Y, thick, curv = item
        ax_t = axes[0, col]
        ax_h = axes[1, col]
        if thick is None:
            ax_t.set_visible(False)
            ax_h.set_visible(False)
            continue
        im0 = _draw_fig7_thickness(ax_t, X, Y, thick, thickness_levels)
        im1 = _draw_fig7_curvature(ax_h, X, Y, curv, curvature_levels)
        _style_map_axes(ax_t, xlabel=False, ylabel=(col == 0))
        _style_map_axes(ax_h, xlabel=True, ylabel=(col == 0))
        ax_t.set_title(label, fontsize=9)
        if col == 0:
            ax_t.set_ylabel("Thickness\nY (Å)", fontsize=8)
            ax_h.set_ylabel("Upper leaflet curvature\nY (Å)", fontsize=8)

    if im0 is not None:
        cbar0 = fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02)
        cbar0.set_label("Thickness (Å)", fontsize=8)
        cbar0.ax.tick_params(labelsize=7)
        cbar0.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    if im1 is not None:
        cbar1 = fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.025, pad=0.02)
        # Fig. 7 colorbar label after ×100 scaling of H (Å⁻¹)
        cbar1.set_label(r"Mean Curvature (Å$^{-2}$)", fontsize=8)
        cbar1.ax.tick_params(labelsize=7)
        cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    fig.suptitle(
        f"Membrane thickness and curvature — {membrane} (last analysis window)",
        fontsize=12,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_scalar_comparison(
    agg_rows: Sequence[Dict],
    out_png: Path,
    *,
    membrane: str,
) -> None:
    """Bar chart of mean thickness and |H| vs quadrant (replicate mean±s.d.)."""
    rows = [r for r in agg_rows if r["membrane"] == membrane]
    # Also include Membrane-Only control on Gram-ve figure
    ctrl = [r for r in agg_rows if r["membrane"] == "Membrane-Only"]
    if membrane == "Gram-ve" and ctrl:
        rows = ctrl + rows

    if not rows:
        return

    # One bar per (quadrant, peptide); group by quadrant
    labels = []
    thick_m, thick_s, h_m, h_s = [], [], [], []
    colors = []
    palette = {
        "Membrane-Only": "#7f8c8d",
        "Neither": "#95a5a6",
        "Energy": "#2980b9",
        "Aggregate": "#8e44ad",
        "Both": "#c0392b",
    }
    for r in sorted(
        rows,
        key=lambda x: (
            QUADRANT_ORDER.index(x["quadrant"])
            if x["quadrant"] in QUADRANT_ORDER
            else 99,
            x["peptide"],
        ),
    ):
        labels.append(f"{r['quadrant']}\n{r['peptide']}")
        thick_m.append(r["thickness_mean_A_mean"])
        thick_s.append(r["thickness_mean_A_std"])
        h_m.append(r["abs_H_mean_per_A_mean"])
        h_s.append(r["abs_H_mean_per_A_std"])
        colors.append(palette.get(r["quadrant"], "#34495e"))

    x = np.arange(len(labels))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(max(8, 0.9 * len(labels) + 2), 4.2), constrained_layout=True)
    ax0.bar(x, thick_m, yerr=thick_s, color=colors, edgecolor="k", linewidth=0.4, capsize=3)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=8)
    ax0.set_ylabel("Mean thickness (Å)")
    ax0.set_title(f"{membrane}: bilayer thickness")

    ax1.bar(x, h_m, yerr=h_s, color=colors, edgecolor="k", linewidth=0.4, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel(r"Mean $|H|$ (Å$^{-1}$)")
    ax1.set_title(f"{membrane}: mean |curvature|")

    fig.suptitle(
        "Replicate mean ± s.d. (last analysis window) — Fig. 8a stratified set",
        fontsize=11,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_delta_vs_control(
    agg_rows: Sequence[Dict],
    out_png: Path,
    *,
    membrane: str,
) -> None:
    """Δthickness and Δ|H| relative to Membrane-Only control (Gram-ve–matched)."""
    ctrl = [r for r in agg_rows if r["membrane"] == "Membrane-Only"]
    if not ctrl:
        return
    c = ctrl[0]
    c_thick = c["thickness_mean_A_mean"]
    c_h = c["abs_H_mean_per_A_mean"]

    rows = [r for r in agg_rows if r["membrane"] == membrane]
    if not rows:
        return

    labels, d_thick, d_h, colors = [], [], [], []
    palette = {
        "Neither": "#95a5a6",
        "Energy": "#2980b9",
        "Aggregate": "#8e44ad",
        "Both": "#c0392b",
    }
    for r in sorted(
        rows,
        key=lambda x: (
            QUADRANT_ORDER.index(x["quadrant"])
            if x["quadrant"] in QUADRANT_ORDER
            else 99,
            x["peptide"],
        ),
    ):
        labels.append(f"{r['quadrant']}\n{r['peptide']}")
        d_thick.append(r["thickness_mean_A_mean"] - c_thick)
        d_h.append(r["abs_H_mean_per_A_mean"] - c_h)
        colors.append(palette.get(r["quadrant"], "#34495e"))

    x = np.arange(len(labels))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(max(8, 0.9 * len(labels) + 2), 4.2), constrained_layout=True)
    ax0.bar(x, d_thick, color=colors, edgecolor="k", linewidth=0.4)
    ax0.axhline(0, color="k", lw=0.8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=8)
    ax0.set_ylabel(r"Δ thickness vs control (Å)")
    ax0.set_title(f"{membrane}: thinning relative to Membrane-Only")

    ax1.bar(x, d_h, color=colors, edgecolor="k", linewidth=0.4)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel(r"Δ mean $|H|$ vs control (Å$^{-1}$)")
    ax1.set_title(f"{membrane}: curvature change vs Membrane-Only")

    note = (
        "Control bilayer is POPE:POPG ≈ 3:1 (Gram-negative–matched). "
        "Gram+ve peptide systems are POPG-rich; interpret Δ cautiously."
        if membrane == "Gram+ve"
        else "Control bilayer matches Gram-negative POPE:POPG ≈ 3:1 composition."
    )
    fig.suptitle(note, fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def write_notes(path: Path, runs: Sequence[RunSpec], summaries: Sequence[Dict]) -> None:
    """Human-readable caveats for the reviewer response."""
    present = sorted({(r.membrane, r.quadrant) for r in runs})
    lines = [
        "R1Q8 membrane analysis notes",
        "============================",
        "",
        "Figure 8a classes (folder names):",
        "  Energy    = strong membrane interaction energy (< −600 kJ mol⁻¹); AMP-like",
        "  Aggregate = high aggregation propensity (oligomer size ≥ 5); AMY-like",
        "  Both      = dual-functional overlap",
        "  Neither   = satisfies neither criterion",
        "",
        "Available (membrane, quadrant) pairs with ≥1 complete trajectory:",
    ]
    for m, q in present:
        n = sum(1 for r in runs if r.membrane == m and r.quadrant == q)
        lines.append(f"  {m:14} {q:14} n_runs={n}")

    has_energy_gneg = any(r.membrane == "Gram-ve" and r.quadrant == "Energy" for r in runs)
    has_gpos_ctrl = any(r.membrane == "Gram+ve" and r.quadrant == "Membrane-Only" for r in runs)
    lines += [
        "",
        "Gaps relative to a full stratified design:",
        f"  - Gram-ve Energy (AMP-like) trajectories: {'PRESENT' if has_energy_gneg else 'MISSING in this archive'}",
        f"  - Gram+ve peptide-free Membrane-Only control: {'PRESENT' if has_gpos_ctrl else 'MISSING (only Gram-ve–matched Membrane-Only deposited)'}",
        "",
        "Protocol mirrors Fig. 7 / Methods 2.4 thickness–curvature analysis",
        "(scripts/simulation/membrane_thickness_curvature.py).",
        "",
        f"Successful analyses: {sum(1 for s in summaries if s.get('ok'))} / {len(summaries)}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="R1Q8 Fig. 8a quadrant membrane thickness/curvature analysis (Fig. 7–style)."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="CG_Paper_production_data_2026-10-14 root",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-root>/analysis_membrane_R1Q8)",
    )
    p.add_argument("--analysis-time-ns", type=float, default=50.0)
    p.add_argument("--window", choices=("first", "last", "both"), default="last")
    p.add_argument("--grid-resolution", type=int, default=50)
    p.add_argument("--smooth-sigma", type=float, default=1.2)
    p.add_argument("--force", action="store_true", help="Recompute even if cached")
    p.add_argument(
        "--exemplar",
        action="append",
        default=[],
        metavar="QUADRANT=CGxx",
        help="Override map exemplar, e.g. Both=CG44 (repeatable)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_cli().parse_args(argv)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "analysis_membrane_R1Q8"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_method_metadata(out_dir)

    exemplars = dict(DEFAULT_EXEMPLARS)
    for item in args.exemplar:
        if "=" not in item:
            raise SystemExit(f"Bad --exemplar {item!r}; expected QUADRANT=CGxx")
        q, pep = item.split("=", 1)
        exemplars[q] = pep

    runs = discover_runs(data_root)
    if not runs:
        raise SystemExit(f"No complete step7_production runs under {data_root}")

    windows = ["first", "last"] if args.window == "both" else [args.window]
    all_summaries: List[Dict] = []

    print(f"Found {len(runs)} complete runs → {out_dir}")
    for w in windows:
        print(f"=== window={w}  last/first {args.analysis_time_ns} ns ===")
        summaries: List[Dict] = []
        for i, spec in enumerate(runs, 1):
            print(f"[{i}/{len(runs)}] {run_key(spec)} ...", flush=True)
            s = analyze_one(
                spec,
                out_dir,
                analysis_time_ns=args.analysis_time_ns,
                time_window=w,
                grid_resolution=args.grid_resolution,
                smooth_sigma=args.smooth_sigma,
                force=args.force,
            )
            summaries.append(s)
            if s.get("ok"):
                print(
                    f"    thickness={s['thickness_mean_A']:.2f} Å  "
                    f"|H|={s['abs_H_mean_per_A']:.3e} Å^-1"
                    f"{' (cached)' if s.get('cached') else ''}"
                )
            else:
                print(f"    FAIL: {s.get('error')}")

        # Per-run CSV
        write_csv(
            out_dir / f"per_run_summary_{w}.csv",
            summaries,
            fieldnames=[
                "tag",
                "membrane",
                "quadrant",
                "peptide",
                "replicate",
                "ok",
                "thickness_mean_A",
                "thickness_std_A",
                "thickness_min_A",
                "upper_H_mean_per_A",
                "lower_H_mean_per_A",
                "abs_H_mean_per_A",
                "abs_H_max_per_A",
                "n_frames_traj",
                "dt_ps",
                "npz",
                "run_dir",
            ],
        )
        agg = aggregate_replicate_stats(summaries)
        write_csv(out_dir / f"replicate_mean_summary_{w}.csv", agg)

        ctrl = [s for s in summaries if s["membrane"] == "Membrane-Only"]
        for membrane in ("Gram-ve", "Gram+ve"):
            plot_exemplar_maps(
                summaries,
                out_dir / f"fig7_style_maps_{membrane}_{w}.png",
                membrane=membrane,
                exemplars=exemplars,
                control_summaries=ctrl,
            )
            plot_scalar_comparison(
                agg,
                out_dir / f"scalar_bars_{membrane}_{w}.png",
                membrane=membrane,
            )
            plot_delta_vs_control(
                agg,
                out_dir / f"delta_vs_control_{membrane}_{w}.png",
                membrane=membrane,
            )

        write_notes(out_dir / "ANALYSIS_NOTES.txt", runs, summaries)
        if w == windows[-1]:
            all_summaries = summaries

    # Inventory of discovery
    inv = [
        {
            "membrane": r.membrane,
            "quadrant": r.quadrant,
            "peptide": r.peptide,
            "replicate": r.replicate,
            "run_dir": str(r.run_dir),
        }
        for r in runs
    ]
    write_csv(out_dir / "run_inventory.csv", inv)
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
