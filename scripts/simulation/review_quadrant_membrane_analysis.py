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
  3. Writes one thickness/curvature map figure per membrane.

Default figure exemplars (one peptide per quadrant):
  Aggregate→CG94, Both→CG40, Energy→CG24, Neither→CG55.
Each figure stacks two replicate blocks. The upper block is the primary
trajectory, or replicate 1 when primary is missing. The lower block is
replicate 2, or the other available replicate when replicate 2 is missing.
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

# Analysis products (figures, tables, grid maps). Trajectories stay on the data drive.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "simulation" / "stratified_membrane_disruption"

# Figure 8a class labels used in folder names → panel subtitles
QUADRANT_LABELS = {
    "Membrane-Only": "Membrane only",
    "Neither": "Neither",
    "Energy": "AMP",
    "Aggregate": "AMY",
    "Both": "Both",
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


# Membrane-Only/<composition>/<run>/ folder names → analysis membrane label
MEMBRANE_ONLY_DIRS = {
    "gram-ve_membrane": "Gram-ve",
    "gram+ve_membrane": "Gram+ve",
    "gram-ve": "Gram-ve",
    "gram+ve": "Gram+ve",
}


def _production_run(
    membrane: str,
    quadrant: str,
    peptide: str,
    run_dir: Path,
) -> Optional[RunSpec]:
    tpr = run_dir / "step7_production.tpr"
    xtc = run_dir / "step7_production.xtc"
    if not (tpr.is_file() and xtc.is_file()):
        return None
    return RunSpec(
        membrane=membrane,
        quadrant=quadrant,
        peptide=peptide,
        replicate=run_dir.name,
        run_dir=run_dir,
        tpr=tpr,
        xtc=xtc,
    )


def discover_runs(data_root: Path) -> List[RunSpec]:
    """Find all complete production runs under the review_sim archive."""
    runs: List[RunSpec] = []
    root = Path(data_root)

    mem_only = root / "Membrane-Only"
    if mem_only.is_dir():
        for child in sorted(mem_only.iterdir()):
            if not child.is_dir():
                continue
            membrane = MEMBRANE_ONLY_DIRS.get(child.name.lower())
            if membrane is not None:
                # Membrane-Only/gram±ve_membrane/{primary,replicate_*}
                for run_dir in sorted(child.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    spec = _production_run(
                        membrane, "Membrane-Only", "control", run_dir
                    )
                    if spec is not None:
                        runs.append(spec)
                continue
            # Legacy flat layout: Membrane-Only/{primary,replicate_*} (Gram−ve lipids)
            spec = _production_run("Gram-ve", "Membrane-Only", "control", child)
            if spec is not None:
                runs.append(spec)

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
                        spec = _production_run(membrane, quadrant, peptide, run_dir)
                        if spec is not None:
                            runs.append(spec)
    return runs


def prefer_replicate(replicates: Sequence[str]) -> str:
    """Upper map block: primary, else replicate 1, else the first available."""
    if "primary" in replicates:
        return "primary"
    if "replicate_1" in replicates:
        return "replicate_1"
    return sorted(replicates)[0]


def second_replicate(replicates: Sequence[str], used: str) -> Optional[str]:
    """Lower map block: replicate 2, else the other replicate that is not ``used``."""
    rest = [name for name in replicates if name != used]
    if "replicate_2" in rest:
        return "replicate_2"
    if "replicate_1" in rest:
        return "replicate_1"
    if "primary" in rest:
        return "primary"
    if rest:
        return sorted(rest)[0]
    return None


def other_remaining_replicate(replicates: Sequence[str], used: str) -> Optional[str]:
    """Membrane-only lower panel: the replicate that is not the upper one and not replicate 2."""
    rest = [name for name in replicates if name != used and name != "replicate_2"]
    if "replicate_1" in rest:
        return "replicate_1"
    if "primary" in rest:
        return "primary"
    if rest:
        return sorted(rest)[0]
    return second_replicate(replicates, used)


def replicate_file_tag(name: str) -> str:
    if name == "replicate_1":
        return "rep1"
    if name == "replicate_2":
        return "rep2"
    return name


def _panel_id(summary: Dict) -> str:
    if summary["quadrant"] == "Membrane-Only":
        return "membrane-only"
    return summary["peptide"]


def _replicate_chunk(summaries: Sequence[Optional[Dict]]) -> str:
    reps: List[str] = []
    ids: List[str] = []
    for summary in summaries:
        if summary is None:
            continue
        reps.append(replicate_file_tag(summary["replicate"]))
        ids.append(_panel_id(summary))
    if not reps:
        return "missing"
    chunks: List[str] = []
    start = 0
    for i, rep in enumerate(reps + [""]):
        if i == len(reps) or rep != reps[start]:
            chunks.append(f"{reps[start]}_{','.join(ids[start:i])}")
            start = i
    return "_".join(chunks)


def map_figure_name(
    membrane: str,
    upper: Sequence[Optional[Dict]],
    lower: Sequence[Optional[Dict]],
) -> str:
    """Filename from both replicate blocks, columns left to right."""
    return f"{membrane}_{_replicate_chunk(upper)}__{_replicate_chunk(lower)}.png"


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
    run_out = (
        out_dir
        / "maps"
        / spec.membrane
        / spec.quadrant
        / spec.peptide
        / spec.replicate
    )
    run_out.mkdir(parents=True, exist_ok=True)
    npz_path = run_out / f"membrane_maps_{time_window}.npz"
    json_path = run_out / f"summary_{time_window}.json"

    if npz_path.is_file() and json_path.is_file() and not force:
        with open(json_path) as fh:
            summary = json.load(fh)
        summary["cached"] = True
        summary["npz"] = str(npz_path)
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


def _levels_from_maps(
    thicks: Sequence[np.ndarray], curvs: Sequence[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared Fig. 7 color levels from smoothed thickness and scaled curvature maps."""
    t_lo = float(np.nanpercentile(np.concatenate([t.ravel() for t in thicks]), 2))
    t_hi = float(np.nanpercentile(np.concatenate([t.ravel() for t in thicks]), 98))
    if t_hi <= t_lo:
        t_lo, t_hi = t_lo - 0.5, t_hi + 0.5
    t_lo, t_hi = np.floor(t_lo * 10) / 10, np.ceil(t_hi * 10) / 10
    thickness_levels = np.linspace(t_lo, t_hi, 30)

    c_abs = float(
        np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in curvs])), 98)
    )
    c_abs = max(c_abs, 0.5)
    c_abs = float(np.ceil(c_abs * 2) / 2)
    curvature_levels = np.linspace(-c_abs, c_abs, 30)
    return thickness_levels, curvature_levels


def _candidates(
    summaries: Sequence[Dict],
    *,
    membrane: str,
    quadrant: str,
    peptide: Optional[str] = None,
) -> Dict[str, Dict]:
    chosen = [
        s
        for s in summaries
        if s.get("ok")
        and s["membrane"] == membrane
        and s["quadrant"] == quadrant
        and (peptide is None or s["peptide"] == peptide)
    ]
    return {s["replicate"]: s for s in chosen}


def select_map_panels(
    summaries: Sequence[Dict],
    *,
    membrane: str,
    exemplars: Dict[str, str],
) -> List[Tuple[str, Dict, Optional[Dict]]]:
    """Columns left to right, each with an upper and a lower replicate.

    Upper is primary, or replicate 1 when primary is absent. Lower is
    replicate 2, except Membrane only, which uses the other remaining
    replicate (not replicate 2). Column order: Membrane only, Neither, AMY, AMP, Both.
    """
    panels: List[Tuple[str, Dict, Optional[Dict]]] = []
    groups: List[Tuple[str, Dict[str, Dict]]] = []
    ctrl = _candidates(summaries, membrane=membrane, quadrant="Membrane-Only")
    if ctrl:
        groups.append((QUADRANT_LABELS["Membrane-Only"], ctrl))
    for quad in ("Neither", "Aggregate", "Energy", "Both"):
        pep = exemplars.get(quad)
        if pep is None:
            continue
        by_rep = _candidates(
            summaries, membrane=membrane, quadrant=quad, peptide=pep
        )
        if by_rep:
            groups.append((QUADRANT_LABELS.get(quad, quad), by_rep))
    for label, by_rep in groups:
        top = prefer_replicate(list(by_rep))
        if label == QUADRANT_LABELS["Membrane-Only"]:
            bottom = other_remaining_replicate(list(by_rep), top)
        else:
            bottom = second_replicate(list(by_rep), top)
        panels.append((label, by_rep[top], by_rep[bottom] if bottom else None))
    return panels


def smoothed_maps_for_panels(
    panels: Sequence[Tuple[str, Dict]],
) -> Tuple[List[Tuple], List[np.ndarray], List[np.ndarray]]:
    """Load and smooth thickness / scaled curvature for the selected panels."""
    loaded: List[Tuple] = []
    thick_s_all: List[np.ndarray] = []
    curv_s_all: List[np.ndarray] = []
    for label, summary in panels:
        hm = _load_npz(summary)
        if hm is None:
            loaded.append((label, None, None, None, None))
            continue
        thick = _smooth_map(hm["thickness"])
        curv = _smooth_map(hm["upper_curvature"]) * CURVATURE_DISPLAY_SCALE
        loaded.append((label, hm["X"], hm["Y"], thick, curv))
        thick_s_all.append(thick)
        curv_s_all.append(curv)
    return loaded, thick_s_all, curv_s_all


def plot_exemplar_maps(
    summaries: Sequence[Dict],
    out_png: Path,
    *,
    membrane: str,
    exemplars: Dict[str, str],
    thickness_levels: Optional[np.ndarray] = None,
    curvature_levels: Optional[np.ndarray] = None,
) -> None:
    """Thickness and curvature maps, primary (or rep1) above replicate 2."""
    panels = select_map_panels(summaries, membrane=membrane, exemplars=exemplars)
    if not panels:
        return

    n = len(panels)
    # Square map cells, a narrow colorbar column, and a short gap between the two blocks.
    fig = plt.figure(figsize=(2.15 * n + 1.35, 10.15))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1, 1],
        hspace=0.26,
        left=0.07,
        right=0.90,
        top=0.97,
        bottom=0.045,
    )

    loaded_upper, thick_u, curv_u = smoothed_maps_for_panels(
        [(label, top) for label, top, _bottom in panels]
    )
    loaded_lower, thick_l, curv_l = smoothed_maps_for_panels(
        [(label, bottom if bottom is not None else {}) for label, _top, bottom in panels]
    )
    thick_s_all = thick_u + thick_l
    curv_s_all = curv_u + curv_l
    if not thick_s_all:
        plt.close(fig)
        return
    if thickness_levels is None or curvature_levels is None:
        thickness_levels, curvature_levels = _levels_from_maps(thick_s_all, curv_s_all)

    def _paint_block(
        subspec, loaded: Sequence[Tuple], letter0: str, block_title: Optional[str]
    ):
        inner = subspec.subgridspec(
            2,
            n + 1,
            width_ratios=[1] * n + [0.055],
            wspace=0.06,
            hspace=0.20,
        )
        im_t = im_h = None
        ax_t0 = ax_h0 = None
        for col, item in enumerate(loaded):
            label, X, Y, thick, curv = item
            ax_t = fig.add_subplot(inner[0, col])
            ax_h = fig.add_subplot(inner[1, col])
            if col == 0:
                ax_t0 = ax_t
                ax_h0 = ax_h
            if thick is None:
                ax_t.set_visible(False)
                ax_h.set_visible(False)
                continue
            im_t = _draw_fig7_thickness(ax_t, X, Y, thick, thickness_levels)
            im_h = _draw_fig7_curvature(ax_h, X, Y, curv, curvature_levels)
            _style_map_axes(ax_t, xlabel=False, ylabel=(col == 0))
            _style_map_axes(ax_h, xlabel=True, ylabel=(col == 0))
            ax_t.set_title(
                f"({chr(ord(letter0) + col)}) {label}",
                loc="left",
                fontsize=10,
                fontweight="bold",
                pad=2,
            )
            if col == 0:
                ax_t.set_ylabel("Thickness\nY (Å)", fontsize=8)
                ax_h.set_ylabel("Upper leaflet curvature\nY (Å)", fontsize=8)
        if im_t is not None:
            cax = fig.add_subplot(inner[0, n])
            cbar = fig.colorbar(im_t, cax=cax)
            cbar.set_label("Thickness (Å)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        if im_h is not None:
            cax = fig.add_subplot(inner[1, n])
            cbar = fig.colorbar(im_h, cax=cax)
            cbar.set_label(r"Mean Curvature (Å$^{-2}$)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        if ax_t0 is not None and block_title:
            ax_t0.annotate(
                block_title,
                xy=(0.0, 1.16),
                xycoords="axes fraction",
                fontsize=12,
                fontweight="bold",
                ha="left",
                va="bottom",
                annotation_clip=False,
            )
        return ax_t0, ax_h0

    _, ax_curv = _paint_block(outer[0], loaded_upper, "a", "Replicate 1")
    ax_thick, _ = _paint_block(outer[1], loaded_lower, "f", None)
    fig.canvas.draw()
    if ax_curv is not None and ax_thick is not None:
        renderer = fig.canvas.get_renderer()
        above = ax_curv.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        below = ax_thick.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        fig.text(
            ax_thick.get_position().x0,
            (above.y0 + below.y1) / 2.0,
            "Replicate 2",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
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
        f"  - Gram+ve peptide-free Membrane-Only control: {'PRESENT' if has_gpos_ctrl else 'MISSING'}",
        "  - Map color scales are shared between the Gram-ve and Gram+ve Fig. 7–style figures.",
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
        help="Output directory (default: results/simulation/stratified_membrane_disruption)",
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
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    for d in (out_dir, fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)
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
            tab_dir / f"per_run_summary_{w}.csv",
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
        write_csv(tab_dir / f"replicate_mean_summary_{w}.csv", agg)

        # One color scale for both membrane figures so the maps can be compared.
        shared_thick: List[np.ndarray] = []
        shared_curv: List[np.ndarray] = []
        for membrane in ("Gram-ve", "Gram+ve"):
            panels = select_map_panels(
                summaries, membrane=membrane, exemplars=exemplars
            )
            for block in (
                [(label, top) for label, top, _bottom in panels],
                [(label, bottom) for label, _top, bottom in panels if bottom is not None],
            ):
                _, thicks, curvs = smoothed_maps_for_panels(block)
                shared_thick.extend(thicks)
                shared_curv.extend(curvs)
        if shared_thick and shared_curv:
            thickness_levels, curvature_levels = _levels_from_maps(
                shared_thick, shared_curv
            )
        else:
            thickness_levels = curvature_levels = None

        written = []
        for membrane in ("Gram-ve", "Gram+ve"):
            panels = select_map_panels(
                summaries, membrane=membrane, exemplars=exemplars
            )
            name = map_figure_name(
                membrane,
                [top for _label, top, _bottom in panels],
                [bottom for _label, _top, bottom in panels],
            )
            written.append(name)
            plot_exemplar_maps(
                summaries,
                fig_dir / name,
                membrane=membrane,
                exemplars=exemplars,
                thickness_levels=thickness_levels,
                curvature_levels=curvature_levels,
            )
        for stale in fig_dir.glob("*.png"):
            if stale.name not in written:
                stale.unlink()

        write_notes(out_dir / "NOTES.txt", runs, summaries)
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
    write_csv(tab_dir / "run_inventory.csv", inv)
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
