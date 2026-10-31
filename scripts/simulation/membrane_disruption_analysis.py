#!/usr/bin/env python3
"""Thickness and curvature for the nine-peptide disruption runs in Figures 7 and S6.

Figure 7 (Gram−ve) is a vertical stack of three windows:

  (a) High-MIC analog, CG9,  9950–9999 ns
  (b) Intermediate-MIC analog, CG44, 5850–5899 ns
  (c) Low-MIC analog, CG40,  9950–9999 ns

Figure S6 (Gram+ve) uses the same peptide order. CG9 and CG40 are 9950–9999 ns.
CG44 Gram+ve is also 9950–9999 ns; that production trajectory is not in the
archive, so the saved ``membrane_maps_9950-9999ns.npz`` from that run is used.

Protocol matches ``aug_24/lip_bilayer_curve_thikness.py``: thickness from
PO4/NH3 headgroup positions, curvature from lipid centre-of-mass, displayed
curvature = H × 100 with a fixed 38–43 Å and −2–2 color scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import MDAnalysis as mda
except ImportError as exc:  # pragma: no cover
    raise ImportError("MDAnalysis is required. pip install MDAnalysis") from exc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .membrane_thickness_curvature import (
    DEFAULT_HEADGROUP_ATOMS,
    DEFAULT_LIPID_RESIDUES,
    interpolate_surface,
    mean_curvature_monge,
    smooth_curvature_map,
    thickness_map,
    write_method_metadata,
)
from .review_quadrant_membrane_analysis import write_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(
    "/media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/"
    "W_BiGAN_GP_monash/model_saved_231206/vijay/aug_24"
)
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "simulation" / "membrane_disruption"

THICKNESS_LEVELS = np.linspace(38, 43, 30)
CURVATURE_LEVELS = np.linspace(-2, 2, 30)
CURVATURE_DISPLAY_SCALE = 100.0

# membrane, peptide, class label, path relative to aug_24, start ns, end ns, figure
PAPER_RUNS: Tuple[Tuple[str, str, str, str, float, float, str], ...] = (
    ("Gram-ve", "CG9", "High-MIC analog", "Gram -ve/CG9_9", 9950.0, 9999.0, "Figure 7"),
    (
        "Gram-ve",
        "CG44",
        "Intermediate-MIC analog",
        "Gram -ve/CG44_9",
        5850.0,
        5899.0,
        "Figure 7",
    ),
    ("Gram-ve", "CG40", "Low-MIC analog", "Gram -ve/CG40_9", 9950.0, 9999.0, "Figure 7"),
    ("Gram+ve", "CG9", "High-MIC analog", "gram+ve/CG9", 9950.0, 9999.0, "Figure S6"),
    (
        "Gram+ve",
        "CG44",
        "Intermediate-MIC analog",
        "gram+ve/CG44",
        9950.0,
        9999.0,
        "Figure S6",
    ),
    ("Gram+ve", "CG40", "Low-MIC analog", "gram+ve/CG40", 9950.0, 9999.0, "Figure S6"),
)


def _window_tag(start_ns: float, end_ns: float) -> str:
    return f"{start_ns:.0f}-{end_ns:.0f}ns"


def _topology_and_traj(run_dir: Path) -> Optional[Tuple[Path, Path]]:
    trajectories = [
        run_dir / "step7_production.xtc",
        run_dir / "step7_production_1ofrms_1ns.xtc",
    ]
    xtc = next((p for p in trajectories if p.is_file()), None)
    if xtc is None:
        return None
    for name in ("step7_production.gro", "step7_production.tpr", "step5_charmm2gmx.pdb"):
        top = run_dir / name
        if top.is_file():
            return top, xtc
    return None


def _sample_frames(universe: "mda.Universe", start_ns: float, end_ns: float) -> List[int]:
    universe.trajectory[0]
    t0 = float(universe.trajectory.ts.time) / 1000.0
    n = len(universe.trajectory)
    if n < 2:
        return [0]
    universe.trajectory[1]
    dt = float(universe.trajectory.ts.time) / 1000.0 - t0
    if dt <= 0:
        dt = 1.0
    step = max(1, int(round(1.0 / dt)))
    start = int(round((start_ns - t0) / dt))
    end = int(round((end_ns - t0) / dt))
    start = max(0, min(start, n - 1))
    end = max(0, min(end, n - 1))
    if end < start:
        return []
    return list(range(start, end + 1, step))


def _leaflets(universe: "mda.Universe"):
    membrane_sel = " or ".join(
        f"(resname {lipid} and name {atom})"
        for lipid in DEFAULT_LIPID_RESIDUES
        for atom in DEFAULT_HEADGROUP_ATOMS
    )
    heads = universe.select_atoms(membrane_sel)
    if len(heads) == 0:
        return None
    z = heads.positions[:, 2]
    z_med = float(np.median(z))
    upper_heads = heads[z > z_med]
    lower_heads = heads[z <= z_med]
    upper_res = universe.select_atoms(
        "resid " + " ".join(str(r) for r in np.unique(upper_heads.resids))
    )
    lower_res = universe.select_atoms(
        "resid " + " ".join(str(r) for r in np.unique(lower_heads.resids))
    )
    return upper_heads, lower_heads, upper_res, lower_res


def analyze_window(
    universe: "mda.Universe",
    start_ns: float,
    end_ns: float,
    *,
    grid_resolution: int,
    smooth_sigma: float,
) -> Optional[Dict[str, np.ndarray]]:
    """Headgroup thickness and lipid-COM curvature over an absolute time window."""
    leaflets = _leaflets(universe)
    if leaflets is None:
        return None
    upper_heads, lower_heads, upper_res, lower_res = leaflets
    frames = _sample_frames(universe, start_ns, end_ns)
    if not frames:
        return None

    thick_sum = curv_u_sum = curv_l_sum = None
    X_grid = Y_grid = None
    n_used = 0
    for iframe in frames:
        ts = universe.trajectory[iframe]
        box = ts.dimensions[:3]
        _, _, z_u = interpolate_surface(upper_heads.positions, box, grid_resolution)
        _, _, z_l = interpolate_surface(lower_heads.positions, box, grid_resolution)
        x_u, y_u, z_uc = interpolate_surface(
            upper_res.center_of_mass(compound="residues"), box, grid_resolution
        )
        x_l, y_l, z_lc = interpolate_surface(
            lower_res.center_of_mass(compound="residues"), box, grid_resolution
        )
        thick = thickness_map(z_u, z_l)
        curv_u = mean_curvature_monge(x_u, y_u, z_uc)
        curv_l = mean_curvature_monge(x_l, y_l, z_lc)
        if thick_sum is None:
            thick_sum = np.zeros_like(thick)
            curv_u_sum = np.zeros_like(curv_u)
            curv_l_sum = np.zeros_like(curv_l)
            X_grid, Y_grid = x_u, y_u
        thick_sum += np.nan_to_num(thick, nan=0.0)
        curv_u_sum += np.nan_to_num(curv_u, nan=0.0)
        curv_l_sum += np.nan_to_num(curv_l, nan=0.0)
        n_used += 1

    thick_avg = thick_sum / n_used
    curv_u = smooth_curvature_map(curv_u_sum / n_used, sigma=smooth_sigma)
    curv_l = smooth_curvature_map(curv_l_sum / n_used, sigma=smooth_sigma)
    return {
        "X": X_grid,
        "Y": Y_grid,
        "thickness": thick_avg,
        "curvature_upper": curv_u * CURVATURE_DISPLAY_SCALE,
        "curvature_lower": curv_l * CURVATURE_DISPLAY_SCALE,
        "n_frames": np.array(n_used),
    }


def _load_archived(run_dir: Path, start_ns: float, end_ns: float) -> Optional[Dict[str, np.ndarray]]:
    path = run_dir / f"membrane_maps_{_window_tag(start_ns, end_ns)}.npz"
    if not path.is_file():
        return None
    data = np.load(path)
    return {k: data[k] for k in ("X", "Y", "thickness", "curvature_upper", "curvature_lower")}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[m].ravel(), b[m].ravel())[0, 1])


def analyze_run(
    membrane: str,
    peptide: str,
    mic_class: str,
    figure: str,
    run_dir: Path,
    out_dir: Path,
    start_ns: float,
    end_ns: float,
    *,
    grid_resolution: int,
    smooth_sigma: float,
    force: bool,
) -> Dict:
    tag = _window_tag(start_ns, end_ns)
    dest = out_dir / "maps" / membrane / peptide
    dest.mkdir(parents=True, exist_ok=True)
    npz_path = dest / f"membrane_maps_{tag}.npz"
    json_path = dest / "summary.json"
    if npz_path.is_file() and json_path.is_file() and not force:
        summary = json.loads(json_path.read_text())
        if summary.get("ok"):
            summary["cached"] = True
            return summary

    archived = _load_archived(run_dir, start_ns, end_ns)
    pair = _topology_and_traj(run_dir)
    maps = None
    source = ""
    topology = trajectory = ""
    compare = {}
    if pair is not None:
        topology, trajectory = str(pair[0]), str(pair[1])
        universe = mda.Universe(topology, trajectory)
        maps = analyze_window(
            universe,
            start_ns,
            end_ns,
            grid_resolution=grid_resolution,
            smooth_sigma=smooth_sigma,
        )
        source = "trajectory"
        if maps is not None and archived is not None:
            compare = {
                "thickness_corr_vs_archived": _corr(maps["thickness"], archived["thickness"]),
                "curvature_corr_vs_archived": _corr(
                    maps["curvature_upper"], archived["curvature_upper"]
                ),
                "archived_thickness_mean_A": float(np.nanmean(archived["thickness"])),
            }
    if maps is None and archived is not None:
        maps = archived
        source = "archived_map"
    if maps is None:
        summary = {
            "membrane": membrane,
            "peptide": peptide,
            "mic_class": mic_class,
            "figure": figure,
            "ok": False,
            "error": f"no trajectory or membrane_maps_{tag}.npz under {run_dir}",
            "run_dir": str(run_dir),
            "time_start_ns": start_ns,
            "time_end_ns": end_ns,
        }
        json_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    np.savez(
        npz_path,
        X=maps["X"],
        Y=maps["Y"],
        thickness=maps["thickness"],
        curvature_upper=maps["curvature_upper"],
        curvature_lower=maps["curvature_lower"],
    )
    summary = {
        "membrane": membrane,
        "peptide": peptide,
        "mic_class": mic_class,
        "figure": figure,
        "ok": True,
        "cached": False,
        "source": source,
        "topology": topology,
        "trajectory": trajectory,
        "run_dir": str(run_dir),
        "time_start_ns": start_ns,
        "time_end_ns": end_ns,
        "n_frames": int(maps["n_frames"]) if "n_frames" in maps else "",
        "thickness_mean_A": float(np.nanmean(maps["thickness"])),
        "thickness_std_A": float(np.nanstd(maps["thickness"])),
        "upper_curvature_mean": float(np.nanmean(maps["curvature_upper"])),
        "lower_curvature_mean": float(np.nanmean(maps["curvature_lower"])),
        "npz": str(npz_path),
        **compare,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _colorbar(ax, im, label: str, ticks: np.ndarray) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=8)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=7)


def _draw_thickness(ax, X, Y, thickness) -> None:
    im = ax.contourf(
        X, Y, thickness, levels=THICKNESS_LEVELS, cmap="viridis", extend="both"
    )
    cs = ax.contour(X, Y, thickness, levels=5, colors="white", alpha=0.6, linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f", colors="white", inline_spacing=5)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    _colorbar(ax, im, "Thickness (Å)", np.arange(38, 43.5, 0.5))


def _draw_curvature(ax, X, Y, curvature) -> None:
    im = ax.contourf(
        X, Y, curvature, levels=CURVATURE_LEVELS, cmap="RdBu_r", extend="both"
    )
    cs = ax.contour(X, Y, curvature, levels=5, colors="black", alpha=0.6, linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f", colors="black", inline_spacing=5)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    _colorbar(ax, im, r"Mean Curvature (Å$^{-2}$)", np.arange(-2, 2.5, 0.5))


def plot_stack(summaries: Sequence[Dict], out_png: Path, *, membrane: str, figure: str) -> None:
    rows = [s for s in summaries if s.get("ok") and s["membrane"] == membrane]
    if not rows:
        return
    order = {"High-MIC analog": 0, "Intermediate-MIC analog": 1, "Low-MIC analog": 2}
    rows = sorted(rows, key=lambda s: order.get(s["mic_class"], 9))
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(6.3, 2.35 * n))
    if n == 1:
        axes = np.asarray(axes).reshape(1, 2)
    for i, summary in enumerate(rows):
        data = np.load(summary["npz"])
        ax_t, ax_c = axes[i, 0], axes[i, 1]
        _draw_thickness(ax_t, data["X"], data["Y"], data["thickness"])
        _draw_curvature(ax_c, data["X"], data["Y"], data["curvature_upper"])
        ax_t.set_xlabel("X (Å)", fontsize=8)
        ax_c.set_xlabel("X (Å)", fontsize=8)
        ax_t.set_ylabel("Y (Å)", fontsize=8)
        ax_c.set_ylabel("Y (Å)", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.98, bottom=0.04, wspace=0.22, hspace=0.30)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Figure 7 / S6 thickness and curvature from the paper time windows."
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--grid-resolution", type=int, default=50)
    p.add_argument("--smooth-sigma", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_cli().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_method_metadata(out_dir)

    summaries: List[Dict] = []
    for membrane, peptide, mic_class, rel, start_ns, end_ns, figure in PAPER_RUNS:
        run_dir = Path(args.data_root) / rel
        print(f"{figure} {membrane} {peptide} {start_ns:.0f}-{end_ns:.0f} ns ...", flush=True)
        summary = analyze_run(
            membrane,
            peptide,
            mic_class,
            figure,
            run_dir,
            out_dir,
            start_ns,
            end_ns,
            grid_resolution=args.grid_resolution,
            smooth_sigma=args.smooth_sigma,
            force=args.force,
        )
        summaries.append(summary)
        if summary.get("ok"):
            extra = ""
            if "thickness_corr_vs_archived" in summary:
                extra = f"  corr(thickness)={summary['thickness_corr_vs_archived']:.3f}"
            print(
                f"  {summary['source']}: thickness={summary['thickness_mean_A']:.2f} Å{extra}"
                f"{' (cached)' if summary.get('cached') else ''}"
            )
        else:
            print(f"  SKIP: {summary.get('error')}")

    fig_dir = out_dir / "figures"
    plot_stack(
        summaries,
        fig_dir / "thickness_curvature_Gram-ve.png",
        membrane="Gram-ve",
        figure="Figure 7",
    )
    plot_stack(
        summaries,
        fig_dir / "thickness_curvature_Gram+ve.png",
        membrane="Gram+ve",
        figure="Figure S6",
    )
    write_csv(
        out_dir / "tables" / "summary.csv",
        summaries,
        fieldnames=[
            "figure",
            "membrane",
            "peptide",
            "mic_class",
            "time_start_ns",
            "time_end_ns",
            "ok",
            "source",
            "thickness_mean_A",
            "thickness_std_A",
            "upper_curvature_mean",
            "lower_curvature_mean",
            "n_frames",
            "thickness_corr_vs_archived",
            "curvature_corr_vs_archived",
            "topology",
            "trajectory",
            "error",
            "npz",
        ],
    )
    lines = [
        "Membrane disruption maps used in Figure 7 and Figure S6",
        "=======================================================",
        "",
        "Produced by scripts/simulation/membrane_disruption_analysis.py.",
        "Each figure is a vertical stack, top to bottom:",
        "  High-MIC analog (CG9), Intermediate-MIC analog (CG44), Low-MIC analog (CG40).",
        "",
        "Figure 7 (Gram-ve):",
        "  CG9  9950-9999 ns   Gram -ve/CG9_9",
        "  CG44 5850-5899 ns   Gram -ve/CG44_9",
        "  CG40 9950-9999 ns   Gram -ve/CG40_9",
        "",
        "Figure S6 (Gram+ve), same peptide order:",
        "  CG9  9950-9999 ns   gram+ve/CG9",
        "  CG44 9950-9999 ns   gram+ve/CG44",
        "  CG40 9950-9999 ns   gram+ve/CG40",
        "",
        "Thickness is the PO4/NH3 headgroup separation (about 40 Å).",
        "Displayed curvature is mean curvature H times 100.",
        "Color scales are fixed at 38-43 Å and -2 to 2, shared by both figures.",
        "",
    ]
    for s in summaries:
        status = "ok" if s.get("ok") else "SKIP"
        lines.append(
            f"{status}  {s.get('figure')}  {s.get('membrane')}  {s.get('peptide')}  "
            f"{s.get('mic_class')}  source={s.get('source', s.get('error'))}"
        )
    (out_dir / "NOTES.txt").write_text("\n".join(lines) + "\n")
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
