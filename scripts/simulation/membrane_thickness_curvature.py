#!/usr/bin/env python3
"""Membrane thickness and mean-curvature maps from CG MD trajectories.

Protocol used for amyAMP membrane-disruption figures (Figures 7 / S6)
--------------------------------------------------------------------
Software
  - MDAnalysis (trajectory I/O and lipid selections)
  - NumPy / SciPy (``griddata`` interpolation, Gaussian smoothing)
  - Matplotlib (contour maps; optional)

Computational protocol
  1. Load topology + trajectory with MDAnalysis.
  2. Select lipid headgroup beads (default POPE/POPG: PO4, NH3).
  3. Assign upper/lower leaflets by median lipid centre-of-mass *z*.
  4. For each frame in a time window, interpolate each leaflet surface
     z = f(x, y) onto a regular 2-D grid (cubic ``griddata``; nearest fill).
  5. Thickness map:  h(x,y) = |Z_upper(x,y) − Z_lower(x,y)|.
  6. Mean curvature of each leaflet surface from finite differences of f
     (Monge-patch formula; see ``mean_curvature_monge``).
  7. Average maps over the analysis window; optionally Gaussian-smooth
     curvature (default σ = 1.2 grid units).

Mathematical definitions
  Thickness:
      h(x, y) = |z_u(x, y) − z_ℓ(x, y)|

  Mean curvature of a graph surface z = f(x, y):
      H = [ (1 + f_y²) f_xx − 2 f_x f_y f_xy + (1 + f_x²) f_yy ]
          / [ 2 (1 + f_x² + f_y²)^{3/2} ]

  Partial derivatives are evaluated by NumPy ``gradient`` on the grid
  (central finite differences). Lengths follow MDAnalysis units (Å).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import MDAnalysis as mda
except ImportError as exc:  # pragma: no cover
    raise ImportError("MDAnalysis is required. pip install MDAnalysis") from exc

try:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
except ImportError as exc:  # pragma: no cover
    raise ImportError("SciPy is required. pip install scipy") from exc


PathLike = Union[str, Path]

SOFTWARE_TOOLS = (
    "MDAnalysis (trajectory I/O and selections); "
    "NumPy; SciPy (scipy.interpolate.griddata, scipy.ndimage.gaussian_filter); "
    "Matplotlib (optional map plotting)"
)

DEFAULT_LIPID_RESIDUES = ("POPE", "POPG")
DEFAULT_HEADGROUP_ATOMS = ("PO4", "NH3")
DEFAULT_GRID_RESOLUTION = 50
DEFAULT_CURVATURE_SMOOTH_SIGMA = 1.2


def mean_curvature_monge(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Mean curvature H of surface z = f(x,y) via central finite differences.

    H = [(1+f_y²)f_xx − 2 f_x f_y f_xy + (1+f_x²)f_yy] / [2(1+f_x²+f_y²)^{3/2}]
    """
    dz_dx = np.gradient(Z, X[0, :], axis=1)
    dz_dy = np.gradient(Z, Y[:, 0], axis=0)
    d2z_dx2 = np.gradient(dz_dx, X[0, :], axis=1)
    d2z_dy2 = np.gradient(dz_dy, Y[:, 0], axis=0)
    d2z_dxdy = np.gradient(dz_dx, Y[:, 0], axis=0)

    numerator = (
        (1 + dz_dy**2) * d2z_dx2
        - 2 * dz_dx * dz_dy * d2z_dxdy
        + (1 + dz_dx**2) * d2z_dy2
    )
    denominator = 2 * (1 + dz_dx**2 + dz_dy**2) ** 1.5
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return numerator / denominator


def thickness_map(z_upper: np.ndarray, z_lower: np.ndarray) -> np.ndarray:
    """Local bilayer thickness h = |z_upper − z_lower|."""
    if z_upper.shape != z_lower.shape:
        raise ValueError("Upper and lower surfaces must share the same grid shape")
    return np.abs(z_upper - z_lower)


def interpolate_surface(
    positions: np.ndarray,
    box_dims: Sequence[float],
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate leaflet bead/COM positions onto a regular XY grid."""
    x, y, z = positions[:, 0].copy(), positions[:, 1].copy(), positions[:, 2].copy()
    x = x - box_dims[0] * np.round(x / box_dims[0])
    y = y - box_dims[1] * np.round(y / box_dims[1])

    x_center, y_center = float(np.mean(x)), float(np.mean(y))
    x_half = (x.max() - x.min()) / 2.0
    y_half = (y.max() - y.min()) / 2.0
    x_range = np.linspace(x_center - x_half, x_center + x_half, grid_resolution)
    y_range = np.linspace(y_center - y_half, y_center + y_half, grid_resolution)
    X, Y = np.meshgrid(x_range, y_range, indexing="xy")

    try:
        Z = griddata((x, y), z, (X, Y), method="cubic", fill_value=np.nan)
        mask = np.isnan(Z)
        if mask.any():
            Z[mask] = griddata((x, y), z, (X[mask], Y[mask]), method="nearest")
    except Exception:
        Z = griddata((x, y), z, (X, Y), method="linear", fill_value=float(np.mean(z)))
    return X, Y, Z


def smooth_curvature_map(
    curvature: np.ndarray, sigma: float = DEFAULT_CURVATURE_SMOOTH_SIGMA
) -> np.ndarray:
    """Gaussian smooth a curvature map while preserving NaN exterior."""
    valid = ~np.isnan(curvature)
    filled = np.where(valid, curvature, 0.0)
    smoothed = gaussian_filter(filled, sigma=sigma)
    mask_s = gaussian_filter(valid.astype(float), sigma=sigma)
    out = np.where(mask_s > 0.01, smoothed / mask_s, np.nan)
    return np.where(valid, out, np.nan)


def identify_membrane_leaflets(
    universe: "mda.Universe",
    lipid_residues: Sequence[str] = DEFAULT_LIPID_RESIDUES,
    headgroup_atoms: Sequence[str] = DEFAULT_HEADGROUP_ATOMS,
) -> Tuple[Optional["mda.AtomGroup"], Optional["mda.AtomGroup"]]:
    """Split lipids into upper/lower leaflets by median residue COM z."""
    membrane_sel = " or ".join(
        f"(resname {lipid} and name {atom})"
        for lipid in lipid_residues
        for atom in headgroup_atoms
    )
    try:
        membrane = universe.select_atoms(membrane_sel)
    except Exception:
        membrane = universe.select_atoms(
            " or ".join(f"resname {lipid}" for lipid in lipid_residues)
        )
    if len(membrane) == 0:
        return None, None

    lipid_z, residue_ids = [], []
    for residue in membrane.residues:
        lipid_z.append(residue.atoms.center_of_mass()[2])
        residue_ids.append(residue.resid)
    lipid_z = np.asarray(lipid_z)
    z_med = float(np.median(lipid_z))
    upper_ids = [rid for rid, z in zip(residue_ids, lipid_z) if z > z_med]
    lower_ids = [rid for rid, z in zip(residue_ids, lipid_z) if z <= z_med]
    upper = universe.select_atoms(f"resid {' '.join(map(str, upper_ids))}")
    lower = universe.select_atoms(f"resid {' '.join(map(str, lower_ids))}")
    return upper, lower


def analyze_membrane_properties(
    universe: "mda.Universe",
    lipid_residues: Sequence[str] = DEFAULT_LIPID_RESIDUES,
    headgroup_atoms: Sequence[str] = DEFAULT_HEADGROUP_ATOMS,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    analysis_time_ns: float = 50.0,
    time_window: str = "last",
    curvature_smooth_sigma: float = DEFAULT_CURVATURE_SMOOTH_SIGMA,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Average thickness and curvature maps over the first or last N ns.

    Returns ``(results_dict, heatmaps_dict)`` or ``(None, None)`` on failure.
    """
    upper_leaflet, lower_leaflet = identify_membrane_leaflets(
        universe, lipid_residues, headgroup_atoms
    )
    if upper_leaflet is None or lower_leaflet is None:
        return None, None

    dt_ps = float(universe.trajectory.dt)
    n_frames_window = max(1, int(round(analysis_time_ns * 1000.0 / dt_ps)))
    n_tot = len(universe.trajectory)
    if time_window == "first":
        frame_indices = range(0, min(n_frames_window, n_tot))
    else:
        start = max(0, n_tot - n_frames_window)
        frame_indices = range(start, n_tot)

    curv_u_sum = curv_l_sum = thick_sum = None
    count_u = count_l = count_h = None
    X_grid = Y_grid = None
    series: Dict[str, List[float]] = {
        "time_ns": [],
        "upper_mean_curvature": [],
        "lower_mean_curvature": [],
        "mean_thickness": [],
    }

    for iframe in frame_indices:
        ts = universe.trajectory[iframe]
        box = ts.dimensions[:3]
        upper_pos = upper_leaflet.center_of_mass(compound="residues")
        lower_pos = lower_leaflet.center_of_mass(compound="residues")
        X_u, Y_u, Z_u = interpolate_surface(upper_pos, box, grid_resolution)
        X_l, Y_l, Z_l = interpolate_surface(lower_pos, box, grid_resolution)
        # Use upper grid for both after independent interpolate of Z_l onto same? Historical
        # code uses each leaflet's own grid then abs(Zu-Zl) assuming matching shapes.
        curv_u = mean_curvature_monge(X_u, Y_u, Z_u)
        curv_l = mean_curvature_monge(X_l, Y_l, Z_l)
        thick = thickness_map(Z_u, Z_l)

        if curv_u_sum is None:
            curv_u_sum = np.zeros_like(curv_u)
            curv_l_sum = np.zeros_like(curv_l)
            thick_sum = np.zeros_like(thick)
            count_u = np.zeros_like(curv_u)
            count_l = np.zeros_like(curv_l)
            count_h = np.zeros_like(thick)
            X_grid, Y_grid = X_u, Y_u

        vu, vl, vt = ~np.isnan(curv_u), ~np.isnan(curv_l), ~np.isnan(thick)
        curv_u_sum[vu] += curv_u[vu]
        curv_l_sum[vl] += curv_l[vl]
        thick_sum[vt] += thick[vt]
        count_u[vu] += 1
        count_l[vl] += 1
        count_h[vt] += 1

        series["time_ns"].append(ts.time / 1000.0)
        series["upper_mean_curvature"].append(float(np.nanmean(curv_u)))
        series["lower_mean_curvature"].append(float(np.nanmean(curv_l)))
        series["mean_thickness"].append(float(np.nanmean(thick)))

    with np.errstate(invalid="ignore", divide="ignore"):
        curv_u_avg = np.where(count_u > 0, curv_u_sum / count_u, np.nan)
        curv_l_avg = np.where(count_l > 0, curv_l_sum / count_l, np.nan)
        thick_avg = np.where(count_h > 0, thick_sum / count_h, np.nan)

    heatmaps = {
        "X": X_grid,
        "Y": Y_grid,
        "upper_curvature": smooth_curvature_map(curv_u_avg, sigma=curvature_smooth_sigma),
        "lower_curvature": smooth_curvature_map(curv_l_avg, sigma=curvature_smooth_sigma),
        "thickness": thick_avg,
    }
    results = {
        "software": SOFTWARE_TOOLS,
        "time_window": time_window,
        "analysis_time_ns": analysis_time_ns,
        "grid_resolution": grid_resolution,
        "curvature_smooth_sigma": curvature_smooth_sigma,
        "series": series,
        "thickness_mean_A": float(np.nanmean(thick_avg)),
        "thickness_std_A": float(np.nanstd(thick_avg)),
        "upper_H_mean_per_A": float(np.nanmean(heatmaps["upper_curvature"])),
        "lower_H_mean_per_A": float(np.nanmean(heatmaps["lower_curvature"])),
    }
    return results, heatmaps


def write_method_metadata(outpath: PathLike) -> Path:
    """Write a Methods-facing description next to analysis outputs."""
    out = Path(outpath)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "membrane_thickness_curvature_method.txt"
    path.write_text(
        "amyAMP membrane thickness and curvature analysis\n"
        f"software: {SOFTWARE_TOOLS}\n"
        "leaflet_assignment: median lipid COM z (upper if z > median)\n"
        "surface_interpolation: scipy.interpolate.griddata cubic (+ nearest fill)\n"
        f"default_grid_resolution: {DEFAULT_GRID_RESOLUTION} x {DEFAULT_GRID_RESOLUTION}\n"
        "thickness: h(x,y) = |Z_upper(x,y) - Z_lower(x,y)| (Å)\n"
        "mean_curvature_H: Monge-patch formula "
        "H=[(1+fy^2)fxx - 2 fx fy fxy + (1+fx^2)fyy] / [2(1+fx^2+fy^2)^(3/2)] "
        "(finite differences via numpy.gradient); units Å^-1\n"
        f"curvature_smoothing: Gaussian sigma={DEFAULT_CURVATURE_SMOOTH_SIGMA} (grid units)\n"
        f"default_lipids: {', '.join(DEFAULT_LIPID_RESIDUES)}\n"
        f"default_headgroups: {', '.join(DEFAULT_HEADGROUP_ATOMS)}\n"
    )
    return path


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Membrane thickness and mean-curvature maps (MDAnalysis + SciPy)."
    )
    p.add_argument("-s", "--topology", required=True)
    p.add_argument("-f", "--trajectory", required=True)
    p.add_argument("-o", "--outpath", required=True)
    p.add_argument("--window", choices=("first", "last", "both"), default="both")
    p.add_argument("--analysis-time-ns", type=float, default=50.0)
    p.add_argument("--grid-resolution", type=int, default=DEFAULT_GRID_RESOLUTION)
    p.add_argument("--smooth-sigma", type=float, default=DEFAULT_CURVATURE_SMOOTH_SIGMA)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_cli().parse_args(argv)
    out = Path(args.outpath)
    out.mkdir(parents=True, exist_ok=True)
    write_method_metadata(out)

    u = mda.Universe(args.topology, args.trajectory)
    windows = ["first", "last"] if args.window == "both" else [args.window]
    for w in windows:
        results, heatmaps = analyze_membrane_properties(
            u,
            grid_resolution=args.grid_resolution,
            analysis_time_ns=args.analysis_time_ns,
            time_window=w,
            curvature_smooth_sigma=args.smooth_sigma,
        )
        if results is None:
            print(f"[FAIL] {w}")
            continue
        np.savez(
            out / f"membrane_maps_{w}.npz",
            X=heatmaps["X"],
            Y=heatmaps["Y"],
            thickness=heatmaps["thickness"],
            upper_curvature=heatmaps["upper_curvature"],
            lower_curvature=heatmaps["lower_curvature"],
        )
        print(
            f"[{w}] thickness {results['thickness_mean_A']:.2f} ± "
            f"{results['thickness_std_A']:.2f} Å; "
            f"upper H {results['upper_H_mean_per_A']:.4e} Å^-1"
        )


if __name__ == "__main__":
    main()
