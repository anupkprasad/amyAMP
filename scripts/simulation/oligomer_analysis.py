#!/usr/bin/env python3
"""Oligomer / aggregate size analysis for peptide self-assembly trajectories.

Method (as used for amyAMP Figures 6 and 8)
-------------------------------------------
**Clustering algorithm:** cutoff-based **connected-component clustering**
(also called single-linkage or friends-of-friends clustering) on an undirected
peptide–peptide contact graph, implemented with NetworkX
(``networkx.node_connected_component`` / connected components).

**Contact cutoff:** minimum bead–bead distance ≤ ``min_distance``
(default **5.0 Å = 0.5 nm**).

Pipeline
~~~~~~~~
1. Load a CG trajectory with MDAnalysis.
2. Treat each peptide molecule (MDA ``fragment``) as a graph node.
3. Add an edge if the minimum bead–bead distance ≤ ``min_distance``.
4. Assign oligomers as connected components of that contact graph.
5. Per frame, report the **largest oligomer size** (max component size).
6. For Figure 8 classification, a peptide is “high aggregation” if at least
   ``occupancy`` (default 0.5) of frames in the final ``tail_frames``
   (default 200; ≡ 0.2 µs at 1 ns/frame) have largest size ≥ ``size_threshold``
   (default 5).

Outputs written to ``outpath`` (filenames match the historical analysis pipeline):
  - oligomer-groups.dat
  - oligomer-states.dat
  - oligo-highest-size.dat   (frame, max cluster size)
  - oligo-block-average.dat
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "MDAnalysis is required for oligomer_analysis. "
        "Install with: pip install MDAnalysis"
    ) from exc

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "networkx is required for oligomer_analysis. "
        "Install with: pip install networkx"
    ) from exc


PathLike = Union[str, Path]

# Named explicitly for Methods / reproducibility (Reviewer 3 Comment 3).
CLUSTERING_ALGORITHM = (
    "connected-component clustering (single-linkage / friends-of-friends) "
    "on a peptide–peptide contact graph (NetworkX)"
)
DEFAULT_CONTACT_CUTOFF_A = 5.0  # Å (= 0.5 nm)


def aggregate_analysis(
    uni: "mda.Universe",
    sel_all_atoms: Union[str, "mda.AtomGroup"] = "protein",
    min_distance: float = DEFAULT_CONTACT_CUTOFF_A,
    outpath: PathLike = "analysis/",
    expected_n_molecules: Optional[int] = 8,
    block_window: int = 25,
    clustering_algorithm: str = CLUSTERING_ALGORITHM,
) -> Path:
    """Compute per-frame oligomer membership and largest aggregate size.

    Clustering
    ----------
    ``clustering_algorithm`` (default): cutoff-based connected-component /
    single-linkage (friends-of-friends) clustering on an undirected contact
    graph. Peptides (MDA fragments) are nodes; an edge is added when the
    minimum bead–bead distance is ≤ ``min_distance``. Connected components
    are oligomers (NetworkX).

    Parameters
    ----------
    uni
        MDAnalysis Universe (topology + trajectory).
    sel_all_atoms
        Selection string or AtomGroup covering all peptide beads.
    min_distance
        Contact cutoff in **Ångström** (MDAnalysis length unit). Default 5.0 Å.
    outpath
        Directory for output ``*.dat`` files.
    expected_n_molecules
        If set, warn when fragment count differs (amyAMP used 8).
    block_window
        Moving-average window for ``oligo-block-average.dat``.
    clustering_algorithm
        Human-readable algorithm name recorded in the run log (Methods text).

    Returns
    -------
    Path
        Output directory.
    """
    out = Path(outpath)
    out.mkdir(parents=True, exist_ok=True)

    all_atoms = (
        uni.select_atoms(sel_all_atoms)
        if isinstance(sel_all_atoms, str)
        else sel_all_atoms
    )
    atom_groups = list(all_atoms.fragments)
    n_mol = len(atom_groups)
    n_frames = len(uni.trajectory)

    print(
        f"Clustering algorithm: {clustering_algorithm}\n"
        f"Contact cutoff: {min_distance} Å ({min_distance / 10.0:.2f} nm)\n"
        f"Molecules: {n_mol}, atoms: {len(all_atoms)}, frames: {n_frames}"
    )
    # Persist Methods-facing metadata next to the .dat outputs
    meta_path = out / "oligomer_analysis_method.txt"
    meta_path.write_text(
        "amyAMP oligomer / aggregate size analysis\n"
        f"clustering_algorithm: {clustering_algorithm}\n"
        f"contact_cutoff_angstrom: {min_distance}\n"
        f"contact_cutoff_nm: {min_distance / 10.0}\n"
        "contact_definition: minimum bead-bead distance between two peptide molecules\n"
        "oligomer_definition: connected component of the contact graph\n"
        "frame_metric: size of the largest connected component\n"
        f"n_molecules: {n_mol}\n"
        f"n_frames: {n_frames}\n"
    )
    if expected_n_molecules is not None and n_mol != expected_n_molecules:
        print(
            f"[WARN] expected {expected_n_molecules} molecules, found {n_mol}. "
            "Continuing with detected count."
        )

    groups_path = out / "oligomer-groups.dat"
    states_path = out / "oligomer-states.dat"
    highest_path = out / "oligo-highest-size.dat"
    block_path = out / "oligo-block-average.dat"

    # Also write historical "analysis*" aliases used by aggre_plot
    alias_highest = out / "analysisoligo-highest-size.dat"
    alias_groups = out / "analysisoligomer-groups.dat"
    alias_states = out / "analysisoligomer-states.dat"
    alias_block = out / "analysisoligo-block-average.dat"

    highest_sizes: List[int] = []

    with groups_path.open("w") as f_groups, states_path.open("w") as f_states:
        for ts in uni.trajectory:
            f_groups.write(f"{ts.frame}\t")
            f_states.write(f"{ts.frame}\t")

            contacts = set()
            for i in range(n_mol - 1):
                for j in range(i + 1, n_mol):
                    dist = distance_array(
                        atom_groups[i].positions,
                        atom_groups[j].positions,
                        ts.dimensions,
                    ).min()
                    if dist <= min_distance:
                        contacts.add((i, j))

            graph = nx.Graph()
            graph.add_nodes_from(range(n_mol))
            graph.add_edges_from(contacts)

            frame_max = 1
            for i in range(n_mol):
                component = sorted(nx.node_connected_component(graph, i))
                size = len(component)
                frame_max = max(frame_max, size)
                f_groups.write(f"{component}\t")
                f_states.write(f"{size}\t")

            highest_sizes.append(frame_max)
            f_groups.write("\n")
            f_states.write("\n")

    highest = np.asarray(highest_sizes, dtype=int)
    times = np.arange(len(highest))
    np.savetxt(highest_path, np.column_stack([times, highest]), fmt="%d\t%d")
    np.savetxt(alias_highest, np.column_stack([times, highest]), fmt="%d\t%d")

    # Copy groups/states aliases
    alias_groups.write_text(groups_path.read_text())
    alias_states.write_text(states_path.read_text())

    if len(highest) >= block_window:
        weights = np.ones(block_window) / block_window
        smoothed = np.convolve(highest.astype(float), weights, mode="valid")
        block = np.column_stack([np.arange(smoothed.size), smoothed])
        np.savetxt(block_path, block, fmt="%d\t%.6f")
        np.savetxt(alias_block, block, fmt="%d\t%.6f")
    else:
        print(f"[NOTE] fewer than {block_window} frames; block average skipped.")

    print(f"Wrote oligomer analysis to {out.resolve()}")
    return out


def load_highest_size_library(
    data_dir: PathLike,
    filename: str = "analysisoligo-highest-size.dat",
    ids: Optional[Sequence[int]] = None,
    n_frames: int = 1001,
    exclude: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, List[int], List[dict]]:
    """Load per-trajectory largest-oligomer time series into an array.

    Returns
    -------
    all_data
        Shape ``(n_frames, max_id+1)``; column ``i`` is trajectory ``i``.
    used_ids
        Trajectory IDs successfully loaded.
    inconsistent_time
        List of ``{id: n_rows}`` for trajectories truncated/padded to ``n_frames``.
    """
    data_dir = Path(data_dir)
    exclude_set = set(exclude or [])
    if ids is None:
        # discover numeric subdirectories
        ids = sorted(
            int(p.name)
            for p in data_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        )

    max_id = max(ids) if ids else 0
    all_data = np.zeros((n_frames, max_id + 1))
    used: List[int] = []
    inconsistent: List[dict] = []

    for i in ids:
        if i in exclude_set:
            continue
        path = data_dir / str(i) / filename
        # also accept non-prefixed name
        if not path.is_file():
            alt = data_dir / str(i) / "oligo-highest-size.dat"
            path = alt if alt.is_file() else path
        if not path.is_file():
            print(f"[SKIP] missing {path}")
            continue

        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        if data.shape[0] > n_frames:
            inconsistent.append({i: int(data.shape[0])})
            data = data[:n_frames, :]
        elif data.shape[0] < n_frames:
            inconsistent.append({i: int(data.shape[0])})
            padded = np.zeros((n_frames, data.shape[1]))
            padded[: data.shape[0], :] = data
            data = padded
        all_data[:, i] = data[:, 1]
        used.append(i)

    return all_data, used, inconsistent


def classify_high_aggregation(
    all_data: np.ndarray,
    ids: Sequence[int],
    size_threshold: int = 5,
    tail_frames: int = 200,
    occupancy: float = 0.5,
) -> List[int]:
    """IDs with fraction of last ``tail_frames`` having size ≥ threshold ≥ occupancy.

    This is the Figure 8 “aggregate size ≥ 5” rule used historically:
    average occupancy of size≥5 over the final 0.2 µs ≥ 0.5.
    """
    selected: List[int] = []
    for i in ids:
        series = all_data[-tail_frames:, i]
        if series.size == 0:
            continue
        frac = float(np.count_nonzero(series >= size_threshold) / series.size)
        if frac >= occupancy:
            selected.append(int(i))
    return selected


def run_from_trajectory(
    topology: PathLike,
    trajectory: PathLike,
    outpath: PathLike,
    selection: str = "protein",
    min_distance: float = 5.0,
    expected_n_molecules: Optional[int] = 8,
) -> Path:
    """Convenience wrapper: load Universe and run ``aggregate_analysis``."""
    uni = mda.Universe(str(topology), str(trajectory))
    return aggregate_analysis(
        uni=uni,
        sel_all_atoms=selection,
        min_distance=min_distance,
        outpath=outpath,
        expected_n_molecules=expected_n_molecules,
    )


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Peptide oligomer size analysis: cutoff-based connected-component "
            "(single-linkage / friends-of-friends) clustering on a contact graph."
        )
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser(
        "compute",
        help=(
            "Compute oligomer sizes from a trajectory "
            f"[{CLUSTERING_ALGORITHM}; cutoff {DEFAULT_CONTACT_CUTOFF_A} Å]"
        ),
    )
    c.add_argument("-s", "--topology", required=True, help="Topology (.tpr/.pdb/...)")
    c.add_argument("-f", "--trajectory", required=True, help="Trajectory (.xtc/.trr/...)")
    c.add_argument("-o", "--outpath", required=True, help="Output directory")
    c.add_argument("--selection", default="protein")
    c.add_argument(
        "--min-distance",
        type=float,
        default=DEFAULT_CONTACT_CUTOFF_A,
        help=f"Contact cutoff in Å (default: {DEFAULT_CONTACT_CUTOFF_A})",
    )
    c.add_argument("--expected-n", type=int, default=8)

    k = sub.add_parser("classify", help="Classify high-aggregation trajectories from *.dat")
    k.add_argument("-d", "--data-dir", required=True, help="Root with <id>/analysisoligo-highest-size.dat")
    k.add_argument("--size-threshold", type=int, default=5)
    k.add_argument("--tail-frames", type=int, default=200)
    k.add_argument("--occupancy", type=float, default=0.5)
    k.add_argument("--exclude", type=int, nargs="*", default=[])
    k.add_argument(
        "-o",
        "--output",
        default="indices_Aggregate5andGreaterSize.txt",
        help="Output index list",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_cli().parse_args(argv)
    if args.cmd == "compute":
        run_from_trajectory(
            topology=args.topology,
            trajectory=args.trajectory,
            outpath=args.outpath,
            selection=args.selection,
            min_distance=args.min_distance,
            expected_n_molecules=args.expected_n,
        )
    elif args.cmd == "classify":
        all_data, used, _ = load_highest_size_library(
            args.data_dir, exclude=args.exclude
        )
        selected = classify_high_aggregation(
            all_data,
            used,
            size_threshold=args.size_threshold,
            tail_frames=args.tail_frames,
            occupancy=args.occupancy,
        )
        out = Path(args.output)
        out.write_text("\n".join(str(i) for i in selected) + ("\n" if selected else ""))
        print(f"Selected {len(selected)} / {len(used)} → {out.resolve()}")


if __name__ == "__main__":
    main()
