"""Simulation analysis utilities for amyAMP CG MD trajectories."""

from .oligomer_analysis import (
    CLUSTERING_ALGORITHM,
    DEFAULT_CONTACT_CUTOFF_A,
    aggregate_analysis,
    classify_high_aggregation,
    load_highest_size_library,
)
from .oligomer_plot import plot_oligomer_size_evolution
from .membrane_thickness_curvature import (
    SOFTWARE_TOOLS as MEMBRANE_SOFTWARE_TOOLS,
    analyze_membrane_properties,
    mean_curvature_monge,
    thickness_map,
)

__all__ = [
    "CLUSTERING_ALGORITHM",
    "DEFAULT_CONTACT_CUTOFF_A",
    "MEMBRANE_SOFTWARE_TOOLS",
    "aggregate_analysis",
    "analyze_membrane_properties",
    "classify_high_aggregation",
    "load_highest_size_library",
    "mean_curvature_monge",
    "plot_oligomer_size_evolution",
    "thickness_map",
]
