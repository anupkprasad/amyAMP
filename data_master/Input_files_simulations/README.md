# CG MD input files

Inputs for the amyAMP coarse-grained simulations (MARTINI 2.2), deposited with the public repository.

```text
Input_files_simulations/
  Input_parameters/     # GROMACS .mdp files
    Aggregation/        # peptide self-assembly (eight peptides in water)
    Single_Peptide/     # peptide–membrane binding (one peptide + bilayer)
    Membrane_Agg/       # membrane disruption (nine peptides + bilayer)
  Initial_structures/   # AlphaFold all-atom starting coordinates (.pdb)
    GenSeq/             # amyAMP-generated peptides used in MD (n = 100)
    LowMIC/             # low-MIC reference peptides (n = 49)
    InterMIC/           # intermediate-MIC reference peptides (n = 50)
    HighMIC/            # high-MIC reference peptides (n = 49)
```

## Parameter files

**GROMACS parameter files (`.mdp`)** for minimization, equilibration, and production.

| Protocol | Steps covered |
|----------|---------------|
| `Input_parameters/Aggregation/` | `step4.0`–`4.1` minimization, `step4.2` equilibration, `step5` production |
| `Input_parameters/Single_Peptide/` | `step6.0`–`6.1` minimization, `step6.2`–`6.6` equilibration, `step7` production |
| `Input_parameters/Membrane_Agg/` | `step6.0`–`6.1` minimization, `step6.2`–`6.6` equilibration, `step7` production |

## Initial structures

All-atom peptide coordinates used as the starting models for the CHARMM-GUI / MARTINI workflow. PDB filenames follow the ODS index for each set (248 structures; see `Initial_structures/conversion_manifest.csv`).

These coordinates were generated with AlphaFold. Secondary structure was assigned with DSSP through CHARMM-GUI and encoded in the MARTINI 2.2 topology.

## Analysis scripts

Post-processing used for Figures 5–8 lives under `scripts/simulation/` (see that folder’s README), including `membrane_thickness_curvature.py` for thickness and mean-curvature maps.
