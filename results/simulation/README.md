# Simulation analysis results

Small analysis products from the coarse-grained MD trajectories. Trajectories and other large run files stay on the external data drive. Scripts that rebuild these files live in `scripts/simulation/`.

```text
results/simulation/
  peptide_membrane_binding/        one peptide on a bilayer: energy and insertion depth
  peptide_self_assembly/           eight peptides in water: oligomer size
  membrane_disruption/             nine peptides on a bilayer: thickness and curvature
  stratified_membrane_disruption/  Fig. 8a classes × Gram±, plus peptide-free controls
```

## Peptide–membrane binding

`peptide_membrane_binding/figures/`

| File | Content |
|------|---------|
| `peptide_membrane_energy.png` | Insertion depth, protein–membrane energy, and energy violin |
| `insertion_depth_by_class.png` | Depth trajectories for low-, intermediate-, and high-MIC references |

```bash
python -m scripts.simulation.pep_mem_energy \
  --amp-data-dir /media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/W_BiGAN_GP_monash/model_saved_231206/vijay/aug_24/AMP_Data \
  --output-dir results/simulation/peptide_membrane_binding/figures \
  --no-show
```

## Peptide self-assembly

`peptide_self_assembly/figures/oligomer_size_evolution.png`  
`peptide_self_assembly/tables/high_aggregation_indices.txt` — peptides with oligomer size ≥ 5 in ≥ 50% of the final 200 ns (87 of 96 plotted trajectories; IDs 24, 59, 84, and 97 excluded).

```bash
python -m scripts.simulation.oligomer_plot \
  -d /media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/W_BiGAN_GP_monash/model_saved_231206/vijay/aug_24/aggregate_data \
  -o results/simulation/peptide_self_assembly/figures/oligomer_size_evolution.png \
  --exclude 24 59 84 97 \
  --write-indices results/simulation/peptide_self_assembly/tables/high_aggregation_indices.txt
```

## Membrane disruption

Thickness and curvature for the nine-peptide runs in Figure 7 (Gram−ve) and Figure S6 (Gram+ve). Each figure is a vertical stack, top to bottom: High-MIC analog (CG9), Intermediate-MIC analog (CG44), Low-MIC analog (CG40).

| Figure | Membrane | Peptide | Window |
|--------|----------|---------|--------|
| 7 | Gram−ve | CG9 | 9950–9999 ns |
| 7 | Gram−ve | CG44 | 5850–5899 ns |
| 7 | Gram−ve | CG40 | 9950–9999 ns |
| S6 | Gram+ve | CG9 | 9950–9999 ns |
| S6 | Gram+ve | CG44 | 9950–9999 ns |
| S6 | Gram+ve | CG40 | 9950–9999 ns |

Thickness is the PO4/NH3 headgroup separation. Displayed curvature is mean curvature times 100. Both figures use the paper color scales (38–43 Å and −2 to 2).

```text
membrane_disruption/
  figures/thickness_curvature_Gram-ve.png
  figures/thickness_curvature_Gram+ve.png
  tables/summary.csv
  maps/{Gram-ve|Gram+ve}/{CG9|CG40|CG44}/membrane_maps_<window>.npz
  NOTES.txt
```

```bash
python -m scripts.simulation.membrane_disruption_analysis \
  --data-root /media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/W_BiGAN_GP_monash/model_saved_231206/vijay/aug_24
```

## Stratified membrane disruption

Same thickness and curvature analysis for one example from each Fig. 8a class (`Energy`, `Aggregate`, `Both`, `Neither`) on both membranes, compared with a peptide-free control. Gram−ve and Gram+ve maps share one color scale.

```text
stratified_membrane_disruption/
  figures/   {Gram-ve|Gram+ve}_primary_…__rep1_membrane-only_rep2_CG55,CG94,CG24,CG40.png
  tables/    run inventory, per-run summary, replicate means
  maps/      {Gram-ve|Gram+ve}/{quadrant}/{peptide}/{replicate}/
  NOTES.txt
```

```bash
python -m scripts.simulation.review_quadrant_membrane_analysis \
  --data-root /media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/W_BiGAN_GP_monash/model_saved_231206/vijay/oct17/review_sim/CG_Paper_production_data_2026-10-14 \
  --analysis-time-ns 50 --window last
```

Existing files under `maps/` are reused. Pass `--force` to rebuild them from the trajectories.
