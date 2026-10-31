# Simulation analysis (`scripts/simulation`)

Modular CG MD post-processing for amyAMP **Figures 5–8** (and SI membrane panels).

## Canonical data root (external drive)

Most original analysis scripts and precomputed files live on BackupPlus under Vijay’s `aug_24` folder:

```text
/media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/
  W_BiGAN_GP_monash/model_saved_231206/vijay/aug_24/
```

Key subfolders:

| Path under `aug_24/` | Role |
|----------------------|------|
| `AMP_Data/` | Fig. 5 energy/depth replot inputs + `energy_trajectory_plots.png` |
| `aggregate_data/` | Fig. 6 / 8 oligomer `analysisoligo-*.dat` library |
| `membrane_analysis/`, `membrane_analysis_gneg/`, `membrane_analysis_gpos/` | Fig. 7 / S6 thickness–curvature outputs |
| `Gram -ve/`, `gram_positive/`, `gram+ve/` | Raw / organized CG trajectories (binding, aggregation, disruption) |
| `pap_insertion_cal.py` | Builds per-run `depth.csv` from traj (upstream of Fig. 5) |
| `aggre_plot.py`, `aggregate_analysis.py` | Historical Fig. 6 / 8 pipeline |
| `membrane_prop.py`, `lip_bilayer_curve_thikness.py` | Historical Fig. 7 / S6 pipeline |

Peptide index for the 100 MD systems: `AMP_Data/GenSeq.ods` (also copied in-repo as `results/seq_validation/GenSeq.ods`).

---

## Figure 5 — peptide–membrane energy / insertion

**Script (copy in this package):** `pep_mem_energy.py`  
**Original:** `aug_24/pep_mem_energy.py`  
**Composite written to:** `results/simulation/peptide_membrane_binding/figures/peptide_membrane_energy.png`

### Panels
| Panel | Content |
|-------|---------|
| (a) | Insertion depth vs time (GenSeq / amyAMP runs; peptide 22 highlighted) |
| (b) | Protein–membrane interaction energy vs time (`prot_mem_interaction.xvg`) |
| (c) | Violin of mean interaction energy: amyAMP vs Low/Inter/High MIC refs |

Also writes `insertion_depth_by_class.png` (LDB / IDB / HDB depth trajectories).

### Data files to replot

```text
AMP_Data/
  GenSeq.ods, LowMIC.ods, InterMIC.ods, HighMIC.ods
  trajectory_data/{1..100}/prot_mem_interaction.xvg
  depth_data/GenSeq/{1..100}/depth.csv
  depth_data/LDB/LDB{1..49}/depth.csv
  depth_data/IDB/IDB{1..49}/depth.csv
  depth_data/HDB/HDB{1..49}/depth.csv
```

Upstream depth extraction: `aug_24/pap_insertion_cal.py` → `depth.csv`  
(`time`, depth / `Z_diff` style columns; Fig. 5 reader uses the 2nd numeric column when present).

### Replot

```bash
python -m scripts.simulation.pep_mem_energy \
  --amp-data-dir /media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/W_BiGAN_GP_monash/model_saved_231206/vijay/aug_24/AMP_Data \
  --output-dir results/simulation/peptide_membrane_binding/figures \
  --no-show
```

---

## Figures 6 and 8 — oligomer / aggregate size

| Step | Implementation |
|------|----------------|
| **Clustering algorithm** | Cutoff-based **connected-component clustering** (single-linkage / friends-of-friends) on an undirected peptide–peptide contact graph (`networkx`) |
| **Contact cutoff** | Min bead–bead distance ≤ **5.0 Å (0.5 nm)** between peptide molecules |
| Size metric | Largest oligomer size per frame |
| Fig. 8 threshold | In the **final 200 ns**, ≥ **50%** of frames have size ≥ **5** |

### Scripts
- `oligomer_analysis.py` — compute from traj, or classify high-aggregation IDs  
- `oligomer_plot.py` — Fig. 6–style time series + early/late size fractions  

Historical: `aug_24/aggregate_analysis.py`, `aug_24/aggre_plot.py`.

### Data files to replot (Fig. 6 / 8)

```text
aug_24/aggregate_data/
  {1..100}/
    analysisoligo-highest-size.dat      # primary for plotting / Fig. 8 classify
    analysisoligo-block-average.dat
    analysisoligomer-groups.dat
    analysisoligomer-states.dat
```

(Aliases without the `analysis` prefix are also accepted: `oligo-highest-size.dat`, ….)

Raw trajs (to **recompute** `.dat`): per-peptide `topol.tpr` / `traj.xtc` under the Gram± / CGMartini trees (not required if `.dat` library is present).

### Compute from a trajectory

```bash
python -m scripts.simulation.oligomer_analysis compute \
  -s topol.tpr -f traj.xtc -o analysis/peptide_22/ \
  --min-distance 5.0 --selection protein
```

### Plot library of precomputed `analysisoligo-highest-size.dat`

```bash
python -m scripts.simulation.oligomer_plot \
  -d /media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/W_BiGAN_GP_monash/model_saved_231206/vijay/aug_24/aggregate_data \
  -o results/simulation/peptide_self_assembly/figures/oligomer_size_evolution.png \
  --exclude 24 59 84 97 \
  --write-indices results/simulation/peptide_self_assembly/tables/high_aggregation_indices.txt
```

### Classify high-aggregation IDs only (Fig. 8 rule)

```bash
python -m scripts.simulation.oligomer_analysis classify \
  -d /media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/W_BiGAN_GP_monash/model_saved_231206/vijay/aug_24/aggregate_data \
  --size-threshold 5 --tail-frames 200 --occupancy 0.5 \
  -o indices_Aggregate5andGreaterSize.txt
```

---

## Figures 7 / S6 — membrane thickness and mean curvature

Figure 7 is the Gram−ve composite. Figure S6 is the same layout on Gram+ve. Top to bottom: High-MIC analog (CG9), Intermediate-MIC analog (CG44), Low-MIC analog (CG40).

| Figure | Run | Window |
|--------|-----|--------|
| 7 | `Gram -ve/CG9_9` | 9950–9999 ns |
| 7 | `Gram -ve/CG44_9` | 5850–5899 ns |
| 7 | `Gram -ve/CG40_9` | 9950–9999 ns |
| S6 | `gram+ve/CG9` | 9950–9999 ns |
| S6 | `gram+ve/CG44` | 9950–9999 ns |
| S6 | `gram+ve/CG40` | 9950–9999 ns |

| Item | Definition |
|------|------------|
| **Software** | MDAnalysis; NumPy; SciPy (`griddata`, `gaussian_filter`); Matplotlib |
| **Leaflets** | Upper/lower by median headgroup *z* (PO4, NH3 of POPE/POPG) |
| **Thickness** | Headgroup-to-headgroup \(h(x,y)=\|z_u-z_\ell\|\), color scale 38–43 Å |
| **Mean curvature** | Monge-patch \(H\) of the lipid centre-of-mass surface; plotted as \(H\times 100\), scale −2 to 2 |
| **Smoothing** | Gaussian σ = 1.2 on the averaged curvature map |

### Script
- `membrane_disruption_analysis.py` — the six windows above → `results/simulation/membrane_disruption/`
- `membrane_thickness_curvature.py` — one trajectory, first/last window

### Recompute

```bash
python -m scripts.simulation.membrane_disruption_analysis \
  --data-root /media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/W_BiGAN_GP_monash/model_saved_231206/vijay/aug_24
```

---

## R1Q8 — Fig. 8a quadrant membrane disruption (Fig. 7–style)

Stratified membrane-disruption trajectories (Energy / Aggregate / Both / Neither × Gram±, plus Membrane-Only control) live under:

```text
.../vijay/oct17/review_sim/CG_Paper_production_data_2026-10-14/
  Quadrant/{Gram-ve,Gram+ve}/{Energy,Aggregate,Both,Neither}/CG*/{primary,replicate_*}
  Membrane-Only/{gram-ve_membrane,gram+ve_membrane}/{primary,replicate_*}
```

Fig. 8a class ↔ folder: **Energy** = strong membrane interaction (< −600 kJ mol⁻¹); **Aggregate** = oligomer size ≥ 5; **Both** / **Neither** as labeled.

```bash
python -m scripts.simulation.review_quadrant_membrane_analysis \
  --data-root /media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/W_BiGAN_GP_monash/model_saved_231206/vijay/oct17/review_sim/CG_Paper_production_data_2026-10-14 \
  --analysis-time-ns 50 --window last
```

Writes small analysis products to `results/simulation/stratified_membrane_disruption/` (figures, tables, and per-run grid maps). Trajectories stay on the data drive. Gram−ve and Gram+ve maps use one shared thickness and curvature color scale. See `results/simulation/README.md`.

---

## Quick figure → script → data map

| Figure | Replot script (this repo) | Primary data for replot |
|------|---------------------------|-------------------------|
| **5** | `pep_mem_energy.py` | `AMP_Data/{*.ods, trajectory_data/*/*.xvg, depth_data/**/depth.csv}` |
| **6** | `oligomer_plot.py` | `aggregate_data/*/analysisoligo-highest-size.dat` |
| **7 / S6** | `membrane_disruption_analysis.py` | `Gram -ve/{CG9_9,CG44_9,CG40_9}` and `gram+ve/{CG9,CG44,CG40}` at the windows in the table above |
| **8** | `oligomer_analysis.py classify` (+ plot indices) | same `.dat` library as Fig. 6 |
| **R1Q8 / SI** | `review_quadrant_membrane_analysis.py` | `oct17/review_sim/CG_Paper_production_data_2026-10-14/` |
