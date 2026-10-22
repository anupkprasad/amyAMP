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
**Composite written to:** `AMP_Data/energy_trajectory_plots.png` → used in `ment/Figures/fig5.png`

### Panels
| Panel | Content |
|-------|---------|
| (a) | Insertion depth vs time (GenSeq / amyAMP runs; peptide 22 highlighted) |
| (b) | Protein–membrane interaction energy vs time (`prot_mem_interaction.xvg`) |
| (c) | Violin of mean interaction energy: amyAMP vs Low/Inter/High MIC refs |

Also writes `depth_by_category.png` (LDB / IDB / HDB depth trajectories).

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
  --output-dir /path/to/out \
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
  -o oligomer_size_evolution.png \
  --exclude 24 59 84 97 \
  --write-indices indices_Aggregate5andGreaterSize.txt
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

| Item | Definition |
|------|------------|
| **Software** | MDAnalysis; NumPy; SciPy (`griddata`, `gaussian_filter`); Matplotlib |
| **Leaflets** | Upper/lower by median lipid COM *z* |
| **Surface** | Cubic `griddata` interpolation of leaflet COMs onto a regular XY grid (default 50×50) |
| **Thickness** | \(h(x,y)=\|z_u(x,y)-z_\ell(x,y)\|\) |
| **Mean curvature** | Monge-patch \(H\) via finite differences |
| **Smoothing** | Gaussian σ = 1.2 (grid units) on averaged curvature maps |

### Scripts
- `membrane_thickness_curvature.py` (this package)  
- Historical: `aug_24/membrane_prop.py`, `aug_24/lip_bilayer_curve_thikness.py`

### Data files to replot / inspect

```text
aug_24/membrane_analysis/
  membrane_analysis_data.csv
  membrane_curvature_data.csv
  membrane_heatmaps.npz
  membrane_*.png

aug_24/membrane_analysis_gneg/
aug_24/membrane_analysis_gpos/
```

To **recompute** maps from traj:

```bash
python -m scripts.simulation.membrane_thickness_curvature \
  -s topol.gro -f traj.xtc -o analysis/membrane_maps/ \
  --window both --analysis-time-ns 50
```

---

## R1Q8 — Fig. 8a quadrant membrane disruption (Fig. 7–style)

Stratified membrane-disruption trajectories (Energy / Aggregate / Both / Neither × Gram±, plus Membrane-Only control) live under:

```text
.../vijay/oct17/review_sim/CG_Paper_production_data_2026-10-14/
  Quadrant/{Gram-ve,Gram+ve}/{Energy,Aggregate,Both,Neither}/CG*/{primary,replicate_*}
  Membrane-Only/{primary,replicate_*}
```

Fig. 8a class ↔ folder: **Energy** = strong membrane interaction (< −600 kJ mol⁻¹); **Aggregate** = oligomer size ≥ 5; **Both** / **Neither** as labeled.

```bash
python -m scripts.simulation.review_quadrant_membrane_analysis \
  --data-root /media/anupkumar/BackupPlus/project/ML_project/pytorch_AMP_BiGAN/W_BiGAN_GP_monash/model_saved_231206/vijay/oct17/review_sim/CG_Paper_production_data_2026-10-14 \
  --analysis-time-ns 50 --window last
```

Writes `analysis_membrane_R1Q8/` next to the data (per-run maps, replicate-mean CSV, Fig. 7–style comparison PNGs). Notes gaps (no Gram−ve Energy set; Membrane-Only is Gram−ve–matched lipids only) in `ANALYSIS_NOTES.txt`.

---

## Quick figure → script → data map

| Figure | Replot script (this repo) | Primary data for replot |
|------|---------------------------|-------------------------|
| **5** | `pep_mem_energy.py` | `AMP_Data/{*.ods, trajectory_data/*/*.xvg, depth_data/**/depth.csv}` |
| **6** | `oligomer_plot.py` | `aggregate_data/*/analysisoligo-highest-size.dat` |
| **7 / S6** | `membrane_thickness_curvature.py` | traj+topol, or cached `membrane_analysis*/*.{csv,npz,png}` |
| **8** | `oligomer_analysis.py classify` (+ plot indices) | same `.dat` library as Fig. 6 |
| **R1Q8 / SI** | `review_quadrant_membrane_analysis.py` | `oct17/review_sim/CG_Paper_production_data_2026-10-14/` |
