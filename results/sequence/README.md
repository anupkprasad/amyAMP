# Sequence files used for property / UMAP analyses

FASTA_FILES = [
    "seqs_generated_postprocessed.fasta",  ### amyAMP
    "trainPep.fasta",                      ### trainPep = AMP + AMY (below)
    "dbaasp_APR_processed_removedJZ.fasta",  ### AMP (trainPep part)
    "amys_uniqueAI4AMP_normalized.fasta",    ### AMY (trainPep part)
    "random_peptides_1000.fasta",            ### randPep
    "dbaasp_non_APR.fasta",                  ### AMP(D)
    "AMY_nonAntibacterial.fasta"             ### AMY(D)
]

## Notes

- **trainPep / `trainPep.fasta`**: Concatenation of
  `dbaasp_APR_processed_removedJZ.fasta` (AMP, 5786) +
  `amys_uniqueAI4AMP_normalized.fasta` (AMY, 1101) → **6887** sequences
  (same grouping as Fig. 3). No sequence-level deduplication.

- **AMP(D) / `dbaasp_non_APR.fasta`**: DBAASP AMPs without amyloid-prone regions (APR).
  Filtered to **standard amino acids only** (ACDEFGHIKLMNPQRSTVWY). Sequences containing
  `X` (DBAASP placeholder for unusual/non-coded residues; 1947 of 10155) were removed,
  leaving **8208** sequences. Pre-filter copy kept at
  `data_master/amps/dbaasp/dbaasp_non_APR_includingX.fasta`.
  Same filtered file is mirrored in `results/sequence/` and `data_master/amps/dbaasp/`.

- **AmyloGram scores** (`amylogram/`): batch predictions via the CRAN [AmyloGram](https://cran.r-project.org/package=AmyloGram)
  R package (same model as the web server). Run:
  `conda activate amylogram && Rscript scripts/run_amylogram_batch.R`
  Outputs one subdirectory per dataset (`amyAMP/`, `trainPep/`, `AMPD/`, `AMYD/`, `randPep/`)
  with `*_AmyloGram_scores.csv` (Name, Probability) plus `summary.csv`;
  aggregate table: `amylogram/summary_all_datasets.csv`.
  Sequences with length &lt; 6 are skipped (AmyloGram hexamer n-grams).
