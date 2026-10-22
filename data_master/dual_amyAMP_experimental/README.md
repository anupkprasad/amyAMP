# Experimentally reported dual-function amyloid / antimicrobial peptides

Curated set of peptides/proteins for which **both** amyloid-related behaviour
and antimicrobial (or antiviral) activity have been reported experimentally
(not prediction-only).

**Current size: 44 entries** (expanded 2026-10-16).

## Inclusion criteria

A sequence is included if peer-reviewed experimental work supports **each** of:

1. **Amyloid side**: fibril / amyloid-like aggregates (ThT, Congo red, TEM/AFM,
   XRD, cryo-EM, CD β-sheet, etc.), **or** it is a canonical disease amyloid.
2. **Antimicrobial side**: MIC / killing / agglutination / antiviral assays,
   **or** the molecule is a well-established AMP with amyloid self-assembly data.

Length is **not** restricted (short peptides and full proteins both allowed).

## Evidence classes

| Class | Meaning | Count (approx.) |
|-------|---------|-----------------|
| `AMY_with_AMP` | Disease / host amyloid with experimental antimicrobial activity | 10 |
| `AMP_with_amyloid` | Known AMP with experimental amyloid-like fibrils (ThT/TEM/…) | 33 |
| `fragment` | Active fragment of a dual-function parent | 1 |

## Files

| File | Description |
|------|-------------|
| `dual_amyAMP_experimental.fasta` | Sequences (standard AA alphabet) |
| `dual_amyAMP_experimental_metadata.tsv` | Full metadata + evidence notes |
| `REFERENCES.md` | Bibliography |

## Expansion sources (this version)

**(a) More AMPs with ThT/TEM amyloid data** — from ChemPlusChem 2022 review tables,
bioRxiv ffAMP screen (Salinas/Landau lab; TEM ± ThT ± XRD for 14 natural AMPs),
and primary papers (PG-1, uperin, dermaseptin S9, indolicidin, PSMs, …).

**(b) More disease amyloids with MIC / antibacterial data** — Aβ40/42 (MIC tables),
APP loop peptides NWC15/NWC20, hIAPP, α-synuclein, SAA, β2-microglobulin
(aggregate-dependent antibacterial), human calcitonin (amyloid + dual literature).

## Caveats

- Not an exhaustive dump of APD/DBAASP × WALTZ-DB (those DBs are not dual-tagged).
- C-terminal amidation and disulfide pairing often present *in vivo* but absent from
  linear FASTA (noted in metadata).
- Some entries (e.g. humanin, buforin II) have weaker / mixed dual evidence — see
  `notes` column.
- Predicted-only overlaps that **failed** experimental MIC (Byszuk *et al.* 2023)
  remain excluded.
- University library access would help pull paywalled supplementary sequence tables
  for a further systematic expansion.

Compiled for the amyAMP project.
