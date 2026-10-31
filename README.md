# amyAMP: Amyloidogenic-Antimicrobial Peptide Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![GitHub stars](https://img.shields.io/github/stars/anupkprasad/amyAMP?style=social)](https://github.com/anupkprasad/amyAMP/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/anupkprasad/amyAMP?style=social)](https://github.com/anupkprasad/amyAMP/network/members)
[![GitHub issues](https://img.shields.io/github/issues/anupkprasad/amyAMP)](https://github.com/anupkprasad/amyAMP/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/anupkprasad/amyAMP)](https://github.com/anupkprasad/amyAMP/commits)

**amyAMP** is a BiGAN (Bidirectional Generative Adversarial Network) for de novo design of peptides that combine antimicrobial and amyloidogenic sequence features. The model is trained on curated antimicrobial (DBAASP) and amyloidogenic peptide sets after cross-property screening, and ships with pretrained weights for immediate generation.

> **Disclaimer:** Generated sequences are computational designs for research use only. They are **not** approved therapeutics. Antimicrobial or amyloid activity must be treated as hypotheses until experimentally validated. This software is provided under Apache-2.0 **without warranty**.

## Features

- **BiGAN architecture**: encoder–generator–critic with gradient penalty
- **PC6 encoding**: six physicochemical properties per residue
- **Ready-to-run generation**: pretrained models in `model_saved/`
- **Analysis scripts**: UMAP/PCA/t-SNE, property distributions, percent identity
- **CPU-friendly default**: generation runs on CPU out of the box

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+ (tested on 3.8–3.11)
- `pip` / `pip3`
- Git

### 1. Clone

```bash
git clone https://github.com/anupkprasad/amyAMP.git
cd amyAMP
```

### 2. Create an environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

**GPU (optional):** install a CUDA build of PyTorch from the [official instructions](https://pytorch.org/get-started/locally/), then install the remaining packages from `requirements.txt`.

### 3. Verify

```bash
python3 -c "import torch; from Bio import SeqIO; print('OK', torch.__version__)"
```

## Quick start

Generate 1000 peptides with the pretrained model:

```bash
chmod +x generate_peptides.sh
./generate_peptides.sh
```

Output: `generated_peptides_1000.fasta` in the project root.

```bash
# Custom file
./generate_peptides.sh my_peptides.fasta

# Custom directory and count
./generate_peptides.sh ./output/ 2000
```

## Usage

Edit the task flag near the bottom of `main.py`, then run:

```bash
python3 main.py
```

| Task | `TASK =` | Purpose |
|------|----------|---------|
| Train | `"train"` | Train BiGAN on bundled FASTA sets |
| Generate | `"generate"` | Sample peptides from a saved checkpoint |
| Summary | `"summary"` | Write `model_saved/models_summary.txt` |
| Analyze | `"analyze"` | Placeholder — use the scripts below |

### Training

In `main.py`:

```python
TASK = "train"
EPOCHS = 7000
RUN_NUM = 1      # 1 = new run; 2+ resumes from checkpoint naming
```

Default training FASTAs (already in the repo):

- `data_master/amps/dbaasp/dbaasp_APR_processed.fasta`
- `data_master/amyloid/amys_uniqueAI4AMP_processedtotrain.fasta`

Key hyperparameters in `main.py`:

- `BATCH_SIZE` (default `128`)
- `EPOCHS` (default `7000`)
- `FILTER_MAX` (latent width, default `100`)
- `RUN_NUM` (checkpoint index; generation expects run `2` for the shipped weights)

### Generation

**Bash (recommended):**

```bash
./generate_peptides.sh [output_path] [num_peptides]
```

**Python:**

```python
# in main.py
TASK = "generate"
BATCH_GENERATE = 1000
OUTPUT_PATH = None   # or "path/to/out.fasta"
```

```bash
python3 main.py
```

**Function call:**

```python
from main import generate_peptides
generate_peptides(output_path="novel_peptides.fasta")
```

### Analysis scripts

Run from the project root after generation / with the bundled result FASTAs:

```bash
# Physicochemical property distributions
python3 scripts/properties_AMP_AMY_separate.py

# Dimensionality reduction (PCA / t-SNE / UMAP)
python3 scripts/tSNE.py

# Percent identity vs reference sets
python3 scripts/percent_identity.py
```

Related helpers:

- `scripts/properties_AMP_AMY.py`
- `scripts/properties_AMP_AMY_separate_includingDiscard.py`
- `scripts/properties_PC6_separate_includingDiscard.py`
- `scripts/plot_fig4_heldout_validators.py` (AMPScanner / AmyloGram comparison figure)

Simulation analysis utilities live under `scripts/simulation/` (see that folder’s README).

## Project structure

```
amyAMP/
├── main.py                      # Train / generate / summary controller
├── generate_peptides.sh         # One-command generation wrapper
├── requirements.txt
├── LICENSE                      # Apache-2.0
├── models_nn/
│   ├── model.py                 # BiGAN modules
│   └── train.py                 # Training + checkpoint I/O
├── scripts/
│   ├── util.py                  # PC6 encode/decode, I/O helpers
│   ├── tSNE.py
│   ├── properties_AMP_AMY_separate.py
│   ├── percent_identity.py
│   └── simulation/              # CG MD post-processing helpers
├── data_master/
│   ├── amps/                    # AMP training / reference FASTAs
│   ├── amyloid/                 # AMY training / reference FASTAs
│   ├── physical_chemical_6.txt  # PC6 table
│   └── Input_files_simulations/ # Optional MD inputs
├── model_saved/                 # Pretrained G/E/D + checkpoints
└── results/                     # Generated FASTAs and analysis outputs
```

## Requirements

See [`requirements.txt`](requirements.txt). Core stack:

- `torch`, `torchvision`
- `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`, `plotly`
- `scikit-learn`, `umap-learn`
- `biopython`, `peptides`
- `torchsummary` (for architecture summary)

Minimum ~4 GB RAM; 8 GB+ recommended for training.

## Output format

```text
>seq_num_1
KFGWLIPKAVGH
>seq_num_2
RLLIWKGFPQSA
```

## Troubleshooting

**CUDA / device errors**

```bash
export CUDA_VISIBLE_DEVICES=""
python3 main.py
```

`main.py` defaults to CPU.

**Model not found**

```bash
ls model_saved/G.pkl model_saved/E.pkl model_saved/D.pkl
ls model_saved/modelsNoptimiser_state_dict_r2.tar
```

Generation uses `RUN_NUM = 2` to match the shipped checkpoint.

**Import / dependency errors**

```bash
pip install --upgrade -r requirements.txt
# or CPU PyTorch wheel:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**PC6 table not found**

Ensure `data_master/physical_chemical_6.txt` is present and run commands from the repo root.

## Citation

If you use amyAMP, please cite:

```bibtex
@article{prasad2025amyamp,
  author  = {Prasad, Anup K. and Patel, Manoj K. and Awatade, Vijay and Plisson, Fabien and Martin, Lisandra L. and Panwar, Ajay S.},
  title   = {Generative Deep Learning and Molecular Dynamics Reveal Design Principles for Amyloid-Like Antimicrobial Peptides},
  journal = {The Journal of Physical Chemistry B},
  year    = {2025},
  note    = {In revision / update with final citation when available},
  url     = {https://github.com/anupkprasad/amyAMP}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Contact

- Email: anupkprasad121@gmail.com
- GitHub: [@anupkprasad](https://github.com/anupkprasad)

## Acknowledgments

- PyTorch, BioPython, and the peptide / MD research communities
- Training data sources include DBAASP, WALTZ-DB, CARs-DB, and CPAD (see manuscript Methods)
