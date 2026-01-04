# amyAMP: Amyloidogenic-Antimicrobial Peptide Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**amyAMP** is a BiGAN (Bidirectional Generative Adversarial Network) based machine learning model for generating novel amyloidogenic-antimicrobial peptides (AMPs). The model learns from both antimicrobial peptides and amyloidogenic sequences to generate peptides with dual functionality.

## 🎯 Features

- **BiGAN Architecture**: Encoder-Generator-Discriminator model with gradient penalty
- **PC6 Encoding**: 6-dimensional physicochemical property-based sequence encoding
- **Peptide Generation**: Generate thousands of novel peptides with desired properties
- **Comprehensive Analysis**: Built-in tools for t-SNE, PCA, UMAP visualization
- **Property Analysis**: Calculate and visualize physicochemical properties
- **Sequence Identity**: Compare generated peptides with reference datasets
- **CPU Optimized**: Stable execution on CPU hardware

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Generation](#generation)
  - [Analysis](#analysis)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Citation](#citation)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip3
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
# Clone to any directory you prefer
git clone https://github.com/anupkumar/amyAMP.git
cd amyAMP
```

### Step 2: Install Dependencies

```bash
pip3 install -r requirements.txt
```

**Note:** The project uses relative paths, so you can place it anywhere on your system.

### Step 3: Verify Installation

```bash
python3 -c "import torch; import Bio; print('✓ Installation successful!')"
```

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip3 install -r requirements.txt
```

## ⚡ Quick Start

### Generate 1000 Peptides (Easiest Way)

```bash
# Make the script executable
chmod +x generate_peptides.sh

# Generate peptides
./generate_peptides.sh
```

This creates `generated_peptides_1000.fasta` in the current directory.

**Note:** All paths are relative to the project directory, so the project is fully portable.

## 📖 Usage

### Training

Train the BiGAN model on your peptide datasets:

```python
# Edit main.py configuration
TASK = "train"
EPOCHS = 7000
RUN_NUM = 1  # First training run

# Run training
python3 main.py
```

**Configuration Options:**
- `BATCH_SIZE`: Training batch size (default: 128)
- `EPOCHS`: Number of training epochs (default: 7000)
- `FILTER_MAX`: Latent space dimension (default: 100)
- `RUN_NUM`: Training run number (1 for new, 2+ to resume)

### Generation

#### Method 1: Using Bash Script (Recommended)

```bash
# Generate 1000 peptides in current directory
./generate_peptides.sh

# Generate to specific file
./generate_peptides.sh my_peptides.fasta

# Generate to specific directory
./generate_peptides.sh ~/output/

# Generate custom number of peptides
./generate_peptides.sh output.fasta 5000

# Combine both
./generate_peptides.sh ~/results/ 2000
```

#### Method 2: Using Python

```python
# Edit main.py
TASK = "generate"
BATCH_GENERATE = 1000
OUTPUT_PATH = "/path/to/output.fasta"  # or None for default

# Run generation
python3 main.py
```

#### Method 3: Direct Function Call

```python
from main import generate_peptides

# Generate to specific location
output_file = generate_peptides(output_path="/path/to/output.fasta")
print(f"Generated peptides saved to: {output_file}")
```

### Analysis

#### Visualize Properties

```python
# Run property analysis
python3 scripts/properties_AMP_AMY_seperate.py
```

**Generates:**
- Violin plots for 9 key properties
- Positive charge distribution histograms
- Correlation matrices for each dataset

#### Dimensionality Reduction

```python
# Run t-SNE, PCA, UMAP analysis
python3 scripts/tSNE_seperate_AMP_AMY.py
```

**Generates:**
- 2D/3D PCA plots
- t-SNE visualizations
- UMAP projections
- Physicochemical property violin plots

#### Sequence Identity Analysis

```python
# Calculate percent identity across epochs
python3 scripts/percent_identity_seperate.py
```

**Generates:**
- Box plots showing identity evolution
- Histogram plots for final epoch
- Statistical comparisons with AMP, AMY, and random peptides

### Model Summary

Generate architecture summaries:

```python
# Edit main.py
TASK = "summary"

# Run
python3 main.py
```

Creates `model_saved/models_summary.txt` with detailed architecture information.

## 📁 Project Structure

```
amyAMP/
├── main.py                          # Main controller script
├── generate_peptides.sh             # Quick peptide generation script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── models_nn/                       # Neural network models
│   ├── model.py                     # BiGAN architecture
│   └── train.py                     # Training and loading functions
│
├── scripts/                         # Utility scripts
│   ├── util.py                      # Core utility functions
│   ├── plotStyle.py                 # Plotting configurations
│   ├── tSNE_seperate_AMP_AMY.py    # Dimensionality reduction
│   ├── properties_AMP_AMY_seperate.py  # Property analysis
│   └── percent_identity_seperate.py    # Sequence identity analysis
│
├── data_master/                     # Training data
│   ├── amps/                        # Antimicrobial peptides
│   ├── amyloid/                     # Amyloidogenic peptides
│   ├── physical_chemical_6.txt      # PC6 encoding table
│   └── random_peptides_1000.fasta   # Random control sequences
│
├── model_saved/                     # Trained models
│   ├── G.pkl                        # Generator model
│   ├── E.pkl                        # Encoder model
│   ├── D.pkl                        # Discriminator model
│   └── *.tar                        # Training checkpoints
│
└── results/                         # Output directory
    ├── sequence/                    # Generated FASTA files
    └── *.png                        # Analysis plots
```

## 💡 Examples

### Example 1: Train a New Model

```bash
# Prepare your data
# - Place AMP sequences in data_master/amps/
# - Place AMY sequences in data_master/amyloid/

# Edit main.py
TASK = "train"
EPOCHS = 7000
RUN_NUM = 1

# Start training
python3 main.py
```

### Example 2: Generate Peptides in Batch

```bash
#!/bin/bash
# Generate 5 batches of 1000 peptides each
for i in {1..5}; do
    ./generate_peptides.sh ~/output/batch_${i}/ 1000
done
```

### Example 3: Complete Workflow

```python
# 1. Train model
from main import main, prepare_dataset, train_model

dataloader = prepare_dataset()
train_model(dataloader)

# 2. Generate peptides
from main import generate_peptides
output = generate_peptides(output_path="~/results/novel_peptides.fasta")

# 3. Analyze properties
import os
os.system("python3 scripts/properties_AMP_AMY_seperate.py")
```

### Example 4: Custom Generation Parameters

```python
from main import main
import main as m

# Configure generation
m.BATCH_GENERATE = 5000
m.FILTER_MAX = 100

# Generate
main("generate", output_path="~/large_batch.fasta")
```

## 📊 Output Format

Generated peptides are saved in FASTA format:

```
>seq_num_1
KFGWLIPKAVGH
>seq_num_2
RLLIWKGFPQSA
>seq_num_3
GWFKLRIPHAQV
```

## 🔧 Configuration

### Path Configuration

The project now uses **relative paths** automatically. All paths are calculated relative to the script location, making the project portable across different systems and directories.

**No configuration needed!** Just place the project anywhere and run.

### Main Configuration (`main.py`)

```python
# Device
DEVICE = torch.device("cpu")

# Training
BATCH_SIZE = 128
EPOCHS = 7000
RUN_NUM = 2
FILTER_MAX = 100

# Generation
BATCH_GENERATE = 1000

# Paths are automatically set relative to project root
```

## 🐛 Troubleshooting

### CUDA/CPU Issues

If you encounter CUDA errors:

```python
# In main.py, ensure:
DEVICE = torch.device("cpu")

# And run with:
export CUDA_VISIBLE_DEVICES=""
python3 main.py
```

### Path Issues

If you encounter path-related errors:

```bash
# Ensure you're in the project directory
cd /path/to/amyAMP

# Run scripts from the project root
python3 main.py
python3 scripts/tSNE_seperate_AMP_AMY.py
```

### Moving the Project

The project is fully portable! You can move it to any location:

```bash
# Move project
mv amyAMP /new/location/

# Navigate to new location
cd /new/location/amyAMP

# Works immediately - no reconfiguration needed!
./generate_peptides.sh
```

### Model Not Found

```bash
# Check if model files exist
ls ~/workspace/amyAMP/model_saved/

# Should show: G.pkl, E.pkl, D.pkl, *.tar files
```

### Import Errors

```bash
# Reinstall dependencies
pip3 install --upgrade -r requirements.txt
```

## 📚 Documentation

- **Model Architecture**: See `models_nn/model.py` for BiGAN implementation
- **Training Details**: See `models_nn/train.py` for training loop
- **Utility Functions**: See `scripts/util.py` for encoding/decoding
- **Analysis Scripts**: See `scripts/` directory for visualization tools

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaborations:
- Email: anupkprasad121@gmail.com
- GitHub: [@anupkprasad](https://github.com/anupkprasad)

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- BioPython for sequence handling tools
- The peptide research community

## 📝 Citation

If you use amyAMP in your research, please cite:

```bibtex
@software{amyamp2024,
  author = {Anup K. Prasad},
  title = {amyAMP: BiGAN-based Amyloidogenic-Antimicrobial Peptide Generator},
  year = {2024},
  url = {https://github.com/anupkprasad/amyAMP}
}

**Made for the peptide research community**