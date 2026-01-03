"""
main.py
=======
Main controller script for the generative ML model project to generate peptides.

This script orchestrates three main tasks:
1. Model Training: Train the generative model using AMP and AMY datasets.
2. Peptide Generation: Generate novel peptides using the trained model.
3. Analysis: Analyze the generated peptides and compare with reference datasets.

Usage:
    Set the `task` variable to "train", "generate", or "analyze" and run the script.

"""

import os
import sys
import torch
from datetime import datetime

# Add project root to system path (relative to this file)
PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
if PATH_ROOT not in sys.path:
    sys.path.append(PATH_ROOT)

# Import project modules
from models_nn import train, model
from scripts import util

# ==============================================================================
# Configuration
# ==============================================================================

# Device configuration (CPU or GPU)
DEVICE = torch.device("cpu")  # Force CPU execution

# Paths (relative to project root)
PATH_MODEL = os.path.join(PATH_ROOT, "model_saved")
PATH_RESULT = os.path.join(PATH_ROOT, "results")
PATH_PC6 = os.path.join(PATH_ROOT, "data_master", "physical_chemical_6.txt")

# Training data paths
FASTA_AMPS = [os.path.join(PATH_ROOT, "data_master", "amps", "dbaasp", "dbaasp_APR_processed.fasta")]
FASTA_AMYS = [os.path.join(PATH_ROOT, "data_master", "amyloid", "amys_uniqueAI4AMP_processedtotrain.fasta")]

# Training hyperparameters
BATCH_SIZE = 128
EPOCHS = 7000
RUN_NUM = 2
FILTER_MAX = 100

# Generation parameters
BATCH_GENERATE = 1000


# ==============================================================================
# Dataset Preparation
# ==============================================================================

def prepare_dataset():
    """
    Prepare and encode the dataset for training.
    
    Returns:
        DataLoader: PyTorch DataLoader containing encoded sequences.
    """
    print("=" * 80)
    print("Preparing Dataset")
    print("=" * 80)
    
    # Load conversion table for PC6 encoding
    table = util.get_conversion_table(PATH_PC6)
    
    # Load sequences from FASTA files
    seqs = {}
    for fasta_file in FASTA_AMPS + FASTA_AMYS:
        cur_seq = util.read_fasta(fasta_file)
        print(f"Loaded {len(cur_seq)} sequences from {os.path.basename(fasta_file)}")
        seqs.update(cur_seq)
    
    print(f"Total sequences loaded: {len(seqs)}")
    
    # Encode sequences using PC6 encoding
    encoded = util.get_encoded_seqs(seqs, table)
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(torch.Tensor(encoded))
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True
    )
    
    print(f"Dataset prepared with batch size: {BATCH_SIZE}")
    return dataloader


# ==============================================================================
# Model Training
# ==============================================================================

def train_model(dataloader):
    """
    Train the generative model using the prepared dataset.
    
    Args:
        dataloader (DataLoader): PyTorch DataLoader containing training data.
    
    Returns:
        tuple: Trained generator (G), loss history (loss_all), collected sequences.
    """
    print("=" * 80)
    print("Starting Model Training")
    print("=" * 80)
    
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%H:%M:%S')}")
    
    # Train the model
    G, loss_all, collected_seqs = train.training(
        path_model=PATH_MODEL,
        epoch=EPOCHS,
        filter_max=FILTER_MAX,
        train_data=dataloader,
        run_num=RUN_NUM
    )
    
    finish_time = datetime.now()
    duration = finish_time - start_time
    print(f"Training finished at {finish_time.strftime('%H:%M:%S')}")
    print(f"Total training time: {duration}")
    
    return G, loss_all, collected_seqs


# ==============================================================================
# Peptide Generation
# ==============================================================================

def generate_peptides(output_path=None):
    """
    Generate novel peptides using the trained model.
    
    Loads the trained model, generates peptides, and saves them to a FASTA file.
    
    Args:
        output_path (str, optional): Custom path to save generated sequences.
                                     If None, uses default path in PATH_RESULT.
    """
    print("=" * 80)
    print("Generating Peptides")
    print("=" * 80)
    print(f"Using device: {DEVICE}")
    
    # Disable CUDA completely
    import os as os_module
    os_module.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices
    
    # FORCE CPU EVERYWHERE
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # Load conversion table
    table = util.get_conversion_table(PATH_PC6)
    
    # Initialize models with explicit CPU device
    with torch.no_grad():  # Prevent any gradient tracking
        G, E, D = model.get_model_and_optimizer()
        
        # Force CPU and disable CUDA
        E = E.cpu()
        G = G.cpu()
        D = D.cpu()
        
        # Manually move ALL parameters to CPU
        for param in E.parameters():
            param.data = param.data.cpu()
            param.requires_grad = False
        for param in G.parameters():
            param.data = param.data.cpu()
            param.requires_grad = False
        for param in D.parameters():
            param.data = param.data.cpu()
            param.requires_grad = False
    
    # Initialize weights
    E.apply(model.weights_init)
    G.apply(model.weights_init)
    D.apply(model.init_weights)
    
    # Initialize optimizers
    optimizer_EG = torch.optim.Adam(
        list(E.parameters()) + list(G.parameters()),
        lr=2e-5, betas=(0.5, 0.999), weight_decay=1e-5
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=2e-5, betas=(0.5, 0.999), weight_decay=1e-5
    )
    
    # Load trained model
    print(f"Loading model from {PATH_MODEL}...")
    all_params = train.load_model(
        PATH_MODEL, E, G, D, optimizer_EG, optimizer_D, RUN_NUM
    )
    E, G, D, optimizer_EG, optimizer_D, dataloader, loss_all, epoch_i = all_params
    
    # FORCE everything back to CPU after loading
    E = E.cpu()
    G = G.cpu()
    D = D.cpu()
    
    # Verify and force all parameters to CPU again
    for param in G.parameters():
        if param.is_cuda:
            param.data = param.data.cpu()
    
    # Set models to evaluation mode
    E.eval()
    G.eval()
    D.eval()
    
    # Generate random latent vectors - explicitly on CPU
    z = torch.randn(BATCH_GENERATE, FILTER_MAX, 1, 1, device='cpu')
    
    # Debug: Comprehensive device check
    print(f"Generator device: {next(G.parameters()).device}")
    print(f"Latent vector device: {z.device}")
    print(f"All G params on CPU: {all(p.device.type == 'cpu' for p in G.parameters())}")
    print(f"Any G params on CUDA: {any(p.is_cuda for p in G.parameters())}")
    
    # Generate sequences with comprehensive CPU enforcement
    print(f"Generating {BATCH_GENERATE} peptides...")
    
    # Create CPU-forced wrapper for generate_seqs
    generate_seqs_cpu = util.force_cpu_execution(util.generate_seqs)
    
    with torch.no_grad():
        generated_seqs = generate_seqs_cpu(G, table, z)
    
    # Determine output file path
    if output_path is None:
        output_file = os.path.join(PATH_RESULT, "sequence", f"seqs_generated{BATCH_GENERATE}.fasta")
    else:
        # If path is relative, make it absolute
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)
        
        output_file = output_path
        if os.path.isdir(output_file):
            output_file = os.path.join(output_file, f"seqs_generated{BATCH_GENERATE}.fasta")
    
    # Create directory if it doesn't exist (only if there's a directory component)
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create if dirname is not empty
        os.makedirs(output_dir, exist_ok=True)
    
    util.write_fasta(generated_seqs, output_file)
    
    print(f"Peptides generated successfully and saved to {output_file}")
    
    return output_file


# ==============================================================================
# Analysis
# ==============================================================================

def analyze_generated_data():
    """
    Analyze the generated peptides and compare with reference datasets.
    
    Note: This function requires the analysis module to be implemented.
    """
    print("=" * 80)
    print("Analyzing Generated Data")
    print("=" * 80)
    
    # Analysis module is currently commented out
    print("Analysis module not yet implemented.")
    print("Please refer to scripts/analysis_generated_seqs.py for analysis functions.")


# ==============================================================================
# Model Summary Generation
# ==============================================================================

def generate_model_summary():
    """
    Generate and save model architecture summaries for all models.
    
    Creates a text file with detailed architecture information for
    the Encoder, Generator, and Discriminator models.
    """
    print("=" * 80)
    print("Generating Model Architecture Summary")
    print("=" * 80)
    
    # Generate model summaries
    train.models_summary(
        filter_max=FILTER_MAX,
        path_model=PATH_MODEL,
        batch_size=BATCH_SIZE
    )
    
    print(f"Model architecture summary generated successfully!")
    print(f"Summary saved to: {os.path.join(PATH_MODEL, 'models_summary.txt')}")


# ==============================================================================
# Main Controller
# ==============================================================================

def main(task, output_path=None):
    """
    Main controller function to execute the specified task.
    
    Args:
        task (str): Task to perform - "train", "generate", "analyze", or "summary".
        output_path (str, optional): Custom output path for generated sequences (for "generate" task only).
    """
    print("\n")
    print("=" * 80)
    print("Generative ML Model for Peptide Generation")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Task: {task.upper()}")
    if output_path and task == "generate":
        print(f"Output path: {output_path}")
    print("=" * 80)
    print("\n")
    
    if task == "train":
        # Prepare dataset and train model
        dataloader = prepare_dataset()
        train_model(dataloader)
        
    elif task == "generate":
        # Generate peptides using trained model
        generate_peptides(output_path=output_path)
        
    elif task == "analyze":
        # Analyze generated peptides
        analyze_generated_data()
        
    elif task == "summary":
        # Generate model architecture summary
        generate_model_summary()
        
    else:
        print(f"ERROR: Invalid task '{task}'")
        print("Valid tasks: 'train', 'generate', 'analyze', 'summary'")
        sys.exit(1)
    
    print("\n")
    print("=" * 80)
    print(f"Task '{task.upper()}' completed successfully!")
    print("=" * 80)


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    # ===== CONFIGURE TASK HERE =====
    # Options: "train", "generate", "analyze", "summary"
    TASK = "generate"
    
    # Optional: Specify custom output path for generated sequences
    # If None, default path will be used: ~/workspace/amyAMP/results/sequence/
    # Examples:
    #   OUTPUT_PATH = "/home/anupkumar/custom_output/peptides.fasta"  # Full file path
    #   OUTPUT_PATH = "/home/anupkumar/custom_output/"                # Directory path
    #   OUTPUT_PATH = None                                            # Use default
    OUTPUT_PATH = None
    # ================================
    
    main(TASK, output_path=OUTPUT_PATH)





