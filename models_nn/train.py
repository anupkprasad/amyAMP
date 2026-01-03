"""
train.py
========
Model training, loading, and saving module for the generative ML model.

This module provides functionality to:
1. Train the Encoder-Generator-Discriminator (BiGAN) architecture
2. Save model checkpoints with optimizer states
3. Load pre-trained models for inference or continued training
4. Generate model architecture summaries

"""

import os
import torch
from models_nn import model
from scripts import util

# ==============================================================================
# Configuration
# ==============================================================================

# Device configuration
DEVICE = torch.device("cpu")  # Force CPU execution for stability

# Get project root (two levels up from this file)
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_PC6 = os.path.join(PATH_ROOT, "data_master", "physical_chemical_6.txt")


# ==============================================================================
# Model Checkpoint Management
# ==============================================================================

def save_model(path_model, E, G, D, optimizer_EG, optimizer_D, dataloader, 
               loss_all, collected_seqs, epoch, run_num):
    """
    Save complete model checkpoint including model states, optimizer states, and training data.
    
    Args:
        path_model (str): Directory path to save model files
        E (nn.Module): Encoder model
        G (nn.Module): Generator model
        D (nn.Module): Discriminator model
        optimizer_EG (Optimizer): Encoder-Generator optimizer
        optimizer_D (Optimizer): Discriminator optimizer
        dataloader (DataLoader): Training data loader
        loss_all (list): List of loss histories [L_D, L_EG, L_E, L_G]
        collected_seqs (dict): Dictionary of generated sequences by epoch
        epoch (int): Current epoch number
        run_num (int): Training run number (for multi-run tracking)
    
    Returns:
        None
    """
    print(f"Saving model checkpoint for run {run_num} at epoch {epoch + 1}...")
    
    # Save model and optimizer state dictionaries
    checkpoint_path = os.path.join(path_model, f'modelsNoptimiser_state_dict_r{run_num}.tar')
    torch.save({
        'E_state_dict': E.state_dict(),
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizer_EG_state_dict': optimizer_EG.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, checkpoint_path)
    
    # Save complete model objects (for easy loading)
    torch.save(G, os.path.join(path_model, "G.pkl"))
    torch.save(E, os.path.join(path_model, "E.pkl"))
    torch.save(D, os.path.join(path_model, "D.pkl"))
    torch.save(dataloader, os.path.join(path_model, 'dataloader.pt'))
    
    # Save training metadata and generated sequences
    metadata_path = os.path.join(path_model, f'collectedseqs_loss_epochinfo_r{run_num}.json')
    torch.save({
        'epoch': epoch + 1,
        'loss_all': loss_all,
        'collected_seqs': collected_seqs
    }, metadata_path)
    
    print(f"Model checkpoint saved successfully at {checkpoint_path}")


def load_model(path_model, E, G, D, optimizer_EG, optimizer_D, run_num):
    """
    Load a pre-trained model checkpoint with all associated states.
    
    Args:
        path_model (str): Directory path containing model files
        E (nn.Module): Encoder model (initialized)
        G (nn.Module): Generator model (initialized)
        D (nn.Module): Discriminator model (initialized)
        optimizer_EG (Optimizer): Encoder-Generator optimizer (initialized)
        optimizer_D (Optimizer): Discriminator optimizer (initialized)
        run_num (int): Training run number to load
    
    Returns:
        tuple: (E, G, D, optimizer_EG, optimizer_D, dataloader, loss_all, epoch)
            - E, G, D: Models with loaded weights
            - optimizer_EG, optimizer_D: Optimizers with loaded states
            - dataloader: Training data loader
            - loss_all: Historical loss data
            - epoch: Last completed epoch number
    """
    print(f"Loading model checkpoint from run {run_num - 1}...")
    
    # Load model and optimizer states with CPU mapping for compatibility
    checkpoint_path = os.path.join(path_model, f'modelsNoptimiser_state_dict_r{run_num - 1}.tar')
    stat = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Debug information
    print(f"Checkpoint keys: {list(stat.keys())}")
    print(f"Encoder keys (sample): {list(E.state_dict().keys())[:5]}")
    
    # Load dataloader
    dataloader = torch.load(os.path.join(path_model, 'dataloader.pt'), map_location=DEVICE)
    
    # Load state dictionaries for models and optimizers
    state_dict_mapping = {
        'E_state_dict': E,
        'G_state_dict': G,
        'D_state_dict': D,
        'optimizer_EG_state_dict': optimizer_EG,
        'optimizer_D_state_dict': optimizer_D
    }
    
    for key, module in state_dict_mapping.items():
        if key in stat:
            # Move models to device before loading state
            if key.endswith('_state_dict') and not key.startswith('optimizer'):
                module.to(DEVICE)
            module.load_state_dict(stat[key])
            print(f"Loaded {key}")
        else:
            print(f"Warning: {key} not found in checkpoint")
    
    # Load training metadata
    metadata_path = os.path.join(path_model, f'collectedseqs_loss_epochinfo_r{run_num - 1}.json')
    metadata = torch.load(metadata_path, map_location=DEVICE)
    loss_all = metadata['loss_all']
    epoch = metadata['epoch']
    
    print(f"Model loaded successfully. Resuming from epoch {epoch}")
    
    return E, G, D, optimizer_EG, optimizer_D, dataloader, loss_all, epoch


# ==============================================================================
# Model Training
# ==============================================================================

def training(path_model, epoch, train_data=None, batch_size=128, filter_max=320, run_num=None):
    """
    Train the BiGAN model (Encoder-Generator-Discriminator) with Gradient Penalty.
    
    Args:
        path_model (str): Directory to save model checkpoints
        epoch (int): Number of epochs to train
        train_data (DataLoader, optional): Training data loader
        batch_size (int): Batch size for training
        filter_max (int): Maximum filter size for latent space
        run_num (int): Training run number (1 for new training, >1 for resuming)
    
    Returns:
        tuple: (G, loss_all, collected_seqs)
            - G: Trained generator model
            - loss_all: List of loss histories [L_D, L_EG, L_E, L_G]
            - collected_seqs: Dictionary of generated sequences by epoch
    """
    print("=" * 80)
    print("Initializing Training")
    print("=" * 80)
    print(f"Run number: {run_num}")
    print(f"Epochs: {epoch}")
    print(f"Batch size: {batch_size}")
    print(f"Filter max: {filter_max}")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    
    # Initialize models
    G, E, D = model.get_model_and_optimizer(filter_max=filter_max)
    E = E.to(DEVICE)
    G = G.to(DEVICE)
    D = D.to(DEVICE)
    
    # Initialize weights
    E.apply(model.weights_init)
    G.apply(model.weights_init)
    D.apply(model.init_weights)
    
    # Initialize sequence collection and conversion table
    collected_seqs = {}
    table = util.get_conversion_table(PATH_PC6)
    
    # Initialize optimizers with weight decay for regularization
    optimizer_EG = torch.optim.Adam(
        list(E.parameters()) + list(G.parameters()),
        lr=2e-5, betas=(0.5, 0.999), weight_decay=1e-5
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=2e-5, betas=(0.5, 0.999), weight_decay=1e-5
    )
    
    # Initialize or load training state
    if run_num == 1:
        # New training session
        L_D, L_EG, L_E, L_G = [], [], [], []
        epoch_start = 0
        dataloader = train_data
        print("Starting fresh training session...")
    else:
        # Resume from checkpoint
        print(f"Resuming training from run {run_num - 1}...")
        all_params = load_model(path_model, E, G, D, optimizer_EG, optimizer_D, run_num)
        E, G, D, optimizer_EG, optimizer_D, dataloader, loss_all, epoch_start = all_params
        L_D, L_EG, L_E, L_G = loss_all
    
    epoch_end = epoch_start + epoch
    
    # ==============================================================================
    # Training Loop
    # ==============================================================================
    
    print("\n" + "=" * 80)
    print("Starting Training Loop")
    print("=" * 80 + "\n")
    
    for current_epoch in range(epoch_start, epoch_end):
        # Initialize epoch loss accumulators
        D_loss_acc = 0.0
        EG_loss_acc = 0.0
        E_loss_acc = 0.0
        G_loss_acc = 0.0
        
        # Set models to training mode
        D.train()
        E.train()
        G.train()
        
        # Iterate over batches
        for batch_idx, data in enumerate(dataloader):
            # Get real sequences
            real_seqs = data[0].to(DEVICE)
            
            # Sample random latent vectors
            z = torch.randn(batch_size, filter_max, 1, 1).to(DEVICE)
            
            # Forward pass
            Gz = G(z)  # Generate fake sequences
            EX = E(real_seqs)  # Encode real sequences
            
            # Compute losses
            loss_instance = model.loss_GP(D, real_seqs, EX, Gz, z)
            loss_D, loss_EG, loss_E, loss_G = loss_instance.loss()
            
            # Accumulate losses
            D_loss_acc += loss_D.item()
            EG_loss_acc += loss_EG.item()
            E_loss_acc += loss_E.item()
            G_loss_acc += loss_G.item()
            
            # Update Discriminator
            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()
            
            # Update Encoder and Generator
            optimizer_EG.zero_grad()
            loss_EG.backward()
            optimizer_EG.step()
        
        # ==============================================================================
        # Logging and Checkpointing
        # ==============================================================================
        
        num_batches = len(dataloader)
        
        # Log losses every 10 epochs
        if (current_epoch + 1) % 10 == 0:
            avg_loss_D = D_loss_acc / num_batches
            avg_loss_EG = EG_loss_acc / num_batches
            
            print(f'Epoch [{current_epoch + 1}/{epoch_end}], '
                  f'Avg_Loss_D: {avg_loss_D:.4f}, '
                  f'Avg_Loss_EG: {avg_loss_EG:.4f}')
            
            # Append to loss history
            L_D.append(avg_loss_D)
            L_EG.append(avg_loss_EG)
            L_E.append(E_loss_acc / num_batches)
            L_G.append(G_loss_acc / num_batches)
        
        # Generate and collect sequences every 100 epochs
        if (current_epoch + 1) % 100 == 0:
            with torch.no_grad():
                # Set models to evaluation mode
                D.eval()
                E.eval()
                G.eval()
                
                # Ensure generator is on correct device before generation
                G = G.to(DEVICE)
                
                # Create fresh latent vectors on the same device as the model
                z_gen = torch.randn(batch_size, filter_max, 1, 1).to(DEVICE)
                
                # Generate sequences
                generated_seqs = util.generate_seqs(G, table, z_gen, current_epoch)
                collected_seqs[current_epoch] = generated_seqs
                
                # Save latest generated sequences
                output_path = os.path.join(path_model, "final_generated_seq.fasta")
                util.write_fasta(generated_seqs, output_path)
                
                print(f"Generated {len(generated_seqs)} sequences at epoch {current_epoch + 1}")
                
                # Set models back to training mode
                D.train()
                E.train()
                G.train()
    
    # ==============================================================================
    # Save Final Model
    # ==============================================================================
    
    loss_all = [L_D, L_EG, L_E, L_G]
    save_model(path_model, E, G, D, optimizer_EG, optimizer_D, 
               dataloader, loss_all, collected_seqs, current_epoch, run_num)
    
    print("\n" + "=" * 80)
    print("Training Completed Successfully")
    print("=" * 80)
    
    return G, loss_all, collected_seqs


# ==============================================================================
# Model Summary
# ==============================================================================

def models_summary(filter_max, path_model, batch_size=128):
    """
    Generate and save architecture summaries for all models.
    
    Args:
        filter_max (int): Maximum filter size for latent space
        path_model (str): Directory to save summary file
        batch_size (int): Batch size for summary generation
    
    Returns:
        None
    """
    print("=" * 80)
    print("Generating Model Summaries")
    print("=" * 80)
    
    # Force CPU everywhere
    import os as os_module
    os_module.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices
    torch.set_default_tensor_type(torch.FloatTensor)  # Force CPU tensors
    
    # Initialize models on CPU explicitly
    G = model.Generator(filter_max).cpu()
    E = model.Encoder(filter_max).cpu()
    D = model.Discriminator().cpu()
    
    # Force all parameters to CPU
    for param in G.parameters():
        param.data = param.data.cpu()
    for param in E.parameters():
        param.data = param.data.cpu()
    for param in D.parameters():
        param.data = param.data.cpu()
    
    try:
        from torchsummary import summary_string
        
        # Generate summaries with explicit device specification
        result_g, _ = summary_string(G, input_size=(filter_max, 1, 1), batch_size=batch_size, device='cpu')
        result_e, _ = summary_string(E, input_size=(1, 30, 6), batch_size=batch_size, device='cpu')
        result_d, _ = summary_string(D, [(1, 30, 6), (filter_max, 1, 1)], batch_size=batch_size, device='cpu')
        
        # Save to file
        summary_path = os.path.join(path_model, 'models_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL ARCHITECTURE SUMMARIES\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("\n####### ENCODER #######\n\n")
            f.write(result_e)
            
            f.write("\n\n####### GENERATOR #######\n\n")
            f.write(result_g)
            
            f.write("\n\n####### DISCRIMINATOR #######\n\n")
            f.write(result_d)
        
        print(f"Model summaries saved to: {summary_path}")
        print(result_e)  # Print encoder summary to console
        
    except Exception as e:
        print(f"Error generating model summary with torchsummary: {e}")
        print("Falling back to manual summary generation...")
        
        # Manual summary generation as fallback
        summary_path = os.path.join(path_model, 'models_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL ARCHITECTURE SUMMARIES\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("\n####### ENCODER #######\n\n")
            f.write(str(E))
            f.write(f"\nTotal parameters: {sum(p.numel() for p in E.parameters())}\n")
            
            f.write("\n\n####### GENERATOR #######\n\n")
            f.write(str(G))
            f.write(f"\nTotal parameters: {sum(p.numel() for p in G.parameters())}\n")
            
            f.write("\n\n####### DISCRIMINATOR #######\n\n")
            f.write(str(D))
            f.write(f"\nTotal parameters: {sum(p.numel() for p in D.parameters())}\n")
        
        print(f"Manual model summaries saved to: {summary_path}")
        print(str(E))
