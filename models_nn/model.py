"""
model.py
========
BiGAN (Bidirectional Generative Adversarial Network) architecture for peptide generation.

This module implements a BiGAN with Wasserstein loss and gradient penalty (WGAN-GP)
for generating novel peptide sequences. The architecture consists of:

1. Encoder (E): Maps real peptide sequences to latent space
2. Generator (G): Generates peptide sequences from latent vectors
3. Discriminator (D): Distinguishes between real and generated (sequence, latent) pairs

The model uses:
- PC6 encoding: 6-dimensional physicochemical property representation
- Gradient Penalty: Improved training stability (WGAN-GP)
- Convolutional architecture: Captures sequential patterns in peptides

Author: Anup K. Prasad
Date: 2024
"""

import torch
from torch import nn
import torch.autograd as autograd


# ==============================================================================
# Weight Initialization Functions
# ==============================================================================

def weights_init(m):
    """
    Initialize weights for convolutional and batch normalization layers.
    
    Uses normal distribution initialization:
    - Conv layers: mean=0.0, std=0.02
    - BatchNorm layers: mean=1.0, std=0.02, bias=0
    
    Args:
        m (nn.Module): Layer to initialize
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)


def init_weights(layer):
    """
    Initialize weights for linear layers.
    
    Uses normal distribution for weights and constant for bias:
    - Weight: mean=0, std=0.02
    - Bias: constant=0
    
    Args:
        layer (nn.Module): Layer to initialize
    """
    name = layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(layer.weight, mean=0, std=0.02)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)


# ==============================================================================
# Building Blocks
# ==============================================================================

class TransposedCBR(nn.Module):
    """
    Transposed Convolution + Batch Normalization + ReLU block.
    
    Used in the Generator for upsampling latent vectors to sequence space.
    
    Args:
        in_dim (int): Input channels
        out_dim (int): Output channels
        kernel (tuple): Kernel size (height, width)
        stride (tuple): Stride (height, width)
        padding (tuple): Padding (height, width)
    """
    def __init__(self, in_dim, out_dim, kernel, stride, padding):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        """Forward pass through the block."""
        return self.model(x)


class CRBlock(nn.Module):
    """
    Convolution + LeakyReLU block.
    
    Used in the Encoder for downsampling sequence space to latent vectors.
    
    Args:
        in_dim (int): Input channels
        out_dim (int): Output channels
        kernel (tuple): Kernel size (height, width)
        stride (tuple): Stride (height, width)
        padding (tuple): Padding (height, width)
        bias (bool): Whether to use bias in convolution
    """
    def __init__(self, in_dim, out_dim, kernel, stride, padding, bias=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, padding, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        """Forward pass through the block."""
        return self.model(x)


# ==============================================================================
# Encoder
# ==============================================================================

class Encoder(nn.Module):
    """
    Encoder network for mapping peptide sequences to latent space.
    
    Architecture:
    - Input: (batch, 1, 30, 6) - PC6 encoded peptide sequences
    - Output: (batch, 100, 1, 1) - Latent representation
    
    The encoder uses convolutional layers to progressively reduce spatial
    dimensions while increasing feature depth, capturing hierarchical patterns
    in peptide sequences.
    
    Args:
        in_dim (int): Input feature dimension (default: 6 for PC6 encoding)
        hidden_num (int): Base number of hidden features (default: 64)
    
    Returns:
        torch.Tensor: Latent representation of shape (batch, 100, 1, 1)
    """
    def __init__(self, in_dim=6, hidden_num=64):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1: (1, 30, 6) -> (64, 30, 1)
            CRBlock(1, hidden_num, (1, in_dim), (1, 1), (0, 0), bias=False),
            
            # Layer 2: (64, 30, 1) -> (128, 15, 1)
            CRBlock(hidden_num, hidden_num * 2, (4, 1), (2, 1), (1, 0), bias=False),
            
            # Layer 3: (128, 15, 1) -> (256, 17, 1)
            CRBlock(hidden_num * 2, hidden_num * 4, (8, 1), (1, 1), (1, 0), bias=False),
            
            # Layer 4: (256, 17, 1) -> (512, 9, 1)
            CRBlock(hidden_num * 4, hidden_num * 8, (4, 1), (2, 1), (1, 0), bias=False),
            
            # Layer 5: (512, 9, 1) -> (100, 5, 1)
            nn.Conv2d(hidden_num * 8, 100, (5, 1), (1, 1), (0, 0), bias=False),
        )

    def forward(self, x):
        """
        Encode peptide sequences to latent space.
        
        Args:
            x (torch.Tensor): Input sequences (batch, 1, 30, 6)
        
        Returns:
            torch.Tensor: Latent vectors (batch, 100, 1, 1)
        """
        return self.model(x)


# ==============================================================================
# Generator
# ==============================================================================

class Generator(nn.Module):
    """
    Generator network for creating peptide sequences from latent vectors.
    
    Architecture:
    - Input: (batch, 100, 1, 1) - Latent vectors
    - Output: (batch, 1, 30, 6) - PC6 encoded peptide sequences
    
    The generator uses transposed convolutional layers to progressively
    increase spatial dimensions while decreasing feature depth, generating
    peptide sequences from random noise.
    
    Args:
        in_dim (int): Latent space dimension (default: 100)
        hidden_num (int): Base number of hidden features (default: 64)
        out_dim (int): Output feature dimension (default: 6 for PC6 encoding)
    
    Returns:
        torch.Tensor: Generated sequences of shape (batch, 1, 30, 6)
    """
    def __init__(self, in_dim=100, hidden_num=64, out_dim=6):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1: (100, 1, 1) -> (512, 5, 1)
            TransposedCBR(in_dim, hidden_num * 8, (5, 1), (1, 1), (0, 0)),
            
            # Layer 2: (512, 5, 1) -> (256, 9, 1)
            TransposedCBR(hidden_num * 8, hidden_num * 4, (4, 1), (2, 1), (1, 0)),
            
            # Layer 3: (256, 9, 1) -> (128, 17, 1)
            TransposedCBR(hidden_num * 4, hidden_num * 2, (8, 1), (1, 1), (1, 0)),
            
            # Layer 4: (128, 17, 1) -> (64, 30, 1)
            TransposedCBR(hidden_num * 2, hidden_num, (4, 1), (2, 1), (1, 0)),
            
            # Layer 5: (64, 30, 1) -> (1, 30, 6)
            nn.ConvTranspose2d(hidden_num, 1, (1, out_dim), (1, 1), (0, 0), bias=False),
            
            # Activation: Scale outputs to [-1, 1]
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Generate peptide sequences from latent vectors.
        
        Args:
            x (torch.Tensor): Latent vectors (batch, 100, 1, 1)
        
        Returns:
            torch.Tensor: Generated sequences (batch, 1, 30, 6)
        """
        return self.model(x)


# ==============================================================================
# Discriminator
# ==============================================================================

class Discriminator(nn.Module):
    """
    Discriminator network for BiGAN training.
    
    The discriminator evaluates (sequence, latent) pairs to determine if they
    come from the real distribution E(x) or the generated distribution (G(z), z).
    
    Architecture:
    - Input: Concatenated [sequence_flattened, latent_flattened]
    - Output: Scalar score (higher = more likely real)
    
    Uses fully connected layers with batch normalization and LeakyReLU
    activation for stable training.
    
    Input dimensions:
        - Sequence: 180 features (30 positions × 6 PC6 features)
        - Latent: 100 features
        - Total: 280 features
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            # Input layer: 280 -> 1024
            nn.Linear(180 + 100, 1024),
            nn.LeakyReLU(0.2),
            
            # Hidden layer: 1024 -> 1024
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            
            # Output layer: 1024 -> 1
            nn.Linear(1024, 1)
        )
    
    def forward(self, X, z):
        """
        Evaluate (sequence, latent) pair.
        
        Args:
            X (torch.Tensor): Peptide sequences (batch, 1, 30, 6)
            z (torch.Tensor): Latent vectors (batch, 100, 1, 1)
        
        Returns:
            torch.Tensor: Discriminator scores (batch, 1)
        """
        # Flatten inputs
        X = X.reshape(-1, 30 * 6)  # (batch, 180)
        z = z.reshape(-1, 100)     # (batch, 100)
        
        # Concatenate and discriminate
        Xz = torch.cat([X, z], dim=1)  # (batch, 280)
        return self.layers(Xz)


# ==============================================================================
# Loss Functions
# ==============================================================================

def D_loss(DG, DE, eps=1e-6):
    """
    Standard GAN discriminator loss (not used with WGAN-GP).
    
    Maximizes log(D(E(x))) + log(1 - D(G(z), z))
    
    Args:
        DG (torch.Tensor): Discriminator output for generated pairs
        DE (torch.Tensor): Discriminator output for real pairs
        eps (float): Small constant for numerical stability
    
    Returns:
        torch.Tensor: Discriminator loss
    """
    loss = torch.log(DE + eps) + torch.log(1 - DG + eps)    
    return -torch.mean(loss)


def EG_loss(DG, DE, eps=1e-6):
    """
    Standard GAN encoder-generator loss (not used with WGAN-GP).
    
    Maximizes log(D(G(z), z)) + log(1 - D(E(x)))
    
    Args:
        DG (torch.Tensor): Discriminator output for generated pairs
        DE (torch.Tensor): Discriminator output for real pairs
        eps (float): Small constant for numerical stability
    
    Returns:
        torch.Tensor: Encoder-Generator loss
    """
    loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
    return -torch.mean(loss)


# ==============================================================================
# WGAN-GP Loss
# ==============================================================================

class loss_GP:
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP) loss for BiGAN.
    
    Implements the WGAN-GP objective with gradient penalty for improved
    training stability. The discriminator (critic) is trained to maximize
    the Wasserstein distance between real and fake distributions.
    
    Key features:
    - Wasserstein distance: More meaningful loss metric
    - Gradient penalty: Enforces Lipschitz constraint
    - Stable training: Reduced mode collapse
    
    Args:
        C (Discriminator): Discriminator/Critic network
        x (torch.Tensor): Real sequences (batch, 1, 30, 6)
        z_hat (torch.Tensor): Encoded latent vectors from E(x)
        x_tilde (torch.Tensor): Generated sequences from G(z)
        z (torch.Tensor): Random latent vectors
    """
    def __init__(self, C, x, z_hat, x_tilde, z):
        self.C = C
        self.x = x
        self.z_hat = z_hat
        self.x_tilde = x_tilde
        self.z = z
    
    def criticize(self):
        """
        Evaluate discriminator on real and generated pairs.
        
        Returns:
            tuple: (data_preds, sample_preds)
                - data_preds: Scores for real pairs (x, E(x))
                - sample_preds: Scores for generated pairs (G(z), z)
        """
        x, z_hat, x_tilde, z = self.x, self.z_hat, self.x_tilde, self.z  
        
        # Concatenate real and fake pairs
        input_x = torch.cat((x, x_tilde), dim=0)
        input_z = torch.cat((z_hat, z), dim=0)
        
        # Get discriminator scores
        output = self.C(input_x, input_z)
        
        # Split into real and fake
        data_preds = output[:x.size(0)]
        sample_preds = output[x.size(0):]
        
        return data_preds, sample_preds

    def calculate_grad_penalty(self):
        """
        Calculate gradient penalty for WGAN-GP.
        
        Computes ||∇D(x_interp, z_interp)||₂ and penalizes deviation from 1.
        This enforces the Lipschitz constraint on the discriminator.
        
        Returns:
            torch.Tensor: Gradient penalty term
        """
        x, z_hat, x_tilde, z = self.x, self.z_hat, self.x_tilde, self.z  
        
        # Sample random interpolation coefficient
        bsize = x.size(0)
        eps = torch.rand(bsize, 1, 1, 1).to(x.device)  # eps ~ Uniform[0, 1]
        
        # Create interpolated samples
        intp_x = eps * x + (1 - eps) * x_tilde
        intp_z = eps * z_hat + (1 - eps) * z
        
        # Enable gradient computation for interpolated samples
        intp_x.requires_grad_(True)
        intp_z.requires_grad_(True)
        
        # Compute discriminator output
        C_intp_loss = self.C(intp_x, intp_z).sum()
        
        # Compute gradients w.r.t. interpolated inputs
        grads = autograd.grad(
            outputs=C_intp_loss, 
            inputs=(intp_x, intp_z), 
            retain_graph=True, 
            create_graph=True
        )
        
        # Flatten and concatenate gradients
        grads_x = grads[0].view(bsize, -1)
        grads_z = grads[1].view(bsize, -1)
        grads = torch.cat((grads_x, grads_z), dim=1)
        
        # Compute gradient penalty
        grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        
        return grad_penalty
    
    def loss(self, lamb=10):
        """
        Compute WGAN-GP losses for all networks.
        
        Args:
            lamb (float): Gradient penalty coefficient (default: 10)
        
        Returns:
            tuple: (C_loss, EG_loss, E_loss, G_loss)
                - C_loss: Discriminator loss with gradient penalty
                - EG_loss: Encoder-Generator loss (Wasserstein distance)
                - E_loss: Encoder loss (for monitoring)
                - G_loss: Generator loss (for monitoring)
        """
        # Get discriminator predictions
        data_preds, sample_preds = self.criticize()
        
        # Wasserstein distance (to be maximized)
        EG_loss = torch.mean(data_preds - sample_preds)
        
        # Discriminator loss (minimize negative Wasserstein + penalty)
        C_loss = -EG_loss + lamb * self.calculate_grad_penalty()
        
        # Individual losses for monitoring
        E_loss = torch.mean(data_preds)    # How well encoder fools discriminator
        G_loss = torch.mean(sample_preds)  # How well generator fools discriminator
        
        return C_loss, EG_loss, E_loss, G_loss


# ==============================================================================
# Model Factory
# ==============================================================================

def get_model_and_optimizer(latent_size=100):
    """
    Factory function to create and initialize all models.
    
    Creates the three components of the BiGAN architecture:
    - Encoder: Maps sequences to latent space
    - Generator: Creates sequences from latent vectors
    - Discriminator: Evaluates (sequence, latent) pairs
    
    Args:
        latent_size (int): Dimension of latent space (default: 100)
    
    Returns:
        tuple: (generator, encoder, discriminator)
            - generator (Generator): Initialized generator network
            - encoder (Encoder): Initialized encoder network
            - discriminator (Discriminator): Initialized discriminator network
    
    Example:
        >>> G, E, D = get_model_and_optimizer(latent_size=100)
        >>> G.apply(weights_init)  # Initialize weights
        >>> E.apply(weights_init)
        >>> D.apply(init_weights)
    """
    # Model hyperparameters
    encoded_num = 6      # PC6 encoding dimension
    hidden_size = 64     # Base feature dimension
    
    # Initialize models
    generator = Generator(latent_size, hidden_size, encoded_num)
    encoder = Encoder(encoded_num, hidden_size)
    discriminator = Discriminator()
    
    return generator, encoder, discriminator
