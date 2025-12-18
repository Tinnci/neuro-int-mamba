import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neuro_int_mamba import NeuroINTMamba

def info_nce_loss(z_emg, z_robot, temperature=0.07):
    """
    InfoNCE Loss for cross-modal alignment.
    z_emg, z_robot: [batch_size, embedding_dim]
    """
    z_emg = F.normalize(z_emg, dim=-1)
    z_robot = F.normalize(z_robot, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(z_emg, z_robot.T) / temperature
    
    # Labels are the diagonal (paired samples)
    labels = torch.arange(z_emg.size(0), device=z_emg.device)
    
    return F.cross_entropy(logits, labels)

def train_emg_alignment():
    """
    Demonstration script for aligning human EMG features with robot latent space
    using InfoNCE (Contrastive Learning).
    """
    # 1. Setup Model
    input_dims = {
        'proprio': 54,
        'tactile': 100,
        'visual': 256,
        'goal': 32,
        'emg': 8
    }
    model = NeuroINTMamba(input_dims, model_dim=128, num_layers=2, use_emg=True)
    
    # 2. Simulated Data
    batch_size = 32
    human_emg = torch.randn(batch_size, 8) # Single time step for simplicity
    robot_proprio = torch.randn(batch_size, 54)
    
    # 3. Alignment Objective
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting EMG-Robot Latent Alignment with InfoNCE...")
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Get EMG latent features
        emg_latent = model.emg_encoder(human_emg)
        
        # Get Proprioception latent features
        proprio_latent = model.proprio_proj(robot_proprio)
        
        # Alignment Loss: InfoNCE
        loss = info_nce_loss(emg_latent, proprio_latent)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, InfoNCE Loss: {loss.item():.4f}")

    print("Alignment complete. The model now maps human EMG intent to robot-compatible latent space using contrastive learning.")

if __name__ == "__main__":
    train_emg_alignment()
