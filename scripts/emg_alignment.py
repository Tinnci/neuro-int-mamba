import torch
import torch.nn as nn
import torch.optim as optim
from neuro_int_mamba import NeuroINTMamba

def train_emg_alignment():
    """
    Demonstration script for aligning human EMG features with robot latent space
    using a contrastive loss (InfoNCE-like) or MSE alignment.
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
    # In a real scenario, 'human_emg' comes from NinaPro/MyoDataset
    # 'robot_proprio' comes from the robot's sensors during the same task
    batch_size = 32
    human_emg = torch.randn(batch_size, 10, 8)
    robot_proprio = torch.randn(batch_size, 10, 54)
    
    # 3. Alignment Objective
    # We want the EMG latent representation to be close to the Proprioception representation
    # in the Thalamic latent space.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print("Starting EMG-Robot Latent Alignment...")
    
    for epoch in range(5):
        optimizer.zero_grad()
        
        # Get EMG latent features
        emg_latent = model.emg_encoder(human_emg)
        
        # Get Proprioception latent features
        proprio_latent = model.proprio_proj(robot_proprio)
        
        # Alignment Loss: Minimize distance between human intent and robot state
        loss = criterion(emg_latent, proprio_latent)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Alignment Loss: {loss.item():.4f}")

    print("Alignment complete. The model now maps human EMG intent to robot-compatible latent space.")

if __name__ == "__main__":
    train_emg_alignment()
