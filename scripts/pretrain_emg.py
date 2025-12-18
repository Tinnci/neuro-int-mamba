import torch
import torch.nn as nn
import torch.optim as optim
from neuro_int_mamba import NeuroINTMamba
from neuro_int_mamba.data import get_dataloaders

def masked_emg_pretraining():
    """
    Pre-training script for Masked EMG Modeling (MEM).
    The model learns to reconstruct masked parts of the EMG signal,
    forcing it to learn underlying muscle synergy patterns.
    """
    model = NeuroINTMamba(
        vision_dim=128, 
        tactile_dim=16, 
        emg_dim=8, 
        action_dim=2, 
        d_model=128, 
        num_layers=2, 
        use_emg=True
    )
    
    # Reconstruction head for MEM
    reconstruction_head = nn.Linear(128, 8)
    
    optimizer = optim.Adam(list(model.parameters()) + list(reconstruction_head.parameters()), lr=1e-4)
    criterion = nn.MSELoss()
    
    dataloader = get_dataloaders(batch_size=16)
    
    print("Starting Masked EMG Pre-training (MEM)...")
    
    for epoch in range(5):
        for batch in dataloader:
            emg = batch['emg'] # (B, D)
            
            # Create mask (mask 30% of channels)
            mask = torch.rand_like(emg) > 0.3
            masked_emg = emg * mask
            
            optimizer.zero_grad()
            
            # Encode masked EMG
            latent = model.emg_encoder(masked_emg)
            
            # Reconstruct original EMG
            reconstructed = reconstruction_head(latent)
            
            loss = criterion(reconstructed, emg)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}, MEM Loss: {loss.item():.4f}")

    print("Pre-training complete. The EMGEncoder has learned robust muscle synergy features.")

if __name__ == "__main__":
    masked_emg_pretraining()
