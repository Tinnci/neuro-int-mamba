import torch
from neuro_int_mamba import NeuroINTMamba

def main():
    print("--- Neuro-INT Mamba Demo ---")
    
    # Vision: 128x128 (flattened or raw), Tactile: 16, EMG: 8, Action: 2
    model = NeuroINTMamba(
        vision_dim=128, 
        tactile_dim=16, 
        emg_dim=8, 
        action_dim=2, 
        d_model=128, 
        num_layers=4
    )
    
    # Batch processing example
    # Proprio: (B, L, action_dim * 2)
    p = torch.randn(1, 10, 4)
    # Tactile: (B, L, tactile_dim)
    t = torch.randn(1, 10, 16)
    # Visual: (B, L, 3, 128, 128)
    v = torch.randn(1, 10, 3, 128, 128)
    
    motor_cmd, next_pred = model(p, t, v)
    print(f"Batch Motor Command Shape: {motor_cmd.shape}") # Should be (1, 10, 2)
    
    # Real-time step example
    p_t = torch.randn(1, 4)
    t_t = torch.randn(1, 16)
    v_t = torch.randn(1, 3, 128, 128)
    
    model.reset_states()
    motor_out, _ = model.step(visual=v_t, tactile=t_t, proprio=p_t)
    print(f"Step Motor Command Shape: {motor_out.shape}")
    print(f"Single Step Motor Output Shape: {motor_out.shape}") # Should be (1, 27)
    print(f"Single Step Motor Output Norm: {motor_out.norm().item():.4f}")

if __name__ == "__main__":
    main()
