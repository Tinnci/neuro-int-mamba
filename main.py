import torch
from neuro_int_mamba import NeuroINTMamba

def main():
    print("--- Neuro-INT Mamba Demo ---")
    
    input_dims = {
        'proprio': 54,
        'tactile': 100,
        'visual': 256,
        'goal': 32
    }
    model = NeuroINTMamba(input_dims, model_dim=512, num_layers=6)
    
    # Batch processing example
    p = torch.randn(1, 10, 54)
    t = torch.randn(1, 10, 100)
    v = torch.randn(1, 10, 256)
    g = torch.randn(1, 10, 32)
    
    motor_cmd, next_pred = model(p, t, v, g)
    print(f"Batch Motor Command Shape: {motor_cmd.shape}")
    
    # Real-time step example
    p_t = torch.randn(1, 54)
    t_t = torch.randn(1, 100)
    v_t = torch.randn(1, 256)
    g_t = torch.randn(1, 32)
    
    motor_out, states, predictions = model.step(p_t, t_t, v_t, g_t)
    print(f"Single Step Motor Output Norm: {motor_out.norm().item():.4f}")

if __name__ == "__main__":
    main()
