import torch
from neuro_int_mamba.model import NeuroINTMamba

def test_model_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroINTMamba(
        vision_dim=128,
        tactile_dim=16,
        emg_dim=8,
        action_dim=2,
        d_model=128,
        use_emg=True
    ).to(device)
    model.eval()
    model.reset_states()

    # Mock inputs
    vision_obs = torch.randn(1, 3, 128, 128).to(device)
    tactile_obs = torch.randn(1, 16).to(device)
    emg_obs = torch.randn(1, 8).to(device)
    proprio_obs = torch.randn(1, 4).to(device) # 2 DOF * 2 (pos, vel)

    print("Running model.step()...")
    with torch.no_grad():
        action, prediction = model.step(
            visual=vision_obs, 
            tactile=tactile_obs, 
            emg=emg_obs, 
            proprio=proprio_obs
        )
    
    print(f"Action shape: {action.shape}")
    assert action.shape == (1, 2)
    print("Step successful!")

if __name__ == "__main__":
    test_model_step()
