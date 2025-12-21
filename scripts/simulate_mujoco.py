import mujoco
import mujoco.viewer
import torch
import numpy as np
import time
import os
from neuro_int_mamba.model import NeuroINTMamba
from neuro_int_mamba.data import get_dataloaders

def run_simulation(use_real_data=True):
    # 1. Load MuJoCo model
    m = mujoco.MjModel.from_xml_path('simulation/dex_robot.xml')
    d = mujoco.MjData(m)
    
    # Initialize Renderer for Vision
    renderer = mujoco.Renderer(m, height=128, width=128)

    # 2. Initialize Neuro-INT Mamba model
    # Updated for 8-DOF (2 arm + 6 finger joints)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroINTMamba(
        vision_dim=128,
        tactile_dim=3, # 3 touch sensors
        emg_dim=8,
        action_dim=m.nu, # Automatically match actuator count (8)
        d_model=128,
        use_emg=True
    ).to(device)
    model.eval()
    model.reset_states()

    # 3. Data Source
    data_iter = None
    if use_real_data and os.path.exists("data/emg_gestures"):
        print("Using real EMG data for simulation...")
        loader = get_dataloaders(batch_size=1)
        data_iter = iter(loader)

    # 4. Simulation Loop
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- Get Observations ---
            # 1. Proprioception: [qpos, qvel]
            proprio_obs = torch.tensor(np.concatenate([d.qpos, d.qvel]), dtype=torch.float32).unsqueeze(0).to(device)
            
            # 2. Vision: Real rendering from hand_cam
            renderer.update_scene(d, camera="hand_cam")
            pixels = renderer.render() # (128, 128, 3)
            vision_obs = torch.tensor(pixels, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            
            # 3. Tactile: Real touch sensor data
            tactile_obs = torch.tensor(d.sensordata[:3], dtype=torch.float32).unsqueeze(0).to(device)
            
            # 4. EMG: Real or Mock
            if data_iter is not None:
                try:
                    batch = next(data_iter)
                    emg_obs = batch['emg'][:, -1, :].to(device)
                except StopIteration:
                    data_iter = iter(loader)
                    emg_obs = torch.randn(1, 8).to(device)
            else:
                emg_obs = torch.randn(1, 8).to(device)

            # --- Model Inference ---
            with torch.no_grad():
                action, _ = model.step(
                    visual=vision_obs, 
                    tactile=tactile_obs, 
                    emg=emg_obs, 
                    proprio=proprio_obs
                )
                action = action.cpu().numpy().flatten()

            # --- Apply Actions ---
            d.ctrl[:] = action # Apply to all 8 actuators
            mujoco.mj_step(m, d)
            viewer.sync()

            # Maintain real-time frequency
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    # Check if data exists, otherwise fallback to mock
    use_real = os.path.exists("data/emg_gestures")
    run_simulation(use_real_data=use_real)

if __name__ == "__main__":
    run_simulation()
