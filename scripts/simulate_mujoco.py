import mujoco
import mujoco.viewer
import torch
import numpy as np
import time
from neuro_int_mamba.model import NeuroINTMamba

def run_simulation(model_path=None):
    # 1. Load MuJoCo model
    m = mujoco.MjModel.from_xml_path('simulation/dex_robot.xml')
    d = mujoco.MjData(m)

    # 2. Initialize Neuro-INT Mamba model
    # Assuming 2-DOF robot (2 motors)
    # Vision: 128x128, Tactile: 16, EMG: 8
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

    # Reset model states
    model.reset_states()

    # 3. Simulation Loop
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- Get Observations from MuJoCo ---
            # 1. Proprioception: [joint_pos, joint_vel]
            # d.qpos and d.qvel are the joint positions and velocities
            proprio_obs = torch.tensor(np.concatenate([d.qpos, d.qvel]), dtype=torch.float32).unsqueeze(0).to(device)
            
            # 2. Vision: Render from camera
            # For simplicity, we'll use a mock image if rendering is not available, 
            # but here is how you'd do it:
            # renderer.update_scene(d)
            # vision_obs = torch.tensor(renderer.render(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            vision_obs = torch.randn(1, 3, 128, 128).to(device) # Mock for now
            
            # 3. Tactile: Mock or from sensors
            # If we had touch sensors in XML, we'd read d.sensordata
            tactile_obs = torch.randn(1, 16).to(device)
            
            # 4. EMG: Human intent (Mock or from dataset)
            emg_obs = torch.randn(1, 8).to(device)

            # --- Model Inference (O(1) Step) ---
            with torch.no_grad():
                # Neuro-INT Mamba step() for real-time control
                action, _ = model.step(
                    visual=vision_obs, 
                    tactile=tactile_obs, 
                    emg=emg_obs, 
                    proprio=proprio_obs
                )
                action = action.cpu().numpy().flatten()

            # --- Apply Actions to MuJoCo ---
            d.ctrl[:2] = action

            # --- Step Simulation ---
            mujoco.mj_step(m, d)

            # --- Sync Viewer ---
            viewer.sync()

            # Maintain real-time frequency
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    run_simulation()
