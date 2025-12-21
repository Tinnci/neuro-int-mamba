import os
import platform

# --- Windows OpenGL Fix ---
# On Windows, MuJoCo's Renderer and Viewer can conflict when using the default 'glfw' backend.
# If 'osmesa' is not supported by your MuJoCo installation, we will fallback to 'glfw'.
if platform.system() == "Windows":
    # We'll try to use the default first, and only set it if we know what we're doing.
    # Setting it to 'osmesa' caused an "invalid value" error on this system.
    pass

import mujoco
import mujoco.viewer
import torch
import numpy as np
import time
from neuro_int_mamba.model import NeuroINTMamba
from neuro_int_mamba.data import get_dataloaders

def run_simulation(use_real_data=True):
    # 1. Load MuJoCo model
    m = mujoco.MjModel.from_xml_path('simulation/dex_robot.xml')
    d = mujoco.MjData(m)
    
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
        # Initialize Renderer inside the viewer context to share the GL context if possible
        renderer = None
        try:
            renderer = mujoco.Renderer(m, height=128, width=128)
            print("Renderer initialized successfully.")
        except Exception as e:
            print(f"Warning: Renderer initialization failed ({e}). Vision input will be zeros.")

        while viewer.is_running():
            step_start = time.time()

            # --- Get Observations ---
            # 1. Proprioception: [qpos, qvel]
            proprio_obs = torch.tensor(np.concatenate([d.qpos, d.qvel]), dtype=torch.float32).unsqueeze(0).to(device)
            
            # 2. Vision: Real rendering from hand_cam
            if renderer is not None:
                renderer.update_scene(d, camera="hand_cam")
                pixels = renderer.render() # (128, 128, 3)
                vision_obs = torch.tensor(pixels, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            else:
                vision_obs = torch.zeros((1, 3, 128, 128)).to(device)
            
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
                
                # Clip actions to prevent instability
                action = np.clip(action, -1.0, 1.0)

            # --- Apply Actions ---
            d.ctrl[:] = action # Apply to all 8 actuators
            try:
                mujoco.mj_step(m, d)
            except Exception as e:
                print(f"Physics step failed: {e}. Resetting simulation...")
                mujoco.mj_resetData(m, d)
                model.reset_states()
                
            viewer.sync()

            # Maintain real-time frequency
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    # Check if data exists, otherwise fallback to mock
    use_real = os.path.exists("data/emg_gestures")
    try:
        run_simulation(use_real_data=use_real)
    except Exception as e:
        print(f"Simulation failed: {e}")
        if platform.system() == "Windows":
            print("\n[TIP] On Windows, if you see rendering artifacts or crashes:")
            print("1. Try updating your GPU drivers.")
            print("2. If vision is not required, the simulation will fallback to zeros.")
            print("3. For stable offscreen rendering, OSMesa is recommended but requires manual setup.")
