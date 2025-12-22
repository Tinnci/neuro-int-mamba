import sapien
import sapien.render as render
import torch
import numpy as np
from neuro_int_mamba.model import NeuroINTMamba
from neuro_int_mamba.data import get_dataloaders

def create_robot(scene: sapien.Scene):
    builder = scene.create_articulation_builder()
    
    # Base
    base = builder.create_link_builder()
    base.set_name("base")
    base.add_cylinder_collision(radius=0.05, half_length=0.1)
    base.add_cylinder_visual(radius=0.05, half_length=0.1, material=render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 1]))
    
    # Link 1
    link1 = builder.create_link_builder(base)
    link1.set_name("arm_link1")
    link1.set_joint_name("joint1")
    link1.set_joint_properties(
        "revolute",
        limits=[[-np.pi/2, np.pi/2]],
        pose_in_parent=sapien.Pose([0, 0, 0.1], [0.707, 0, 0.707, 0]), # axis 0 1 0
        pose_in_child=sapien.Pose(),
    )
    link1.add_capsule_collision(radius=0.03, half_length=0.15)
    link1.add_capsule_visual(radius=0.03, half_length=0.15, material=render.RenderMaterial(base_color=[0.7, 0.7, 0.7, 1]))
    
    # Link 2
    link2 = builder.create_link_builder(link1)
    link2.set_name("arm_link2")
    link2.set_joint_name("joint2")
    link2.set_joint_properties(
        "revolute",
        limits=[[-np.pi/2, np.pi/2]],
        pose_in_parent=sapien.Pose([0, 0, 0.3], [0.707, 0, 0.707, 0]), # axis 0 1 0
        pose_in_child=sapien.Pose(),
    )
    link2.add_capsule_collision(radius=0.03, half_length=0.15)
    link2.add_capsule_visual(radius=0.03, half_length=0.15, material=render.RenderMaterial(base_color=[0.7, 0.7, 0.7, 1]))
    
    # Palm
    palm = builder.create_link_builder(link2)
    palm.set_name("palm")
    palm.set_joint_name("palm_joint")
    palm.set_joint_properties(
        "fixed", 
        limits=[], 
        pose_in_parent=sapien.Pose([0, 0, 0.3]),
        pose_in_child=sapien.Pose(),
    )
    palm.add_box_collision(half_size=(0.05, 0.02, 0.05))
    palm.add_box_visual(half_size=(0.05, 0.02, 0.05), material=render.RenderMaterial(base_color=[0.1, 0.1, 0.8, 1]))
    
    # Fingers (Simplified for now, can add more detail later)
    for i, pos in enumerate([[0.04, 0, 0.05], [-0.02, 0, 0.05], [-0.06, 0, 0.05]]):
        f_base = builder.create_link_builder(palm)
        f_base.set_name(f"finger{i+1}")
        f_base.set_joint_name(f"f{i+1}_j1")
        f_base.set_joint_properties(
            "revolute", 
            limits=[[0, np.pi/2]], 
            pose_in_parent=sapien.Pose(pos),
            pose_in_child=sapien.Pose(),
        )
        f_base.add_capsule_collision(radius=0.01, half_length=0.02)
        f_base.add_capsule_visual(radius=0.01, half_length=0.02, material=render.RenderMaterial(base_color=[0.2, 0.8, 0.2, 1]))
        
        f_tip = builder.create_link_builder(f_base)
        f_tip.set_name(f"finger{i+1}_tip")
        f_tip.set_joint_name(f"f{i+1}_j2")
        f_tip.set_joint_properties(
            "revolute", 
            limits=[[0, np.pi/2]], 
            pose_in_parent=sapien.Pose([0, 0, 0.04]),
            pose_in_child=sapien.Pose(),
        )
        f_tip.add_capsule_collision(radius=0.01, half_length=0.015)
        f_tip.add_capsule_visual(radius=0.01, half_length=0.015, material=render.RenderMaterial(base_color=[0.2, 0.8, 0.2, 1]))

    robot = builder.build()
    
    # Configure drives
    for joint in robot.get_active_joints():
        joint.set_drive_property(stiffness=1000, damping=100)
        
    return robot

def run_simulation(use_real_data=True):
    # 1. Setup SAPIEN
    scene = sapien.Scene()
    scene.set_timestep(1/100.0)
    
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    
    # Add a small cube to interact with
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=(0.02, 0.02, 0.02))
    builder.add_box_visual(half_size=(0.02, 0.02, 0.02), material=render.RenderMaterial(base_color=[1, 0, 0, 1]))
    cube = builder.build()
    cube.set_pose(sapien.Pose([0.1, 0, 0.2]))
    
    robot = create_robot(scene)
    robot.set_pose(sapien.Pose([0, 0, 0.1]))
    
    # 2. Initialize Neuro-INT Mamba model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroINTMamba(
        vision_dim=128,
        tactile_dim=3,
        emg_dim=8,
        action_dim=8, # 2 arm + 6 finger joints
        d_model=128,
        use_emg=True
    ).to(device)
    model.eval()
    model.reset_states()

    # 3. Data Source (EMG)
    data_iter = None
    if use_real_data:
        try:
            loader = get_dataloaders(batch_size=1)
            data_iter = iter(loader)
            print("Real EMG data loader initialized.")
        except Exception as e:
            print(f"Could not initialize real EMG data: {e}")

    # 4. Camera Setup
    camera = scene.add_camera(
        name="hand_cam",
        width=128,
        height=128,
        fovy=np.deg2rad(35),
        near=0.1,
        far=10.0,
    )
    camera.set_local_pose(sapien.Pose([0, -0.6, 0.4], [0.9238795, 0.3826834, 0, 0])) # Looking at palm

    # 5. Viewer
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # 6. Simulation Loop
    print("Starting simulation loop...")
    for i in range(2000):
        if viewer.closed:
            break
            
        # --- Get Observations ---
        # 1. Proprioception
        qpos = robot.get_qpos()
        qvel = robot.get_qvel()
        proprio_obs = torch.tensor(np.concatenate([qpos, qvel]), dtype=torch.float32).unsqueeze(0).to(device)
        
        # 2. Vision
        scene.update_render()
        camera.take_picture()
        rgba = camera.get_picture('Color') # [H, W, 4]
        vision_obs = torch.tensor(rgba[:, :, :3], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # 3. Tactile (Sum impulses on finger tips)
        tactile_data = np.zeros(3)
        for contact in scene.get_contacts():
            for point in contact.points:
                # Check if one of the bodies is a finger tip
                for body in contact.bodies:
                    if "finger" in body.entity.name and "tip" in body.entity.name:
                        # Simple heuristic: sum up impulse magnitudes
                        tactile_data[0] += np.linalg.norm(point.impulse)
        tactile_obs = torch.tensor(tactile_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 4. EMG
        if data_iter is not None:
            try:
                batch = next(data_iter)
                # Assuming batch['emg'] is (B, L, D)
                emg_obs = batch['emg'][:, -1, :].to(device)
            except StopIteration:
                data_iter = iter(loader)
                emg_obs = torch.zeros((1, 8)).to(device)
        else:
            emg_obs = torch.zeros((1, 8)).to(device)
        
        # --- Inference ---
        with torch.no_grad():
            action, _ = model.step(
                visual=vision_obs,
                tactile=tactile_obs,
                emg=emg_obs,
                proprio=proprio_obs
            )
            action = action.cpu().numpy().flatten()
        
        # --- Control ---
        # Use drive targets for PD control
        for j, target in zip(robot.get_active_joints(), action):
            j.set_drive_target(target)
        
        scene.step()
        scene.update_render()
        viewer.render()
        
        if i % 100 == 0:
            print(f"Step {i}, Action: {action[:3]}..., Tactile: {tactile_data}")

    print("Simulation finished.")

if __name__ == "__main__":
    run_simulation()
