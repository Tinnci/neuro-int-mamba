import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NinaProDataset(Dataset):
    """
    Skeleton for NinaPro (DB2/DB5) Dataset loading.
    In a real scenario, this would load .mat files using scipy.io.
    """
    def __init__(self, file_paths, window_size=200, stride=50):
        self.window_size = window_size
        self.stride = stride
        # Placeholder for loaded data
        self.data = [] 
        self.labels = []
        
    def _preprocess(self, raw_emg):
        """
        Apply filtering and rectification.
        """
        # 1. Rectification
        rectified = np.abs(raw_emg)
        # 2. Low-pass filter (Envelope) - simplified as moving average
        return rectified 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emg_window = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(emg_window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class DexRobotAlignmentDataset(Dataset):
    """
    Dataset for aligning Human EMG with Robot Proprioception.
    Pairs human intent (EMG) with retargeted robot states.
    """
    def __init__(self, emg_data, robot_states):
        self.emg_data = emg_data
        self.robot_states = robot_states

    def __len__(self):
        return len(self.emg_data)

    def __getitem__(self, idx):
        return {
            'emg': torch.tensor(self.emg_data[idx], dtype=torch.float32),
            'proprio': torch.tensor(self.robot_states[idx], dtype=torch.float32)
        }

def get_dataloaders(batch_size=32):
    # Mock data for demonstration
    mock_emg = np.random.randn(100, 8)
    mock_robot = np.random.randn(100, 54)
    
    dataset = DexRobotAlignmentDataset(mock_emg, mock_robot)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
