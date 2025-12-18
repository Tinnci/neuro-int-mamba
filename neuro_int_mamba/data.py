import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.signal import butter, sosfilt

class GestureEMGDataset(Dataset):
    """
    UCI EMG Data for Gestures.
    Each file has 10 columns: Time, 8 EMG channels, Class.
    """
    def __init__(self, data_dir, window_size=200, stride=50):
        self.window_size = window_size
        self.stride = stride
        self.data = []
        self.labels = []
        
        # Find all subject folders
        subject_folders = glob.glob(os.path.join(data_dir, "EMG_data_for_gestures-master", "*"))
        for folder in subject_folders:
            if not os.path.isdir(folder):
                continue
            txt_files = glob.glob(os.path.join(folder, "*.txt"))
            for f in txt_files:
                df = pd.read_csv(f, sep='\t')
                # Columns 1-8 are EMG
                emg_values = df.iloc[:, 1:9].values
                # Column 9 is Class
                label_values = df.iloc[:, 9].values
                
                # Windowing
                for i in range(0, len(emg_values) - window_size, stride):
                    self.data.append(emg_values[i:i+window_size])
                    # Use the most frequent label in the window
                    counts = np.bincount(label_values[i:i+window_size].astype(int))
                    self.labels.append(np.argmax(counts))
                    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'emg': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class NinaProDataset(Dataset):
    """
    NinaPro (DB2/DB5) Dataset loading with preprocessing.
    """
    def __init__(self, file_paths, window_size=200, stride=50, fs=2000):
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        # Placeholder for loaded data
        self.data = [] 
        self.labels = []
        # Pre-calculate filter coefficients for optimization
        self.sos = self._create_filter(20, 500)
        
    def _create_filter(self, lowcut, highcut, order=4):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        # Use SOS (Second-Order Sections) for better numerical stability
        return butter(order, [low, high], btype='bandpass', output='sos')

    def _preprocess(self, raw_emg):
        """
        Apply Butterworth band-pass filtering and rectification.
        """
        # 1. Band-pass filter (20-500Hz) using pre-calculated SOS
        filtered = sosfilt(self.sos, raw_emg, axis=0)
        
        # 2. Rectification
        rectified = np.abs(filtered)
        
        # 3. Feature Extraction (RMS/MAV)
        # In a real scenario, this would be done per window
        return rectified 

    def extract_features(self, window):
        """
        Extract RMS and MAV features from a window.
        """
        rms = np.sqrt(np.mean(window**2, axis=0))
        mav = np.mean(np.abs(window), axis=0)
        return np.concatenate([rms, mav])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        emg_window = self.data[index]
        label = self.labels[index]
        return torch.tensor(emg_window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class DexRobotAlignmentDataset(Dataset):
    """
    Dataset for aligning Human EMG with Robot Proprioception.
    Pairs human intent (EMG) with retargeted robot states.
    """
    def __init__(self, emg_data, robot_states):
        self.emg_data = emg_data
        self.robot_states = robot_states

    def __len__(self) -> int:
        return len(self.emg_data)

    def __getitem__(self, index: int):
        return {
            'emg': torch.tensor(self.emg_data[index], dtype=torch.float32),
            'proprio': torch.tensor(self.robot_states[index], dtype=torch.float32)
        }

def get_dataloaders(batch_size=32, data_dir="data/emg_gestures"):
    if os.path.exists(data_dir):
        print(f"Loading real EMG data from {data_dir}...")
        dataset = GestureEMGDataset(data_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        print(f"Warning: {data_dir} not found. Using mock data.")
        # Mock data for demonstration
        mock_emg = np.random.randn(100, 8)
        mock_robot = np.random.randn(100, 4)
        
        dataset = DexRobotAlignmentDataset(mock_emg, mock_robot)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
