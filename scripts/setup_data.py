import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

def setup_emg_dataset():
    data_dir = "data/emg_gestures"
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, "emg_dataset.zip")
    url = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"
    
    if not os.path.exists(zip_path):
        print(f"Downloading EMG dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")

    # Process a sample file to verify
    # The dataset has folders for each subject (1-36)
    sample_dir = os.path.join(data_dir, "EMG_data_for_gestures-master", "01")
    if os.path.exists(sample_dir):
        files = [f for f in os.listdir(sample_dir) if f.endswith('.txt')]
        if files:
            sample_file = os.path.join(sample_dir, files[0])
            # The file has 10 columns: Time, 8 EMG channels, Class
            df = pd.read_csv(sample_file, sep='\t')
            print(f"Sample data shape from {files[0]}: {df.shape}")
            print("EMG Dataset is ready.")
        else:
            print("No .txt files found in subject 01 directory.")
    else:
        # Try to find the master folder if it's named differently
        print(f"Directory {sample_dir} not found. Checking structure...")
        print(os.listdir(data_dir))

def setup_mock_vision_tactile():
    """
    Since real vision/tactile datasets are huge, we create a structured 
    directory to show how they should be configured.
    """
    vision_dir = "data/vision_samples"
    tactile_dir = "data/tactile_samples"
    os.makedirs(vision_dir, exist_ok=True)
    os.makedirs(tactile_dir, exist_ok=True)
    
    print("Creating structured directories for Vision and Tactile data...")
    # Create a dummy .npy file to represent a processed batch
    dummy_vision = np.random.randn(10, 3, 128, 128).astype(np.float32)
    dummy_tactile = np.random.randn(10, 16).astype(np.float32)
    
    np.save(os.path.join(vision_dir, "sample_0.npy"), dummy_vision)
    np.save(os.path.join(tactile_dir, "sample_0.npy"), dummy_tactile)
    print("Mock structured data created in data/vision_samples and data/tactile_samples.")

if __name__ == "__main__":
    setup_emg_dataset()
    setup_mock_vision_tactile()
