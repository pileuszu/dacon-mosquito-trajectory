import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MosquitoDatasetStep5(Dataset):
    def __init__(self, data_dir, labels_file=None, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
        if mode == 'train':
            self.labels = pd.read_csv(labels_file)
            self.ids = self.labels['id'].values
        else:
            self.ids = sorted([f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.csv')])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_id = self.ids[idx]
        file_path = os.path.join(self.data_dir, f"{file_id}.csv")
        
        # Load sequence (11 points, -400ms to 0ms)
        df = pd.read_csv(file_path)
        coords = df[['x', 'y', 'z']].values.astype(np.float32) # (11, 3)
        
        # 1. Translation Normalization (Shift origin to 0ms)
        origin = coords[-1].copy()
        coords_norm = coords - origin
        
        # 2. Constant Velocity (CV) Baseline
        # V = P_0 - P_{-40}
        v = coords[-1] - coords[-2]
        cv_pred = coords[-1] + 2 * v # P_80 = P_0 + 2 * (P_0 - P_{-40})
        
        # 3. FFT Features
        # Apply FFT per axis
        fft_features = []
        for i in range(3):
            axis_data = coords_norm[:, i]
            fft_res = np.fft.rfft(axis_data)
            fft_features.extend(np.abs(fft_res))
            fft_features.extend(np.angle(fft_res))
        fft_features = np.array(fft_features, dtype=np.float32)
        
        if self.mode == 'train':
            target_abs = self.labels.iloc[idx][['x', 'y', 'z']].values.astype(np.float32) # (3,)
            # Target for model: Residual from CV prediction
            target_residual = target_abs - cv_pred
            
            return {
                'coords': torch.from_numpy(coords_norm),
                'fft': torch.from_numpy(fft_features),
                'cv_pred': torch.from_numpy(cv_pred),
                'target': torch.from_numpy(target_residual),
                'origin': torch.from_numpy(origin)
            }
        else:
            return {
                'coords': torch.from_numpy(coords_norm),
                'fft': torch.from_numpy(fft_features),
                'cv_pred': torch.from_numpy(cv_pred),
                'origin': torch.from_numpy(origin),
                'id': file_id
            }

if __name__ == "__main__":
    # Quick test
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "../data/open/train")
    labels_path = os.path.join(base_dir, "../data/open/train_labels.csv")
    
    if not os.path.exists(train_dir):
        train_dir = "data/open/train"
        labels_path = "data/open/train_labels.csv"

    dataset = MosquitoDatasetStep5(train_dir, labels_path)
    sample = dataset[0]
    print(f"Coords shape: {sample['coords'].shape}")
    print(f"FFT shape: {sample['fft'].shape}")
    print(f"CV Pred: {sample['cv_pred']}")
    print(f"Target Residual: {sample['target']}")
