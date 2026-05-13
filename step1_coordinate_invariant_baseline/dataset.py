import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm.auto import tqdm

class MosquitoDataset(Dataset):
    def __init__(self, data_dir, label_path, file_list=None, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        
        if file_list is None:
            self.file_list = sorted(glob(os.path.join(data_dir, '*.csv')))
        else:
            self.file_list = file_list
            
        if mode == 'train':
            self.labels = pd.read_csv(label_path).set_index('id')
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        file_id = os.path.basename(file_path).split('.')[0]
        
        df = pd.read_csv(file_path)
        # Assuming 11 points (-400ms to 0ms)
        coords = df[['x', 'y', 'z']].values.astype(np.float32)
        
        # 1. Coordinate Normalization: Subtract last point (t=0)
        origin = coords[-1].copy()
        rel_coords = coords - origin
        
        if self.mode == 'train':
            target_abs = self.labels.loc[file_id].values.astype(np.float32)
            target_rel = target_abs - origin
            return torch.tensor(rel_coords), torch.tensor(target_rel), torch.tensor(origin)
        else:
            return torch.tensor(rel_coords), torch.tensor(origin), file_id

def get_dataloaders(data_dir, label_path, batch_size=64, split_ratio=0.8):
    all_files = sorted(glob(os.path.join(data_dir, '*.csv')))
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    train_ds = MosquitoDataset(data_dir, label_path, train_files, mode='train')
    val_ds = MosquitoDataset(data_dir, label_path, val_files, mode='train')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test loading
    DATA_DIR = 'data/open/train/'
    LABEL_PATH = 'data/open/train_labels.csv'
    
    if os.path.exists(DATA_DIR):
        train_loader, val_loader = get_dataloaders(DATA_DIR, LABEL_PATH, batch_size=4)
        hist, target, origin = next(iter(train_loader))
        print(f"History Shape: {hist.shape}") # [Batch, 11, 3]
        print(f"Target Shape: {target.shape}")   # [Batch, 3]
        print(f"Origin Sample: {origin[0]}")
