import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from glob import glob

class AdvancedMosquitoDataset(Dataset):
    def __init__(self, data_dir, label_path, file_list=None, mode='train', augment=False):
        self.data_dir = data_dir
        self.mode = mode
        self.augment = augment
        
        # Arena Boundaries (from EDA)
        self.x_min, self.x_max = 0.54, 6.79
        self.y_min, self.y_max = -2.43, 2.16
        self.z_min, self.z_max = -1.60, 2.53
        
        if file_list is None:
            self.file_list = sorted(glob(os.path.join(data_dir, '*.csv')))
        else:
            self.file_list = file_list
            
        if mode == 'train':
            self.labels = pd.read_csv(label_path).set_index('id')
        
    def __len__(self):
        return len(self.file_list)
    
    def rotate_z(self, coords, angle):
        """Rotate 3D coordinates around Z-axis"""
        rad = np.radians(angle)
        c, s = np.cos(rad), np.sin(rad)
        rot_matrix = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ], dtype=np.float32)
        return coords @ rot_matrix.T

    def get_arena_features(self, p):
        """Calculate distance to 6 walls and absolute height"""
        dist_x_min = p[0] - self.x_min
        dist_x_max = self.x_max - p[0]
        dist_y_min = p[1] - self.y_min
        dist_y_max = self.y_max - p[1]
        dist_z_min = p[2] - self.z_min
        dist_z_max = self.z_max - p[2]
        # Return as normalized features (approximate scaling)
        return np.array([dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max, p[2]], dtype=np.float32)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        file_id = os.path.basename(file_path).split('.')[0]
        
        df = pd.read_csv(file_path)
        coords_abs = df[['x', 'y', 'z']].values.astype(np.float32)
        origin_abs = coords_abs[-1].copy()
        
        # 1. Arena Awareness: Get features for the current point (t=0)
        arena_feat = self.get_arena_features(origin_abs)
        
        # 2. Coordinate Normalization
        rel_coords = coords_abs - origin_abs
        
        # 3. Data Augmentation (Rotation)
        if self.augment and self.mode == 'train':
            angle = np.random.uniform(0, 360)
            rel_coords = self.rotate_z(rel_coords, angle)
            # Note: Arena features should NOT be rotated as walls are fixed in global frame
        
        if self.mode == 'train':
            target_abs = self.labels.loc[file_id].values.astype(np.float32)
            target_rel = target_abs - origin_abs
            if self.augment:
                target_rel = self.rotate_z(target_rel.reshape(1, 3), angle).flatten()
            
            return torch.tensor(rel_coords), torch.tensor(arena_feat), torch.tensor(target_rel)
        else:
            return torch.tensor(rel_coords), torch.tensor(arena_feat), torch.tensor(origin_abs), file_id

def get_dataloaders(data_dir, label_path, batch_size=64, split_ratio=0.8):
    all_files = sorted(glob(os.path.join(data_dir, '*.csv')))
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    train_ds = AdvancedMosquitoDataset(data_dir, label_path, train_files, mode='train', augment=True)
    val_ds = AdvancedMosquitoDataset(data_dir, label_path, val_files, mode='train', augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
