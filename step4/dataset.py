import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MosquitoDataset(Dataset):
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
        
        if self.mode == 'train':
            target = self.labels.iloc[idx][['x', 'y', 'z']].values.astype(np.float32) # (3,)
            
            if self.transform:
                coords, target = self.transform(coords, target)
                
            return torch.from_numpy(coords), torch.from_numpy(target)
        else:
            return torch.from_numpy(coords), file_id

def get_dataloader(data_dir, labels_file=None, mode='train', batch_size=32, shuffle=True):
    dataset = MosquitoDataset(data_dir, labels_file, mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Test dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "../data/open/train")
    labels_path = os.path.join(base_dir, "../data/open/train_labels.csv")
    
    if not os.path.exists(train_dir):
        # Fallback for running from project root
        train_dir = "data/open/train"
        labels_path = "data/open/train_labels.csv"

    dataset = MosquitoDataset(train_dir, labels_path)
    print(f"Dataset size: {len(dataset)}")
    coords, target = dataset[0]
    print(f"Coords shape: {coords.shape}")
    print(f"Target shape: {target.shape}")
