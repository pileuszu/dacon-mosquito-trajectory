import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from .config import TRAIN_DIR, METADATA_PATH, TEST_SIZE, RANDOM_STATE, TRAIN_LABELS_PATH

class MosquitoPhysicsDataset(Dataset):
    def __init__(self, file_list, labels_df=None, is_test=False):
        self.file_list = file_list
        self.labels_df = labels_df
        self.is_test = is_test

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        df = pd.read_csv(file_path)
        
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        vel = np.diff(xyz, axis=0, prepend=xyz[0:1])
        accel = np.diff(vel, axis=0, prepend=vel[0:1])
        features = np.concatenate([xyz, vel, accel], axis=1)
        
        last_xyz = xyz[-1]
        features[:, 0:3] = features[:, 0:3] - last_xyz
        
        prev_xyz = xyz[-2]
        cv_prior_rel = 2.0 * (last_xyz - prev_xyz)
        
        item = {
            "id": file_path.stem,
            "seq": torch.tensor(features),
            "cv_prior": torch.tensor(cv_prior_rel),
            "last_pos": torch.tensor(last_xyz)
        }
        
        if not self.is_test and self.labels_df is not None:
            label = self.labels_df.loc[self.labels_df['id'] == file_path.stem, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
            target_rel = label - last_xyz
            residual = target_rel - cv_prior_rel
            
            item["target"] = torch.tensor(label)
            item["residual"] = torch.tensor(residual)
            
        return item

def get_dataloaders(batch_size=128):
    if METADATA_PATH.exists():
        df = pd.read_csv(METADATA_PATH)
    else:
        # Check if we can reuse split from step2
        step2_metadata = Path(__file__).parent.parent / 'step2' / 'metadata.csv'
        if step2_metadata.exists():
            print("Reusing split from step2...")
            df = pd.read_csv(step2_metadata)
            df.to_csv(METADATA_PATH, index=False)
        else:
            train_files = sorted(TRAIN_DIR.glob('*.csv'))
            file_ids = [f.stem for f in train_files]
            train_ids, test_ids = train_test_split(file_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            metadata = []
            for fid in train_ids: metadata.append({"id": fid, "split": "train"})
            for fid in test_ids: metadata.append({"id": fid, "split": "test"})
            df = pd.DataFrame(metadata)
            df.to_csv(METADATA_PATH, index=False)

    train_ids = df[df['split'] == 'train']['id'].tolist()
    test_ids = df[df['split'] == 'test']['id'].tolist()
    
    train_files = [TRAIN_DIR / f"{fid}.csv" for fid in train_ids]
    test_files = [TRAIN_DIR / f"{fid}.csv" for fid in test_ids]
    
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    
    train_ds = MosquitoPhysicsDataset(train_files, labels_df)
    test_ds = MosquitoPhysicsDataset(test_files, labels_df)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
