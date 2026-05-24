import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from .config import TRAIN_DIR, METADATA_PATH, TEST_SIZE, RANDOM_STATE

class MosquitoDataset(Dataset):
    def __init__(self, file_list, labels_df=None, is_test=False):
        self.file_list = file_list
        self.labels_df = labels_df
        self.is_test = is_test

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        df = pd.read_csv(file_path)
        
        # Historical coordinates (10 points, 400ms)
        # Shape: (10, 3)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        
        # Last known point (P0)
        last_xyz = xyz[-1]
        prev_xyz = xyz[-2]
        
        # Translation Normalization: Set last point as origin
        normalized_xyz = xyz - last_xyz
        
        # Constant Velocity Prior
        # P_pred = P_0 + 2 * (P_0 - P_{-40})
        # Relative to P0, CV_pred is: 2 * (P_0 - P_{-40})
        cv_prior_rel = 2.0 * (last_xyz - prev_xyz)
        
        item = {
            "id": file_path.stem,
            "seq": torch.tensor(normalized_xyz),
            "cv_prior": torch.tensor(cv_prior_rel),
            "last_pos": torch.tensor(last_xyz)
        }
        
        if not self.is_test and self.labels_df is not None:
            label = self.labels_df.loc[self.labels_df['id'] == file_path.stem, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
            # The target for the model is the residual: Label - (P0 + CV_Prior)
            # which is: (Label - P0) - CV_Prior
            target_rel = label - last_xyz
            residual = target_rel - cv_prior_rel
            
            item["target"] = torch.tensor(label)
            item["residual"] = torch.tensor(residual)
            
        return item

def get_dataloaders(batch_size=128):
    if METADATA_PATH.exists():
        df = pd.read_csv(METADATA_PATH)
    else:
        # For Step 1, we can reuse Step 0 split if available, 
        # but let's assume we want a fresh one or just check if it exists in parent
        parent_metadata = Path(__file__).parent.parent / 'step0' / 'metadata.csv'
        if parent_metadata.exists():
            print(f"Copying metadata from step0...")
            df = pd.read_csv(parent_metadata)
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
    
    from .config import TRAIN_LABELS_PATH
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    
    train_ds = MosquitoDataset(train_files, labels_df)
    test_ds = MosquitoDataset(test_files, labels_df)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
