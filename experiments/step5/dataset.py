import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .config import *

class GMMDataset(Dataset):
    def __init__(self, file_list, labels_df=None, is_test=False, augment=False):
        self.file_list = file_list
        self.labels_df = labels_df
        self.is_test = is_test
        self.augment = augment
        
        print(f"Pre-loading {len(file_list)} files into memory...")
        self.data_cache = []
        for fpath in tqdm(file_list, desc="Caching Data"):
            df = pd.read_csv(fpath)
            xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
            self.data_cache.append(xyz)

    def __len__(self):
        return len(self.file_list)

    def get_rotation_matrix(self, vector):
        x, y, _ = vector
        mag_xy = np.sqrt(x**2 + y**2)
        if mag_xy < 1e-6: return np.eye(3, dtype=np.float32)
        cos_theta = x / mag_xy
        sin_theta = y / mag_xy
        return np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]], dtype=np.float32)

    def __getitem__(self, idx):
        xyz = self.data_cache[idx].copy()
        
        # ⚡ Data Augmentation (Training only)
        if self.augment:
            # Random position shift
            shift = np.random.uniform(-AUG_POSITION_SHIFT, AUG_POSITION_SHIFT, size=(1, 3))
            xyz += shift
            # Random noise
            noise = np.random.normal(0, AUG_NOISE_STD, size=xyz.shape).astype(np.float32)
            xyz += noise

        vel = np.diff(xyz, axis=0, prepend=xyz[0:1])
        accel = np.diff(vel, axis=0, prepend=vel[0:1])
        
        last_move = xyz[-1] - xyz[-2]
        rot_mat = self.get_rotation_matrix(last_move)
        
        last_xyz = xyz[-1]
        xyz_norm = (xyz - last_xyz) @ rot_mat.T
        vel_norm = vel @ rot_mat.T
        accel_norm = accel @ rot_mat.T
        
        v_mag = np.linalg.norm(vel, axis=1, keepdims=True)
        v_mag_safe = np.where(v_mag < 1e-6, 1e-6, v_mag)
        cross_prod = np.cross(vel, accel)
        curvature = np.linalg.norm(cross_prod, axis=1, keepdims=True) / (v_mag_safe**3)
        curvature = np.clip(curvature, 0, 100)
        speed = v_mag
        vel_prev = np.roll(vel, 1, axis=0); vel_prev[0] = vel[0]
        v_mag_prev = np.roll(v_mag, 1, axis=0); v_mag_prev[0] = v_mag[0]
        dot_prod = np.sum(vel * vel_prev, axis=1, keepdims=True)
        cos_sim = dot_prod / (v_mag_safe * (v_mag_prev + 1e-6))
        turn_angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))
        
        features = np.concatenate([xyz_norm, vel_norm, accel_norm, curvature, speed, turn_angle], axis=1)
        cv_prior_rel = (2.0 * last_move) @ rot_mat.T
        
        item = {
            "id": self.file_list[idx].stem,
            "seq": torch.tensor(features, dtype=torch.float32),
            "cv_prior": torch.tensor(cv_prior_rel, dtype=torch.float32),
            "last_pos": torch.tensor(last_xyz, dtype=torch.float32),
            "rot_mat": torch.tensor(rot_mat, dtype=torch.float32)
        }
        
        if not self.is_test and self.labels_df is not None:
            label = self.labels_df.loc[self.labels_df['id'] == self.file_list[idx].stem, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
            target_rel = (label - last_xyz) @ rot_mat.T
            residual = (target_rel - cv_prior_rel) * TARGET_SCALE
            item["target"] = torch.tensor(label, dtype=torch.float32)
            item["residual"] = torch.tensor(residual, dtype=torch.float32)
            
        return item

def get_dataloaders(batch_size=1024):
    if METADATA_PATH.exists():
        df = pd.read_csv(METADATA_PATH)
    else:
        step4_metadata = Path(__file__).parent.parent / 'step4' / 'metadata.csv'
        if step4_metadata.exists():
            df = pd.read_csv(step4_metadata)
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
    
    return DataLoader(GMMDataset(train_files, labels_df, augment=True), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True), \
           DataLoader(GMMDataset(test_files, labels_df, augment=False), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
