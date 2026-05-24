import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .config import *

class DiscreteDataset(Dataset):
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
        if self.augment:
            noise = np.random.normal(0, 0.001, size=xyz.shape).astype(np.float32)
            xyz += noise

        # 1. Basic Physics
        vel = np.diff(xyz, axis=0, prepend=xyz[0:1])
        accel = np.diff(vel, axis=0, prepend=vel[0:1])
        jerk = np.diff(accel, axis=0, prepend=accel[0:1]) # 3rd derivative
        
        # 2. Rotational Normalization
        last_move = xyz[-1] - xyz[-2]
        rot_mat = self.get_rotation_matrix(last_move)
        last_xyz = xyz[-1]
        
        xyz_norm = (xyz - last_xyz) @ rot_mat.T
        vel_norm = vel @ rot_mat.T
        accel_norm = accel @ rot_mat.T
        
        # 3. Bio-Physics Invariants (12 -> 18)
        v_mag = np.linalg.norm(vel, axis=1, keepdims=True)
        v_mag_safe = np.where(v_mag < 1e-6, 1e-6, v_mag)
        
        # [Existing 10-12]
        # Curvature
        cross_va = np.cross(vel, accel)
        curvature = np.linalg.norm(cross_va, axis=1, keepdims=True) / (v_mag_safe**3)
        # Speed
        speed = v_mag
        # Turning Angle
        vel_prev = np.roll(vel, 1, axis=0); vel_prev[0] = vel[0]
        v_mag_prev = np.roll(v_mag, 1, axis=0); v_mag_prev[0] = v_mag[0]
        cos_sim = np.sum(vel * vel_prev, axis=1, keepdims=True) / (v_mag_safe * (v_mag_prev + 1e-6))
        turn_angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))
        
        # [New Theoretical Features]
        # 13. Jerk Magnitude
        j_mag = np.linalg.norm(jerk, axis=1, keepdims=True)
        
        # 14. Torsion (3D Twist)
        # tau = |(v x a) . j| / |v x a|^2
        va_mag_sq = np.sum(cross_va**2, axis=1, keepdims=True)
        torsion = np.abs(np.sum(cross_va * jerk, axis=1, keepdims=True)) / (va_mag_sq + 1e-8)
        
        # 15. Kinetic Energy Flux (dK/dt)
        ke_flux = np.diff(0.5 * v_mag**2, axis=0, prepend=v_mag[0:1]**2)
        
        # 16. Angular Momentum Magnitude (relative to origin)
        # L = |r x v|
        ang_mom = np.linalg.norm(np.cross(xyz_norm, vel_norm), axis=1, keepdims=True)
        
        # 17. Centripetal Acceleration (perpendicular component of a)
        # a_perp = a - (a.v/v^2)v
        proj_av = np.sum(accel * vel, axis=1, keepdims=True) / (v_mag_safe**2) * vel
        centrip_accel = np.linalg.norm(accel - proj_av, axis=1, keepdims=True)
        
        # 18. Radial Velocity (velocity along the position vector)
        # v_rad = v . (r/|r|)
        r_mag = np.linalg.norm(xyz_norm, axis=1, keepdims=True)
        r_mag_safe = np.where(r_mag < 1e-6, 1e-6, r_mag)
        radial_vel = np.sum(vel_norm * (xyz_norm / r_mag_safe), axis=1, keepdims=True)
        
        features = np.concatenate([
            xyz_norm, vel_norm, accel_norm, # 9
            curvature, speed, turn_angle, # 12
            j_mag, torsion, ke_flux, ang_mom, centrip_accel, radial_vel # 18
        ], axis=1)
        
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
            residual = (target_rel - cv_prior_rel)
            bin_indices = np.round(residual / BIN_SIZE) + CENTER_BIN
            bin_indices = np.clip(bin_indices, 0, NUM_BINS - 1).astype(np.int64)
            offset = (residual - (bin_indices - CENTER_BIN) * BIN_SIZE) / BIN_SIZE
            item["target"] = torch.tensor(label, dtype=torch.float32)
            item["bin_labels"] = torch.tensor(bin_indices, dtype=torch.long)
            item["offsets"] = torch.tensor(offset, dtype=torch.float32)
            
        return item

def get_dataloaders(batch_size=1024):
    if METADATA_PATH.exists():
        df = pd.read_csv(METADATA_PATH)
    else:
        df = pd.read_csv(Path(__file__).parent.parent / 'step6' / 'metadata.csv')
        df.to_csv(METADATA_PATH, index=False)
    train_ids = df[df['split'] == 'train']['id'].tolist()
    test_ids = df[df['split'] == 'test']['id'].tolist()
    train_files = [TRAIN_DIR / f"{fid}.csv" for fid in train_ids]
    test_files = [TRAIN_DIR / f"{fid}.csv" for fid in test_ids]
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    return DataLoader(DiscreteDataset(train_files, labels_df, augment=True), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True), \
           DataLoader(DiscreteDataset(test_files, labels_df, augment=False), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
