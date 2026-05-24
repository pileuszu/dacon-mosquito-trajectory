import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from .config import *
from .physics import make_candidates, CANDIDATES, EPS

class CandidateDataset(Dataset):
    def __init__(self, file_list, labels_df=None, is_test=False, augment=False):
        self.file_list = file_list
        self.labels_df = labels_df
        self.is_test = is_test
        self.augment = augment
        
        print(f"Pre-loading {len(file_list)} files for Candidate Selector...")
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
        if mag_xy < 1e-5: return np.eye(3, dtype=np.float32)
        cos_theta = x / mag_xy
        sin_theta = y / mag_xy
        return np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]], dtype=np.float32)

    def make_candidate_features(self, xyz, end_idx, cands, horizon=2):
        p0 = xyz[end_idx]
        d1 = xyz[end_idx] - xyz[end_idx - 1]
        speed = np.linalg.norm(d1)
        tangent = d1 / (speed + EPS)
        
        delta = cands - p0 
        par = np.sum(delta * tangent, axis=1, keepdims=True)
        perp_vec = delta - par * tangent
        perp = np.linalg.norm(perp_vec, axis=1, keepdims=True)
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        
        scale = max(speed * float(horizon), EPS)
        
        # 9 Features: [rel_par, rel_perp, rel_dist, d1_coeff, par_coeff, perp_coeff, d2_coeff, jerk_coeff, time_scale]
        rel_geo = np.concatenate([par / scale, perp / scale, dist / scale], axis=1)
        spec_feats = np.array([[s.d1, s.par, s.perp, s.d2, s.jerk, s.time_scale] for s in CANDIDATES], dtype=np.float32)
        
        return np.concatenate([rel_geo, spec_feats], axis=1).astype(np.float32)

    def __getitem__(self, idx):
        xyz = self.data_cache[idx].copy()
        if self.augment:
            noise = np.random.normal(0, 0.0003, size=xyz.shape).astype(np.float32)
            xyz += noise

        vel = np.diff(xyz, axis=0, prepend=xyz[0:1])
        accel = np.diff(vel, axis=0, prepend=vel[0:1])
        
        last_move = xyz[-1] - xyz[-2]
        rot_mat = self.get_rotation_matrix(last_move)
        last_xyz = xyz[-1]
        
        # Seq features for Encoder (Same as successful version)
        xyz_norm = (xyz - last_xyz) @ rot_mat.T
        vel_norm = vel @ rot_mat.T
        accel_norm = accel @ rot_mat.T
        
        v_mag = np.linalg.norm(vel, axis=1, keepdims=True)
        v_mag_safe = np.clip(v_mag, 1e-4, 10.0)
        cross_va = np.cross(vel, accel)
        curvature = np.linalg.norm(cross_va, axis=1, keepdims=True) / (v_mag_safe**3 + 1e-6)
        curvature = np.clip(curvature, 0, 100)
        
        seq_features = np.concatenate([
            xyz_norm, vel_norm, accel_norm, 
            curvature, v_mag, np.zeros_like(v_mag) 
        ], axis=1)
        seq_features = np.nan_to_num(seq_features, nan=0.0)
        
        cands = make_candidates(xyz, end_idx=-1, horizon=2)
        cand_feats = self.make_candidate_features(xyz, -1, cands) # (81, 9)
        
        item = {
            "id": self.file_list[idx].stem,
            "seq": torch.tensor(seq_features, dtype=torch.float32),
            "candidates": torch.tensor(cands, dtype=torch.float32),
            "cand_feats": torch.tensor(cand_feats, dtype=torch.float32),
            "rot_mat": torch.tensor(rot_mat, dtype=torch.float32)
        }
        
        if not self.is_test and self.labels_df is not None:
            target = self.labels_df.loc[self.labels_df['id'] == self.file_list[idx].stem, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
            dists = np.linalg.norm(cands - target, axis=1)
            oracle_hit = (np.min(dists) <= R_HIT)
            tau = 0.0045
            scores = -dists / tau
            scores += (dists <= R_HIT).astype(np.float32) * 0.75
            prob_targets = np.exp(scores - np.max(scores))
            prob_targets /= prob_targets.sum()
            item["target"] = torch.tensor(target, dtype=torch.float32)
            item["prob_targets"] = torch.tensor(prob_targets, dtype=torch.float32)
            item["best_cand_idx"] = torch.tensor(np.argmin(dists), dtype=torch.long)
            item["oracle_hit"] = torch.tensor(oracle_hit, dtype=torch.float32)
            
        return item

def get_dataloaders(batch_size=1024):
    if METADATA_PATH.exists():
        df = pd.read_csv(METADATA_PATH)
    else:
        s6_meta = Path(__file__).parent.parent / 'step6' / 'metadata.csv'
        df = pd.read_csv(s6_meta)
        df.to_csv(METADATA_PATH, index=False)
    train_ids = df[df['split'] == 'train']['id'].tolist()
    test_ids = df[df['split'] == 'test']['id'].tolist()
    train_files = [TRAIN_DIR / f"{fid}.csv" for fid in train_ids]
    test_files = [TRAIN_DIR / f"{fid}.csv" for fid in test_ids]
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    return DataLoader(CandidateDataset(train_files, labels_df, augment=True), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True), \
           DataLoader(CandidateDataset(test_files, labels_df, augment=False), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
