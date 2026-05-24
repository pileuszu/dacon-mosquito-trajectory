import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(os.getcwd())
from step25_adaptive_grid.physics import make_candidates, EPS

def extract_context_features(xyz):
    vel = np.diff(xyz, axis=0)
    acc = np.diff(vel, axis=0)
    
    last_vel = vel[-1]
    speed = np.linalg.norm(last_vel)
    prev_speed = np.linalg.norm(vel[-2])
    
    acc_norm = np.linalg.norm(acc[-1])
    z_vel = last_vel[2]
    z_acc = acc[-1, 2]
    
    speed_ratio = speed / (prev_speed + EPS)
    
    cross_va = np.cross(last_vel, acc[-1])
    curv = np.linalg.norm(cross_va) / (speed**3 + 1e-6)
    
    cos_theta = np.sum(vel[-1] * vel[-2]) / (speed * prev_speed + EPS)
    
    return {
        "ctx_speed": speed,
        "ctx_acc": acc_norm,
        "ctx_curv": curv,
        "ctx_turn": cos_theta,
        "ctx_z_vel": z_vel,
        "ctx_z_acc": z_acc,
        "ctx_speed_ratio": speed_ratio
    }

def prepare_adaptive_grid_data():
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    
    sample_ids = labels_df['id'].unique()
    rows = []
    
    TARGET_THRESHOLD = 0.01  # 1.0cm hit threshold
    
    print(f"Preparing Step 25 Speed-Adaptive Data (Threshold: {TARGET_THRESHOLD}m)...")
    
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        
        p0 = xyz[-1]
        ctx = extract_context_features(xyz)
        cands, cands_list = make_candidates(xyz, end_idx=-1, horizon=2)
        
        dists = np.linalg.norm(cands - target, axis=1)
        
        if len(cands_list) <= 50:
            # Slow cruising: use all candidates (20) to fully map low-entropy state
            selected_indices = list(range(len(cands_list)))
        else:
            # Fast turning: balance dataset to prevent target distortion
            best_idx = np.argmin(dists)
            neg_indices = np.where(dists > TARGET_THRESHOLD)[0]
            
            if len(neg_indices) > 10:
                random_negs = np.random.choice(neg_indices, 10, replace=False)
            else:
                random_negs = neg_indices
                
            near_miss_indices = neg_indices[np.argsort(dists[neg_indices])[:5]]
            selected_indices = list(set([best_idx]) | set(random_negs) | set(near_miss_indices))
            
        for idx in selected_indices:
            spec = cands_list[idx]
            cand_pos = cands[idx]
            dist = dists[idx]
            
            row = {
                "id": fid,
                "cand_idx": idx,
                "spec_par": spec.par,
                "spec_perp": spec.perp,
                "spec_ts": spec.time_scale,
                "spec_dmp": spec.damping,
                "spec_jerk": spec.jerk,
                **ctx,
                "dist_to_p0": np.linalg.norm(cand_pos - p0),
                "target": 1 if dist <= TARGET_THRESHOLD else 0,
                "reg_target": dist
            }
            rows.append(row)
            
    out_df = pd.DataFrame(rows)
    out_path = Path("step25_adaptive_grid/train_ranker_v25.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    
    positives = out_df['target'].sum()
    print(f"Step 25 Data saved to {out_path} ({len(out_df)} rows)")
    print(f"Positive samples (<= 1.0cm): {positives} ({positives/len(out_df):.2%})")

if __name__ == "__main__":
    prepare_adaptive_grid_data()
