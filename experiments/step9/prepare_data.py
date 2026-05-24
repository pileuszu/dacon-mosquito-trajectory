import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from step9.physics import make_candidates, CANDIDATES, EPS

def extract_context_features(xyz):
    # xyz: (T, 3)
    vel = np.diff(xyz, axis=0)
    acc = np.diff(vel, axis=0)
    
    speed = np.linalg.norm(vel[-1])
    prev_speed = np.linalg.norm(vel[-2])
    acc_norm = np.linalg.norm(acc[-1])
    
    # Curvature
    cross_va = np.cross(vel[-1], acc[-1])
    curv = np.linalg.norm(cross_va) / (speed**3 + 1e-6)
    
    # Turn angle
    cos_theta = np.sum(vel[-1] * vel[-2]) / (speed * prev_speed + EPS)
    
    # Z-movement
    z_vel = vel[-1, 2]
    
    return {
        "ctx_speed": speed,
        "ctx_acc": acc_norm,
        "ctx_curv": curv,
        "ctx_turn": cos_theta,
        "ctx_z_vel": z_vel
    }

def prepare_tabular_data(limit=10000):
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    
    sample_ids = labels_df['id'].unique()[:limit]
    
    rows = []
    
    print(f"Preparing lean tabular data for {len(sample_ids)} samples...")
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        
        ctx = extract_context_features(xyz)
        cands = make_candidates(xyz, end_idx=-1, horizon=2)
        dists = np.linalg.norm(cands - target, axis=1)
        
        # 1. Best Candidate (even if not hit, it's our best target)
        best_idx = np.argmin(dists)
        
        # 2. Random Negatives (indices not the best one and not hits)
        neg_indices = np.where(dists > 0.01)[0]
        if len(neg_indices) > 10:
            random_negs = np.random.choice(neg_indices, 10, replace=False)
        else:
            random_negs = neg_indices
            
        # 3. Near-misses (top 5 among negatives)
        near_miss_indices = neg_indices[np.argsort(dists[neg_indices])[:5]]
        
        selected_indices = list(set([best_idx]) | set(random_negs) | set(near_miss_indices))
        
        for idx in selected_indices:
            spec = CANDIDATES[idx]
            dist = dists[idx]
            is_hit = 1 if dist <= 0.01 else 0
            
            row = {
                "id": fid,
                "cand_idx": idx,
                "spec_par": spec.par,
                "spec_perp": spec.perp,
                "spec_ts": spec.time_scale,
                "spec_jerk": spec.jerk,
                **ctx,
                "dist_to_p0": np.linalg.norm(cands[idx] - xyz[-1]),
                "target": is_hit,
                "reg_target": dist
            }
            rows.append(row)
            
    out_df = pd.DataFrame(rows)
    out_path = Path("step9/train_ranker_balanced.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Lean tabular data saved to {out_path} ({len(out_df)} rows)")

if __name__ == "__main__":
    prepare_tabular_data()
