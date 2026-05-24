import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from step12.physics import make_candidates, CANDIDATES_GLOBAL, EPS

def extract_context_features(xyz):
    vel = np.diff(xyz, axis=0)
    acc = np.diff(vel, axis=0)
    last_vel = vel[-1]
    speed = np.linalg.norm(last_vel)
    acc_norm = np.linalg.norm(acc[-1])
    return {
        "ctx_speed": speed,
        "ctx_acc": acc_norm,
        "ctx_z_vel": last_vel[2],
        "ctx_z_acc": acc[-1, 2]
    }

def calculate_prior_features(cand, prior_pos, p0, prefix="s7"):
    cand_vec = cand - p0
    prior_vec = prior_pos - p0
    cand_norm = np.linalg.norm(cand_vec)
    prior_norm = np.linalg.norm(prior_vec)
    
    cos_sim = np.sum(cand_vec * prior_vec) / (cand_norm * prior_norm + EPS)
    dist = np.linalg.norm(cand - prior_pos)
    
    return {
        f"dist_to_{prefix}": dist,
        f"angle_to_{prefix}": cos_sim,
        f"z_diff_to_{prefix}": abs(cand[2] - prior_pos[2])
    }

def prepare_tabular_data_v12(limit=10000):
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    
    s7_preds_df = pd.read_csv("step10/step7_preds_train.csv").set_index('id')
    s4_preds_df = pd.read_csv("step12/step4_preds_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()[:limit]
    rows = []
    
    print(f"Preparing Step 12 tabular data with Dual Priors...")
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        p0 = xyz[-1]
        
        s7_pos = s7_preds_df.loc[fid].to_numpy()
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        
        ctx = extract_context_features(xyz)
        # Generate 614 candidates
        cands = make_candidates(xyz, priors=[s7_pos, s4_pos], horizon=2)
        
        dists = np.linalg.norm(cands - target, axis=1)
        best_idx = np.argmin(dists)
        
        # Sampling for training
        neg_indices = np.where(dists > 0.01)[0]
        if len(neg_indices) > 15:
            random_negs = np.random.choice(neg_indices, 10, replace=False)
            near_misses = neg_indices[np.argsort(dists[neg_indices])[:5]]
            selected_indices = list(set([best_idx]) | set(random_negs) | set(near_misses))
        else:
            selected_indices = range(len(cands))
            
        # Agreement feature
        s7_s4_dist = np.linalg.norm(s7_pos - s4_pos)
        
        for idx in selected_indices:
            cand_pos = cands[idx]
            feat_s7 = calculate_prior_features(cand_pos, s7_pos, p0, "s7")
            feat_s4 = calculate_prior_features(cand_pos, s4_pos, p0, "s4")
            
            row = {
                "id": fid,
                "cand_idx": idx,
                "is_local": 1 if idx >= len(CANDIDATES_GLOBAL) else 0,
                **feat_s7,
                **feat_s4,
                **ctx,
                "s7_s4_dist": s7_s4_dist,
                "dist_to_p0": np.linalg.norm(cand_pos - p0),
                "target": 1 if dists[idx] <= 0.01 else 0
            }
            rows.append(row)
            
    out_df = pd.DataFrame(rows)
    out_path = Path("step12/train_ranker.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Step 12 tabular data saved to {out_path} ({len(out_df)} rows)")

if __name__ == "__main__":
    prepare_tabular_data_v12()
