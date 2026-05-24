import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from step11.physics import make_candidates, CANDIDATES, EPS

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

def calculate_advanced_features(cand, s7_pos, p0):
    cand_vec = cand - p0
    s7_vec = s7_pos - p0
    
    cand_norm = np.linalg.norm(cand_vec)
    s7_norm = np.linalg.norm(s7_vec)
    
    # Angular similarity
    cos_sim = np.sum(cand_vec * s7_vec) / (cand_norm * s7_norm + EPS)
    
    dist_to_s7 = np.linalg.norm(cand - s7_pos)
    z_diff = abs(cand[2] - s7_pos[2])
    
    return {
        "dist_to_s7": dist_to_s7,
        "angle_to_s7": cos_sim,
        "z_diff_to_s7": z_diff,
        "dist_to_s7_rel": dist_to_s7 / (cand_norm + EPS)
    }

def prepare_tabular_data(limit=10000):
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    s7_preds_df = pd.read_csv("step10/step7_preds_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()[:limit]
    rows = []
    
    print(f"Preparing Step 11 tabular data (N={len(CANDIDATES)} candidates)...")
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        p0 = xyz[-1]
        
        s7_pos = s7_preds_df.loc[fid].to_numpy()
        ctx = extract_context_features(xyz)
        cands = make_candidates(xyz, end_idx=-1, horizon=2)
        
        dists = np.linalg.norm(cands - target, axis=1)
        
        # 1. Best Candidate
        best_idx = np.argmin(dists)
        
        # 2. Negatives (More diverse: some near target, some random)
        neg_indices = np.where(dists > 0.01)[0]
        if len(neg_indices) > 12:
            random_negs = np.random.choice(neg_indices, 10, replace=False)
            near_misses = neg_indices[np.argsort(dists[neg_indices])[:5]]
            selected_indices = list(set([best_idx]) | set(random_negs) | set(near_misses))
        else:
            selected_indices = range(len(CANDIDATES))
        
        for idx in selected_indices:
            spec = CANDIDATES[idx]
            cand_pos = cands[idx]
            adv = calculate_advanced_features(cand_pos, s7_pos, p0)
            
            row = {
                "id": fid,
                "cand_idx": idx,
                "spec_d1": spec.d1,
                "spec_par": spec.par,
                "spec_perp": spec.perp,
                "spec_ts": spec.ts,
                **adv,
                **ctx,
                "dist_to_p0": np.linalg.norm(cand_pos - p0),
                "target": 1 if dists[idx] <= 0.01 else 0
            }
            rows.append(row)
            
    out_df = pd.DataFrame(rows)
    out_path = Path("step11/train_ranker.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Step 11 tabular data saved to {out_path} ({len(out_df)} rows)")

if __name__ == "__main__":
    prepare_tabular_data()
