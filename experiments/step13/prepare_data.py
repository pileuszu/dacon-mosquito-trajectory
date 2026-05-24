import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from step13.physics import make_candidates, CANDIDATES_GLOBAL, EPS

def extract_advanced_context(xyz):
    vel = np.diff(xyz, axis=0)
    acc = np.diff(vel, axis=0)
    
    last_vel = vel[-1]
    last_acc = acc[-1]
    
    return {
        "ctx_speed": np.linalg.norm(last_vel),
        "ctx_acc_norm": np.linalg.norm(last_acc),
        "ctx_z_vel": last_vel[2],
        "last_vel_vec": last_vel,
        "last_acc_vec": last_acc
    }

def calculate_physics_features(cand, p0, ctx):
    # Predicted velocity if we go to cand (dt=2)
    # v_next = (cand - p0) / 2
    v_pred = (cand - p0) / 2.0
    a_pred = (v_pred - ctx['last_vel_vec']) / 1.0 # simplistic dt
    
    # Jerk-like: Change in acceleration
    jerk_norm = np.linalg.norm(a_pred - ctx['last_acc_vec'])
    speed_diff = abs(np.linalg.norm(v_pred) - ctx['ctx_speed'])
    
    return {
        "phys_jerk": jerk_norm,
        "phys_speed_diff": speed_diff,
        "phys_dist_p0": np.linalg.norm(cand - p0)
    }

def calculate_prior_features(cand, prior_pos, p0, prefix="s7"):
    dist = np.linalg.norm(cand - prior_pos)
    cand_vec = cand - p0
    prior_vec = prior_pos - p0
    cos_sim = np.sum(cand_vec * prior_vec) / (np.linalg.norm(cand_vec) * np.linalg.norm(prior_vec) + EPS)
    
    return {
        f"dist_to_{prefix}": dist,
        f"angle_to_{prefix}": cos_sim,
        f"z_diff_to_{prefix}": abs(cand[2] - prior_pos[2])
    }

def prepare_tabular_data_v13():
    data_dir = Path("data/open")
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    s7_preds_df = pd.read_csv("step10/step7_preds_train.csv").set_index('id')
    s4_preds_df = pd.read_csv("step12/step4_preds_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()
    rows = []
    
    print(f"Preparing Step 13 Tabular Data (Extreme Precision)...")
    for fid in tqdm(sample_ids):
        fpath = data_dir / "train" / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        p0 = xyz[-1]
        
        s7_pos = s7_preds_df.loc[fid].to_numpy()
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        
        ctx = extract_advanced_context(xyz)
        cands = make_candidates(xyz, priors=[s7_pos, s4_pos], horizon=2)
        
        dists = np.linalg.norm(cands - target, axis=1)
        best_idx = np.argmin(dists)
        
        # Smart Sampling
        neg_indices = np.where(dists > 0.01)[0]
        near_misses = neg_indices[np.argsort(dists[neg_indices])[:10]]
        random_negs = np.random.choice(neg_indices, 10, replace=False)
        selected_indices = list(set([best_idx]) | set(near_misses) | set(random_negs))
        
        s7_s4_dist = np.linalg.norm(s7_pos - s4_pos)
        
        for idx in selected_indices:
            cand_pos = cands[idx]
            phys = calculate_physics_features(cand_pos, p0, ctx)
            feat_s7 = calculate_prior_features(cand_pos, s7_pos, p0, "s7")
            feat_s4 = calculate_prior_features(cand_pos, s4_pos, p0, "s4")
            
            rows.append({
                "id": fid,
                "cand_idx": idx,
                "is_local": 1 if idx >= len(CANDIDATES_GLOBAL) else 0,
                **phys,
                **feat_s7,
                **feat_s4,
                "ctx_speed": ctx['ctx_speed'],
                "ctx_acc_norm": ctx['ctx_acc_norm'],
                "s7_s4_dist": s7_s4_dist,
                "target": 1 if dists[idx] <= 0.01 else 0
            })
            
    out_df = pd.DataFrame(rows)
    out_path = Path("step13/train_ranker_v13.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Step 13 Data saved to {out_path} ({len(out_df)} rows)")

if __name__ == "__main__":
    prepare_tabular_data_v13()
