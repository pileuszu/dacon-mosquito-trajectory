import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(os.getcwd())
from step16.physics import make_candidates, CANDIDATES_GLOBAL

EPS = 1e-8

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

def calculate_prior_features(cand, s7_pos, p0):
    cand_vec = cand - p0
    s7_vec = s7_pos - p0
    
    cand_norm = np.linalg.norm(cand_vec)
    s7_norm = np.linalg.norm(s7_vec)
    
    cos_sim = np.sum(cand_vec * s7_vec) / (cand_norm * s7_norm + EPS)
    dist_to_s7 = np.linalg.norm(cand - s7_pos)
    z_diff = abs(cand[2] - s7_pos[2])
    
    return {
        "dist_to_s7": dist_to_s7,
        "angle_to_s7": cos_sim,
        "z_diff_to_s7": z_diff,
        "dist_to_s7_rel": dist_to_s7 / (cand_norm + EPS)
    }

def prepare_hybrid_physics_data_v20():
    data_dir = Path("data/open")
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    
    # Using 100% CLEAN OOF Predictions for Step 7 prior (Discard Step 4 entirely!)
    s7_oof_df = pd.read_csv("step17_oof/step7_oof_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()
    rows = []
    
    # Standard competition threshold (1.0cm = 0.01m)
    TARGET_THRESHOLD = 0.01
    
    print(f"Preparing Step 20 HYBRID PHYSICS Data (Threshold: {TARGET_THRESHOLD}m, 100% Clean s7 OOF Prior)...")
    
    for fid in tqdm(sample_ids):
        fpath = data_dir / "train" / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz_full = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        
        # Load OOF prior
        s7_pos = s7_oof_df.loc[fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)
        
        # 3x Augmentation for robust generalizability
        total_len = len(xyz_full)
        if total_len >= 50:
            windows = [xyz_full[0:40], xyz_full[5:45], xyz_full[10:50]]
        else:
            windows = [xyz_full]
            
        for w_idx, xyz in enumerate(windows):
            if len(xyz) < 5: continue
            
            p0 = xyz[-1]
            ctx = extract_context_features(xyz)
            
            # Pass only Step 7 prior (Step 4 prior is None to discard it)
            cands = make_candidates(xyz, priors=[s7_pos, None], horizon=2)
            
            dists = np.linalg.norm(cands - target, axis=1)
            best_idx = np.argmin(dists)
            
            # Diverse negative sampling (Near misses + random)
            neg_indices = np.where(dists > TARGET_THRESHOLD)[0]
            if len(neg_indices) > 30:
                near_misses = neg_indices[np.argsort(dists[neg_indices])[:15]]
                random_negs = np.random.choice(neg_indices, 15, replace=False)
                selected_indices = list(set([best_idx]) | set(near_misses) | set(random_negs))
            else:
                selected_indices = range(len(cands))
                
            for idx in selected_indices:
                cand_pos = cands[idx]
                adv = calculate_prior_features(cand_pos, s7_pos, p0)
                
                row = {
                    "id": f"{fid}_w{w_idx}",
                    "cand_idx": idx,
                    "is_global": 1 if idx < len(CANDIDATES_GLOBAL) else 0,
                    **ctx,
                    **adv,
                    "dist_to_p0": np.linalg.norm(cand_pos - p0),
                    "target": 1 if dists[idx] <= TARGET_THRESHOLD else 0
                }
                
                # Restore global grid parameters
                if idx < len(CANDIDATES_GLOBAL):
                    spec = CANDIDATES_GLOBAL[idx]
                    row.update({
                        "spec_par": spec.par, 
                        "spec_perp": spec.perp, 
                        "spec_ts": spec.ts,
                        "spec_jerk": spec.jerk
                    })
                else:
                    row.update({
                        "spec_par": 0.0, 
                        "spec_perp": 0.0, 
                        "spec_ts": 1.0,
                        "spec_jerk": 0.0
                    })
                    
                rows.append(row)
                
    out_df = pd.DataFrame(rows)
    out_path = Path("step20_hybrid_physics/train_ranker_v20.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    
    positives = out_df['target'].sum()
    print(f"Step 20 Data saved to {out_path} ({len(out_df)} rows)")
    print(f"Positive samples (<= 1.0cm): {positives} ({positives/len(out_df):.2%})")

if __name__ == "__main__":
    prepare_hybrid_physics_data_v20()
