import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(os.getcwd())
from step27_advanced_dynamics.physics import make_candidates, extract_multi_scale_derivatives, EPS

CRUISING_CENTROID = np.array([0.0120, 0.0035, 0.0020, 0.0040, 0.0010], dtype=np.float32)
SACCADIC_CENTROID = np.array([0.0350, 0.0145, 0.0110, 0.0125, 0.0055], dtype=np.float32)

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
    
    # Calculate perpendicular acceleration
    tangent = last_vel / (speed + EPS)
    acc_par_scalar = np.sum(acc[-1] * tangent)
    acc_perp_vec = acc[-1] - acc_par_scalar * tangent
    perp_acc = np.linalg.norm(acc_perp_vec)
    
    # Behavior profiles
    feat = np.array([speed, acc_norm, perp_acc, np.abs(z_vel), np.abs(z_acc)], dtype=np.float32)
    dist_to_cruising = np.linalg.norm(feat - CRUISING_CENTROID)
    dist_to_saccadic = np.linalg.norm(feat - SACCADIC_CENTROID)
    
    raw_ctx = {
        "ctx_speed": speed,
        "ctx_acc": acc_norm,
        "ctx_curv": curv,
        "ctx_turn": cos_theta,
        "ctx_z_vel": z_vel,
        "ctx_z_acc": z_acc,
        "ctx_speed_ratio": speed_ratio,
        "dist_to_cruising": dist_to_cruising,
        "dist_to_saccadic": dist_to_saccadic
    }
    
    # Extract multi-scale smoothed derivatives (W3-Quad, W5-Quad, W5-Cubic)
    smooth_ctx = extract_multi_scale_derivatives(xyz)
    
    return {**raw_ctx, **smooth_ctx}

def prepare_step27_data():
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    
    # Load EqMotion predictions for train set
    print("Loading EqMotion train predictions...")
    s4_preds_df = pd.read_csv("step12/step4_preds_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()
    rows = []
    
    TARGET_THRESHOLD = 0.01  # 1.0cm
    
    print("Preparing Step 27 Advanced Dynamics Ranker Data...")
    
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        
        # Calculate Priors
        p0 = xyz[-1]
        last_vel = xyz[-1] - xyz[-2]
        s7_pos = p0 + 2.0 * last_vel
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        priors = [s7_pos, s4_pos]
        
        # Extract features and candidates
        ctx = extract_context_features(xyz)
        cands, cands_list = make_candidates(xyz, priors=priors, end_idx=-1, horizon=2)
        
        dists = np.linalg.norm(cands - target, axis=1)
        
        # Determine candidate list size (Slow vs Fast)
        num_physical = len(cands_list) - 2
        
        if num_physical <= 50:
            # Slow: Keep all candidates
            selected_indices = list(range(len(cands_list)))
        else:
            # Fast: Balance candidate list to avoid target dilution
            best_idx = np.argmin(dists)
            neg_indices = np.where(dists > TARGET_THRESHOLD)[0]
            
            if len(neg_indices) > 15:
                random_negs = np.random.choice(neg_indices, 15, replace=False)
            else:
                random_negs = neg_indices
                
            near_miss_indices = neg_indices[np.argsort(dists[neg_indices])[:5]]
            # Always ensure prior candidates (last 2 indices) are included in the balanced training data
            prior_indices = [len(cands_list) - 2, len(cands_list) - 1]
            
            selected_indices = list(set([best_idx]) | set(random_negs) | set(near_miss_indices) | set(prior_indices))
            
        for idx in selected_indices:
            spec = cands_list[idx]
            cand_pos = cands[idx]
            dist = dists[idx]
            
            is_prior_val = 1 if idx >= num_physical else 0
            
            row = {
                "id": fid,
                "cand_idx": idx,
                "spec_par": spec.par,
                "spec_perp": spec.perp,
                "spec_ts": spec.time_scale,
                "spec_dmp": spec.damping,
                "spec_jerk": spec.jerk,
                "is_prior": is_prior_val,
                **ctx,
                "dist_to_p0": np.linalg.norm(cand_pos - p0),
                "dist_to_s7": np.linalg.norm(cand_pos - s7_pos),
                "dist_to_s4": np.linalg.norm(cand_pos - s4_pos),
                "target": 1 if dist <= TARGET_THRESHOLD else 0,
                "reg_target": dist
            }
            rows.append(row)
            
    out_df = pd.DataFrame(rows)
    out_path = Path("step27_advanced_dynamics/train_ranker_v27.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    
    positives = out_df['target'].sum()
    print(f"Step 27 Data saved successfully to {out_path} ({len(out_df)} rows)")
    print(f"Positive samples: {positives} ({positives/len(out_df):.2%})")

if __name__ == "__main__":
    prepare_step27_data()
