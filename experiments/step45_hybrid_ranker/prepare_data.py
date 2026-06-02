import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os
import pickle

sys.path.append(os.getcwd())
from step45_hybrid_ranker.physics import make_candidates, extract_multi_scale_derivatives, EPS

CLUSTER_FEATURES = [
    "ctx_speed",
    "ctx_acc",
    "smooth_curv_w5",
    "ctx_lat_accel",
    "ctx_z_vel",
    "ctx_z_acc",
    "ctx_p_saccade",
    "roll_speed_cv_all"
]

REGIME_MAPPING = {
    0: "fast_straight_low",
    1: "slow_moderate_turning",
    2: "fast_moderate_turning",
    3: "fast_straight_high",
    4: "fast_extreme_turning",
    5: "slow_extreme_turning"
}

def get_group_regime(gmm_idx):
    if gmm_idx == 0:
        return "fast_straight_low"
    elif gmm_idx == 1:
        return "slow_moderate_turning"
    elif gmm_idx == 3:
        return "fast_straight_high"
    else: # 2, 4, 5
        return "steering"

def extract_context_features(xyz):
    ctx = extract_multi_scale_derivatives(xyz)
    
    vel = np.diff(xyz, axis=0)
    acc = np.diff(vel, axis=0)
    last_vel = vel[-1]
    speed = np.linalg.norm(last_vel)
    prev_speed = np.linalg.norm(vel[-2])
    
    cross_va = np.cross(last_vel, acc[-1])
    curv = np.linalg.norm(cross_va) / (speed**3 + 1e-6)
    cos_theta = np.sum(vel[-1] * vel[-2]) / (speed * prev_speed + EPS)
    
    ctx["ctx_curv"] = curv
    ctx["ctx_turn"] = cos_theta
    
    return ctx

def prepare_step45_data_with_params(denoise_threshold=0.8):
    print(f"\n=== Step 45: Preparing Hybrid Datasets with Denoising Threshold = {denoise_threshold} ===")
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv").set_index('id')
    
    # Load EqMotion train predictions
    print("Loading EqMotion train predictions...")
    s4_preds_df = pd.read_csv("experiments/step12/step4_preds_train.csv").set_index('id')
    
    sample_ids = labels_df.index.unique()
    
    # Load GMM-6 components from step39 to ensure consistent cluster routing
    models_dir = Path("step39_six_regime/models")
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "gmm_model.pkl", "rb") as f:
        gmm = pickle.load(f)

    # Load BGM-12 components from step40
    bgm_dir = Path("step40_dual_specialized/models/bgm")
    with open(bgm_dir / "scaler.pkl", "rb") as f:
        bgm_scaler = pickle.load(f)
    with open(bgm_dir / "bgm_model.pkl", "rb") as f:
        bgm = pickle.load(f)
        
    print("Extracting features and predicting GMM-6 & BGM-12 probabilities...")
    traj_features = []
    traj_ids = []
    
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        ctx = extract_context_features(xyz)
        
        feat_vector = [ctx[feat] for feat in CLUSTER_FEATURES]
        traj_features.append(feat_vector)
        traj_ids.append(fid)
        
    traj_features = np.array(traj_features, dtype=np.float32)
    features_scaled = scaler.transform(traj_features)
    bgm_features_scaled = bgm_scaler.transform(traj_features)
    
    gmm_probs = gmm.predict_proba(features_scaled)
    clusters = gmm.predict(features_scaled)
    
    bgm_probs = bgm.predict_proba(bgm_features_scaled)
    bgm_clusters = bgm.predict(bgm_features_scaled)
    
    max_probs = gmm_probs.max(axis=1)
    
    clear_mask = max_probs >= denoise_threshold
    num_clear = clear_mask.sum()
    num_ambiguous = len(sample_ids) - num_clear
    print(f"Total Trajectories: {len(sample_ids)}")
    print(f"Clear Trajectories (Max Prob >= {denoise_threshold:.2f}): {num_clear} ({num_clear/len(sample_ids):.2%})")
    print(f"Ambiguous Trajectories: {num_ambiguous} ({num_ambiguous/len(sample_ids):.2%}) - EXCLUDED from training")
    
    traj_regimes = {traj_ids[i]: REGIME_MAPPING[clusters[i]] for i in range(len(traj_ids))}
    traj_group_regimes = {traj_ids[i]: get_group_regime(clusters[i]) for i in range(len(traj_ids))}
    traj_is_clear = {traj_ids[i]: clear_mask[i] for i in range(len(traj_ids))}
    
    traj_gmm_probs = {traj_ids[i]: gmm_probs[i] for i in range(len(traj_ids))}
    traj_bgm_probs = {traj_ids[i]: bgm_probs[i] for i in range(len(traj_ids))}
    traj_bgm_cluster = {traj_ids[i]: bgm_clusters[i] for i in range(len(traj_ids))}
    traj_gmm_cluster = {traj_ids[i]: clusters[i] for i in range(len(traj_ids))}
    
    # Initialize containers for the 4 groups
    regime_rows = {
        "fast_straight_low": [],
        "slow_moderate_turning": [],
        "fast_straight_high": [],
        "steering": []
    }
    
    TARGET_THRESHOLD = 0.01  # 1.0cm
    
    print("\nPreparing Datasets with physics-explicit candidate pools...")
    for fid in tqdm(sample_ids):
        if not traj_is_clear[fid]:
            continue
            
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[fid].to_numpy(dtype=np.float32)
        
        p0 = xyz[-1]
        last_vel = xyz[-1] - xyz[-2]
        
        s7_pos = p0 + 2.0 * last_vel
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        priors = [s7_pos, s4_pos]
        
        regime = traj_regimes[fid]
        group_regime = traj_group_regimes[fid]
        
        ctx = extract_context_features(xyz)
        cands, cands_list, c_features = make_candidates(xyz, priors=priors, end_idx=-1, horizon=2, regime=regime)
        
        dists = np.linalg.norm(cands - target, axis=1)
        num_physical = len(cands_list) - 2
        
        if regime == "slow_moderate_turning":
            selected_indices = list(range(len(cands_list)))
        else:
            best_idx = np.argmin(dists)
            neg_indices = np.where(dists > TARGET_THRESHOLD)[0]
            
            if len(neg_indices) > 30:
                random_negs = np.random.choice(neg_indices, 30, replace=False)
            else:
                random_negs = neg_indices
                
            near_miss_indices = neg_indices[np.argsort(dists[neg_indices])[:10]]
            prior_indices = [len(cands_list) - 2, len(cands_list) - 1]
            
            selected_indices = list(set([best_idx]) | set(random_negs) | set(near_miss_indices) | set(prior_indices))
            
        for idx in selected_indices:
            spec = cands_list[idx]
            cand_pos = cands[idx]
            dist = dists[idx]
            
            is_prior_val = 1 if idx >= num_physical else 0
            is_hit = 1 if dist <= TARGET_THRESHOLD else 0
            
            weight = 15.0 if is_hit == 1 else 1.0
            
            row = {
                "id": fid,
                "cand_idx": idx,
                "regime": regime,
                "spec_par": spec.par,
                "spec_perp": spec.perp,
                "spec_binormal": spec.binormal,
                "spec_ts": spec.time_scale,
                "spec_dmp": spec.damping,
                "spec_jerk": spec.jerk,
                "is_prior": is_prior_val,
                "grid_scale": c_features["grid_scale"][idx],
                "cand_speed": c_features["cand_speed"][idx],
                "cand_speed_ratio": c_features["cand_speed_ratio"][idx],
                "cand_turn_angle": c_features["cand_turn_angle"][idx],
                "cand_turn_rate": c_features["cand_turn_rate"][idx],
                "cand_accel": c_features["cand_accel"][idx],
                "cand_lat_accel": c_features["cand_lat_accel"][idx],
                "cand_disp_normal": c_features["cand_disp_normal"][idx],
                "cand_disp_binormal": c_features["cand_disp_binormal"][idx],
                "cand_acc_binormal": c_features["cand_acc_binormal"][idx],
                **ctx,
            }
            
            for j in range(6):
                row[f"gmm_p{j}"] = traj_gmm_probs[fid][j]
            row["gmm_cluster"] = traj_gmm_cluster[fid]
            
            for k in range(12):
                row[f"bgm_p{k}"] = traj_bgm_probs[fid][k]
            row["bgm_cluster"] = traj_bgm_cluster[fid]
            
            row["dist_to_p0"] = np.linalg.norm(cand_pos - p0)
            row["dist_to_s7"] = np.linalg.norm(cand_pos - s7_pos)
            row["dist_to_s4"] = np.linalg.norm(cand_pos - s4_pos)
            row["target"] = is_hit
            row["reg_target"] = dist
            row["weight"] = weight
            
            regime_rows[group_regime].append(row)
            
    out_dir = Path("step45_hybrid_ranker/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for r_name, rows in regime_rows.items():
        df_reg = pd.DataFrame(rows)
        save_path = out_dir / f"train_ranker_v45_{r_name}_th{denoise_threshold}.csv"
        df_reg.to_csv(save_path, index=False)
        pos_count = df_reg['target'].sum() if 'target' in df_reg.columns else 0
        print(f"  * Saved {r_name} (Th={denoise_threshold}): {len(df_reg)} rows ({pos_count} positive) to {save_path}")

def main():
    # 1. Prepare validation datasets (threshold = 0.0)
    prepare_step45_data_with_params(denoise_threshold=0.0)
    
    # 2. Prepare train datasets (threshold = 0.8)
    prepare_step45_data_with_params(denoise_threshold=0.8)

if __name__ == "__main__":
    main()
