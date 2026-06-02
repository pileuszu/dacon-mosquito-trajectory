import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os
import pickle

sys.path.append(os.getcwd())
from step38_2regime_regression.physics import make_candidates, extract_multi_scale_derivatives, EPS

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

def prepare_step38_data():
    np.random.seed(42)
    
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    
    # Load EqMotion train predictions for priors
    print("Loading EqMotion train predictions...")
    s4_preds_df = pd.read_csv("experiments/step12/step4_preds_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()
    
    # Load GMM and scaler from step35
    models_dir = Path("experiments/step35_four_regime/models")
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "gmm_model.pkl", "rb") as f:
        gmm = pickle.load(f)
    with open(models_dir / "regime_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
        
    print("Loaded GMM and scaling models. Mapping:", mapping)
    
    # First Pass: Extract context features and predict GMM regimes
    print("Assigning trajectories to GMM regimes...")
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
    clusters = gmm.predict(features_scaled)
    
    traj_regimes = {traj_ids[i]: mapping[clusters[i]] for i in range(len(traj_ids))}
    
    # Second Pass: Generate customized continuous candidates with Uniform Distance Bin Sampling
    regime_rows = {
        "slow_straight": [],
        "slow_extreme_turning": [],
        "fast_straight": [],
        "fast_turning": []
    }
    
    TARGET_THRESHOLD = 0.01  # 1.0cm
    
    # To monitor candidate generator hit rates (coverage)
    regime_total_counts = {r: 0 for r in regime_rows}
    regime_hit_counts = {r: 0 for r in regime_rows}
    
    print("\nGenerating continuous candidates with Uniform Distance Sampling...")
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
        
        regime = traj_regimes[fid]
        
        # Generate candidates using routed GMM regime
        cands, cands_list, c_features = make_candidates(xyz, priors=priors, end_idx=-1, horizon=2, regime=regime)
        
        dists = np.linalg.norm(cands - target, axis=1)
        num_physical = len(cands_list) - 2
        
        # Monitor coverage (is there ANY candidate <= 1cm from target?)
        has_hit = np.any(dists <= TARGET_THRESHOLD)
        regime_total_counts[regime] += 1
        if has_hit:
            regime_hit_counts[regime] += 1
            
        if regime == "slow_straight":
            # Keep all candidates for slow straight (24 cands: 22 base + 2 priors)
            selected_indices = list(range(len(cands_list)))
        else:
            # Other 3 high-variance regimes: Uniform Distance Bin Sampling (resolves data vacuum)
            best_idx = np.argmin(dists)
            prior_indices = [len(cands) - 2, len(cands) - 1]
            
            # Candidates to sample from (excluding oracle best and priors)
            remaining_indices = [i for i in range(len(cands)) if i != best_idx and i not in prior_indices]
            
            selected_indices = [best_idx] + prior_indices
            
            # Define 5 distance bins (in meters) to build a continuous spectrum
            bin_definitions = [
                (0.0, 0.005),        # 0.0cm - 0.5cm
                (0.005, 0.010),      # 0.5cm - 1.0cm
                (0.010, 0.015),      # 1.0cm - 1.5cm
                (0.015, 0.030),      # 1.5cm - 3.0cm
                (0.030, float('inf'))# > 3.0cm
            ]
            
            for low, high in bin_definitions:
                bin_mask = (dists[remaining_indices] >= low) & (dists[remaining_indices] < high)
                bin_cand_indices = np.array(remaining_indices)[bin_mask]
                
                # Sample up to 6 candidates per bin to avoid class representation imbalance
                if len(bin_cand_indices) > 6:
                    sampled = np.random.choice(bin_cand_indices, 6, replace=False)
                else:
                    sampled = bin_cand_indices
                selected_indices.extend(sampled)
                
            selected_indices = list(set(selected_indices))
            
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
                "grid_scale": c_features["grid_scale"][idx],
                "cand_speed": c_features["cand_speed"][idx],
                "cand_speed_ratio": c_features["cand_speed_ratio"][idx],
                "cand_turn_angle": c_features["cand_turn_angle"][idx],
                "cand_turn_rate": c_features["cand_turn_rate"][idx],
                "cand_accel": c_features["cand_accel"][idx],
                "cand_lat_accel": c_features["cand_lat_accel"][idx],
                **ctx,
                "dist_to_p0": np.linalg.norm(cand_pos - p0),
                "dist_to_s7": np.linalg.norm(cand_pos - s7_pos),
                "dist_to_s4": np.linalg.norm(cand_pos - s4_pos),
                "target": 1 if dist <= TARGET_THRESHOLD else 0,
                "reg_target": dist
            }
            regime_rows[regime].append(row)
            
    # Save the datasets
    print("\n--- Step 38 Candidate Generator Hit Rate (Coverage) ---")
    overall_hits = 0
    overall_total = 0
    
    # Ensure folder exists
    Path("step38_2regime_regression").mkdir(exist_ok=True)
    
    for r in regime_rows:
        df_reg = pd.DataFrame(regime_rows[r])
        save_path = Path(f"step38_2regime_regression/train_ranker_v38_{r}.csv")
        df_reg.to_csv(save_path, index=False)
        
        hit_rate = regime_hit_counts[r] / regime_total_counts[r] if regime_total_counts[r] > 0 else 0
        overall_hits += regime_hit_counts[r]
        overall_total += regime_total_counts[r]
        
        print(f"Regime {r:20}: Coverage = {hit_rate:.2%} ({regime_hit_counts[r]} / {regime_total_counts[r]}) | Saved {len(df_reg)} rows, {df_reg['target'].sum()} positives")
        
    print(f"Overall Coverage    : {overall_hits / overall_total:.2%} ({overall_hits} / {overall_total})")

if __name__ == "__main__":
    prepare_step38_data()
