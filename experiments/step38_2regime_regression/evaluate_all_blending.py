import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor
import sys
import os

sys.path.append(os.getcwd())
from step38_2regime_regression.physics import (
    CANDIDATES_SLOW, CANDIDATES_FAST, CandidateSpec, get_damping_factors, extract_multi_scale_derivatives,
    PINV_W5_QUAD, PINV_W3_QUAD, EPS
)

def compute_multiplier_matrix(cands_list, horizon=2):
    v_scale = horizon
    acc_scale = 0.5 * (horizon**2)
    jerk_scale = (1.0/6.0) * (horizon**3)
    dmps = np.array([s.damping for s in cands_list], dtype=np.float32)
    F_v, F_a, F_j = get_damping_factors(dmps)
    M = []
    for i, spec in enumerate(cands_list):
        ts = spec.time_scale
        ts2 = ts ** 2
        ts3 = ts ** 3
        fv = F_v[i]
        fa = F_a[i]
        fj = F_j[i]
        col0 = spec.d1 * (v_scale * ts * fv)
        col1 = spec.d2 * (v_scale * ts * fv)
        col2 = spec.par * (acc_scale * ts2 * fa)
        col3 = spec.perp * (acc_scale * ts2 * fa)
        col4 = spec.jerk * (jerk_scale * ts3 * fj)
        M.append([col0, col1, col2, col3, col4])
    return np.array(M, dtype=np.float32)

fallback_specs = [
    CandidateSpec(name="straight_fallback_d0.2", d1=1.0, par=0.0, perp=0.0, damping=0.2),
    CandidateSpec(name="straight_fallback_d0.5", d1=1.0, par=0.0, perp=0.0, damping=0.5),
    CandidateSpec(name="straight_fallback_d0.8", d1=1.0, par=0.0, perp=0.0, damping=0.8),
]
CANDS_FAST_TURNING = CANDIDATES_FAST + fallback_specs

M_SLOW_STRAIGHT = compute_multiplier_matrix(CANDIDATES_SLOW)
M_FAST_STRAIGHT = compute_multiplier_matrix(CANDIDATES_FAST)
M_SLOW_EXTREME_TURNING = compute_multiplier_matrix(CANDIDATES_FAST)
M_FAST_TURNING = compute_multiplier_matrix(CANDS_FAST_TURNING)

CANDS_SLOW_SPECS = {
    "cand_idx": np.arange(len(CANDIDATES_SLOW) + 2, dtype=np.int32),
    "spec_par": np.array([s.par for s in CANDIDATES_SLOW] + [0.0, 0.0], dtype=np.float32),
    "spec_perp": np.array([s.perp for s in CANDIDATES_SLOW] + [0.0, 0.0], dtype=np.float32),
}
CANDS_FAST_SPECS = {
    "cand_idx": np.arange(len(CANDIDATES_FAST) + 2, dtype=np.int32),
    "spec_par": np.array([s.par for s in CANDIDATES_FAST] + [0.0, 0.0], dtype=np.float32),
    "spec_perp": np.array([s.perp for s in CANDIDATES_FAST] + [0.0, 0.0], dtype=np.float32),
}
CANDS_FAST_TURNING_SPECS = {
    "cand_idx": np.arange(len(CANDS_FAST_TURNING) + 2, dtype=np.int32),
    "spec_par": np.array([s.par for s in CANDS_FAST_TURNING] + [0.0, 0.0], dtype=np.float32),
    "spec_perp": np.array([s.perp for s in CANDS_FAST_TURNING] + [0.0, 0.0], dtype=np.float32),
}

def make_candidates_vectorized(x, priors, end_idx=-1, horizon=2, regime=None):
    x_sliced = x[:end_idx+1] if end_idx != -1 else x
    p0 = x_sliced[-1]
    d1 = x_sliced[-1] - x_sliced[-2]
    d2 = x_sliced[-2] - x_sliced[-3]
    d3 = x_sliced[-3] - x_sliced[-4]
    acc = d1 - d2
    prev_acc = d2 - d3
    jerk = acc - prev_acc
    speed = np.linalg.norm(d1)
    tangent = d1 / (speed + EPS)
    
    ctx = extract_multi_scale_derivatives(x_sliced)
    p_sacc = ctx["ctx_p_saccade"]
    ctx_lat_accel = ctx["ctx_lat_accel"]
    ctx_curv = ctx["smooth_curv_w5"]
    
    if regime == "slow_straight":
        acc_smooth = acc
        if len(x_sliced) >= 5:
            x_w5 = x_sliced[-5:]
            coeffs_w5 = PINV_W5_QUAD @ x_w5
            acc_smooth = 2.0 * coeffs_w5[0]
        M = M_SLOW_STRAIGHT.copy()
        spec_arr = CANDS_SLOW_SPECS
        S_grid = float(np.clip(1.0 + 0.15 * ctx_curv, 1.0, 1.8))
    elif regime == "fast_straight":
        acc_smooth = acc
        if len(x_sliced) >= 5:
            x_w5 = x_sliced[-5:]
            coeffs_w5 = PINV_W5_QUAD @ x_w5
            acc_smooth = 2.0 * coeffs_w5[0]
        M = M_FAST_STRAIGHT.copy()
        spec_arr = CANDS_FAST_SPECS
        S_grid = float(np.clip(1.0 + 0.6 * p_sacc, 1.0, 2.5))
    elif regime == "slow_extreme_turning":
        x_w3 = x_sliced[-3:]
        coeffs_w3 = PINV_W3_QUAD @ x_w3
        acc_smooth = 2.0 * coeffs_w3[0]
        speed_w3 = np.linalg.norm(coeffs_w3[1])
        tangent = coeffs_w3[1] / (speed_w3 + EPS)
        M = M_SLOW_EXTREME_TURNING.copy()
        spec_arr = CANDS_FAST_SPECS
        S_grid = float(np.clip(1.5 + 0.1 * ctx["smooth_curv_w3"], 1.5, 3.5))
    elif regime == "fast_turning":
        x_w3 = x_sliced[-3:]
        coeffs_w3 = PINV_W3_QUAD @ x_w3
        acc_smooth = 2.0 * coeffs_w3[0]
        speed_w3 = np.linalg.norm(coeffs_w3[1])
        tangent = coeffs_w3[1] / (speed_w3 + EPS)
        M = M_FAST_TURNING.copy()
        spec_arr = CANDS_FAST_TURNING_SPECS
        S_grid = float(np.clip(1.2 + 0.6 * p_sacc, 1.2, 3.0))
        
    acc_par_scalar = np.sum(acc_smooth * tangent)
    acc_par = acc_par_scalar * tangent
    acc_perp_vec = acc_smooth - acc_par
    D = np.vstack([d1, d2, acc_par, acc_perp_vec, jerk])
    M[:, 2] *= S_grid
    M[:, 3] *= S_grid
    preds = p0 + M @ D
    s7_pos, s4_pos = priors
    all_cands = np.vstack([preds, s7_pos, s4_pos])
    return all_cands, tangent, S_grid, p_sacc

def test_regime(regime_name, sample_size=500):
    print(f"\n--- Testing Regime: {regime_name.upper()} ---")
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv").set_index('id')
    s4_preds_df = pd.read_csv("experiments/step18_oof/step4_oof_train.csv").set_index('id')
    
    train_data_path = f"step38_2regime_regression/train_ranker_v38_{regime_name}.csv"
    df = pd.read_csv(train_data_path)
    
    model_path = f"step38_2regime_regression/models/ranker_v38_{regime_name}"
    predictor = TabularPredictor.load(model_path)
    
    print("Loading OOF predictions...")
    oof_pred = predictor.predict_oof()
    df['oof_pred_dist'] = oof_pred.values
    
    # 1. Baseline: Raw Minimum selection (no blending)
    idx_best = df.groupby('id')['oof_pred_dist'].idxmin()
    best_cands = df.loc[idx_best]
    hit_rate_raw_all = (best_cands['reg_target'] <= 0.01).mean()
    print(f"  All-data Raw minimum selection Hit@1.0cm: {hit_rate_raw_all:.4%}")
    
    unique_ids = df['id'].unique()
    np.random.seed(42)
    sample_ids = np.random.choice(unique_ids, min(sample_size, len(unique_ids)), replace=False)
    
    results = {
        "raw": [],
        "blend_0.015": [],
        "blend_0.005": [],
        "blend_0.002": [],
        "blend_0.0005": []
    }
    
    for fid in sample_ids:
        # Load trajectory
        fpath = train_dir / f"{fid}.csv"
        traj_df = pd.read_csv(fpath)
        xyz = traj_df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)
        
        # Calculate Priors
        p0 = xyz[-1]
        last_vel = xyz[-1] - xyz[-2]
        s7_pos = p0 + 2.0 * last_vel
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        priors = [s7_pos, s4_pos]
        
        # Generate candidate coordinates
        cands, tangent, S_grid, p_sacc = make_candidates_vectorized(xyz, priors, regime=regime_name)
        
        # OOF Predictions for this ID
        id_df = df[df['id'] == fid].sort_values('cand_idx')
        pred_dists = id_df['oof_pred_dist'].values
        cand_indices = id_df['cand_idx'].values
        
        pred_map = {idx: dist for idx, dist in zip(cand_indices, pred_dists)}
        full_pred_dists = np.array([pred_map.get(i, 10.0) for i in range(len(cands))], dtype=np.float32)
        full_pred_dists = np.clip(full_pred_dists, 0.0, None)
        
        # Raw Selection
        best_idx_raw = np.argmin(full_pred_dists)
        err_raw = np.linalg.norm(cands[best_idx_raw] - target)
        results["raw"].append(1 if err_raw <= 0.01 else 0)
        
        # Blending setup
        speed = np.linalg.norm(last_vel)
        speed_m_s = speed / 0.01
        S_scale = S_grid ** 1.5
        
        sigma_tangential = np.clip(0.0035 + 0.005 * speed_m_s * p_sacc, 0.003, 0.011) * S_scale
        sigma_normal = np.clip(0.0035 + 0.0015 * speed_m_s * p_sacc, 0.003, 0.006) * S_scale
        
        # Test different sigma_dists
        for sd in [0.015, 0.005, 0.002, 0.0005]:
            probs = np.exp(- full_pred_dists / sd)
            
            active_indices = np.where(probs > 1e-5)[0]
            if len(active_indices) == 0:
                active_indices = np.array([np.argmax(probs)])
            active_probs = probs[active_indices]
            
            cands_diff = cands[:, None, :] - cands[None, active_indices, :]
            dx_tangential = np.dot(cands_diff, tangent)
            dx_sq = np.sum(cands_diff ** 2, axis=-1)
            dx_normal_sq = np.maximum(dx_sq - dx_tangential ** 2, 0.0)
            
            weights = np.exp(- (dx_tangential ** 2) / (2.0 * (sigma_tangential ** 2)) 
                              - dx_normal_sq / (2.0 * (sigma_normal ** 2)))
            
            smoothed_probs = weights.dot(active_probs)
            best_idx_blend = np.argmax(smoothed_probs)
            
            err_blend = np.linalg.norm(cands[best_idx_blend] - target)
            results[f"blend_{sd}"].append(1 if err_blend <= 0.01 else 0)
            
    summary = {k: np.mean(v) for k, v in results.items()}
    print(f"Sample of {len(sample_ids)} trajectories:")
    for k, v in summary.items():
        print(f"  {k:15}: {v:.2%}")
    return summary

def main():
    regimes = ["slow_straight", "fast_straight", "slow_extreme_turning", "fast_turning"]
    weights = {
        "slow_straight": 0.3315,
        "fast_straight": 0.3697,
        "slow_extreme_turning": 0.0854,
        "fast_turning": 0.2134
    }
    
    all_summaries = {}
    for r in regimes:
        all_summaries[r] = test_regime(r, sample_size=500)
        
    print("\n================ OVERALL ESTIMATED OOF PERFORMANCE ================")
    for key in ["raw", "blend_0.015", "blend_0.005", "blend_0.002", "blend_0.0005"]:
        weighted_score = sum(all_summaries[r][key] * weights[r] for r in regimes)
        print(f"Overall {key:12}: {weighted_score:.4%}")

if __name__ == "__main__":
    main()
