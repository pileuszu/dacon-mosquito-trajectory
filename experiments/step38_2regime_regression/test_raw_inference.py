import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor
import sys
import os
import pickle

sys.path.append(os.getcwd())
from step38_2regime_regression.physics import (
    CANDIDATES_SLOW, CANDIDATES_FAST, CandidateSpec, get_damping_factors, extract_multi_scale_derivatives,
    PINV_W5_QUAD, PINV_W3_QUAD, EPS
)
from step38_2regime_regression.prepare_data import extract_context_features, CLUSTER_FEATURES

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
    "spec_ts": np.array([s.time_scale for s in CANDIDATES_SLOW] + [1.0, 1.0], dtype=np.float32),
    "spec_dmp": np.array([s.damping for s in CANDIDATES_SLOW] + [0.0, 0.0], dtype=np.float32),
    "spec_jerk": np.array([s.jerk for s in CANDIDATES_SLOW] + [0.0, 0.0], dtype=np.float32),
    "is_prior": np.array([0]*len(CANDIDATES_SLOW) + [1, 1], dtype=np.int32)
}
CANDS_FAST_SPECS = {
    "cand_idx": np.arange(len(CANDIDATES_FAST) + 2, dtype=np.int32),
    "spec_par": np.array([s.par for s in CANDIDATES_FAST] + [0.0, 0.0], dtype=np.float32),
    "spec_perp": np.array([s.perp for s in CANDIDATES_FAST] + [0.0, 0.0], dtype=np.float32),
    "spec_ts": np.array([s.time_scale for s in CANDIDATES_FAST] + [1.0, 1.0], dtype=np.float32),
    "spec_dmp": np.array([s.damping for s in CANDIDATES_FAST] + [0.0, 0.0], dtype=np.float32),
    "spec_jerk": np.array([s.jerk for s in CANDIDATES_FAST] + [0.0, 0.0], dtype=np.float32),
    "is_prior": np.array([0]*len(CANDIDATES_FAST) + [1, 1], dtype=np.int32)
}
CANDS_FAST_TURNING_SPECS = {
    "cand_idx": np.arange(len(CANDS_FAST_TURNING) + 2, dtype=np.int32),
    "spec_par": np.array([s.par for s in CANDS_FAST_TURNING] + [0.0, 0.0], dtype=np.float32),
    "spec_perp": np.array([s.perp for s in CANDS_FAST_TURNING] + [0.0, 0.0], dtype=np.float32),
    "spec_ts": np.array([s.time_scale for s in CANDS_FAST_TURNING] + [1.0, 1.0], dtype=np.float32),
    "spec_dmp": np.array([s.damping for s in CANDS_FAST_TURNING] + [0.0, 0.0], dtype=np.float32),
    "spec_jerk": np.array([s.jerk for s in CANDS_FAST_TURNING] + [0.0, 0.0], dtype=np.float32),
    "is_prior": np.array([0]*len(CANDS_FAST_TURNING) + [1, 1], dtype=np.int32)
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
    
    d_cands = all_cands - p0
    c_speeds = np.linalg.norm(d_cands, axis=-1) / 2.0
    c_speed_ratio = c_speeds / (np.linalg.norm(d1) + EPS)
    v0_norm = np.linalg.norm(d1)
    v0_hat = d1 / (v0_norm + EPS)
    d_cands_norm = np.linalg.norm(d_cands, axis=-1)
    d_cands_hat = d_cands / (d_cands_norm[:, None] + EPS)
    cos_theta = np.sum(d_cands_hat * v0_hat, axis=-1)
    c_turn_angles = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180.0 / np.pi)
    hist_turn_deg = float(ctx["smooth_turn_w5"]) * (180.0 / np.pi)
    c_turn_rates = c_turn_angles - hist_turn_deg
    c_acc = (all_cands - p0 - 2.0 * d1) / 2.0
    c_accels = np.linalg.norm(c_acc, axis=-1)
    c_acc_par = np.sum(c_acc * v0_hat, axis=-1)[:, None] * v0_hat
    c_acc_perp = c_acc - c_acc_par
    c_lat_accels = np.linalg.norm(c_acc_perp, axis=-1)
    
    c_features = {
        "grid_scale": np.full(len(all_cands), S_grid, dtype=np.float32),
        "cand_speed": c_speeds,
        "cand_speed_ratio": c_speed_ratio,
        "cand_turn_angle": c_turn_angles,
        "cand_turn_rate": c_turn_rates,
        "cand_accel": c_accels,
        "cand_lat_accel": c_lat_accels
    }
    return all_cands, spec_arr, speed, tangent, S_grid, c_features

def test_raw_vs_blend(sample_size=300):
    data_dir = Path("data/open")
    test_dir = data_dir / "test"
    s4_preds_df = pd.read_csv("experiments/step12/step4_preds_test.csv").set_index('id')
    
    models_dir = Path("experiments/step35_four_regime/models")
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "gmm_model.pkl", "rb") as f:
        gmm = pickle.load(f)
    with open(models_dir / "regime_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
        
    predictors = {}
    for regime in ["slow_straight", "fast_straight", "slow_extreme_turning", "fast_turning"]:
        model_path = Path(f"step38_2regime_regression/models/ranker_v38_{regime}")
        predictors[regime] = TabularPredictor.load(str(model_path))
        
    submission_df = pd.read_csv(data_dir / "sample_submission.csv")
    test_ids = submission_df['id'].values[:sample_size]
    
    raw_coords_list = []
    blend_coords_list = []
    
    print(f"Running comparison for {sample_size} test IDs...")
    for fid in test_ids:
        fpath = test_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        p0 = xyz[-1]
        last_vel = xyz[-1] - xyz[-2]
        s7_pos = p0 + 2.0 * last_vel
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        priors = [s7_pos, s4_pos]
        
        ctx = extract_context_features(xyz)
        feat_vector = np.array([[ctx[feat] for feat in CLUSTER_FEATURES]], dtype=np.float32)
        feat_scaled = scaler.transform(feat_vector)
        cluster_idx = gmm.predict(feat_scaled)[0]
        regime = mapping[cluster_idx]
        
        cands, spec_arr, speed, tangent, S_grid, c_features = make_candidates_vectorized(xyz, priors, regime=regime)
        
        N_c = len(cands)
        pred_dict = {
            "cand_idx": spec_arr["cand_idx"],
            "spec_par": spec_arr["spec_par"] * S_grid,
            "spec_perp": spec_arr["spec_perp"] * S_grid,
            "spec_ts": spec_arr["spec_ts"],
            "spec_dmp": spec_arr["spec_dmp"],
            "spec_jerk": spec_arr["spec_jerk"],
            "is_prior": spec_arr["is_prior"],
            "grid_scale": c_features["grid_scale"],
            "cand_speed": c_features["cand_speed"],
            "cand_speed_ratio": c_features["cand_speed_ratio"],
            "cand_turn_angle": c_features["cand_turn_angle"],
            "cand_turn_rate": c_features["cand_turn_rate"],
            "cand_accel": c_features["cand_accel"],
            "cand_lat_accel": c_features["cand_lat_accel"],
            "dist_to_p0": np.linalg.norm(cands - p0, axis=-1),
            "dist_to_s7": np.linalg.norm(cands - s7_pos, axis=-1),
            "dist_to_s4": np.linalg.norm(cands - s4_pos, axis=-1),
        }
        for k, v in ctx.items():
            pred_dict[k] = np.full(N_c, v, dtype=np.float32)
            
        pred_df = pd.DataFrame(pred_dict)
        pred_dists = predictors[regime].predict(pred_df).values
        pred_dists = np.clip(pred_dists, 0.0, None)
        
        # 1. Raw Selection
        best_idx_raw = np.argmin(pred_dists)
        raw_coords = cands[best_idx_raw]
        raw_coords_list.append(raw_coords)
        
        # 2. Blended Selection (0.005)
        probs = np.exp(- pred_dists / 0.005)
        t = tangent
        speed_m_s = speed / 0.01
        S_scale = S_grid ** 1.5
        sigma_tangential = np.clip(0.0035 + 0.005 * speed_m_s * ctx["ctx_p_saccade"], 0.003, 0.011) * S_scale
        sigma_normal = np.clip(0.0035 + 0.0015 * speed_m_s * ctx["ctx_p_saccade"], 0.003, 0.006) * S_scale
        
        active_indices = np.where(probs > 1e-5)[0]
        if len(active_indices) == 0:
            active_indices = np.array([np.argmax(probs)])
        active_probs = probs[active_indices]
        
        cands_diff = cands[:, None, :] - cands[None, active_indices, :]
        dx_tangential = np.dot(cands_diff, t)
        dx_sq = np.sum(cands_diff ** 2, axis=-1)
        dx_normal_sq = np.maximum(dx_sq - dx_tangential ** 2, 0.0)
        
        weights = np.exp(- (dx_tangential ** 2) / (2.0 * (sigma_tangential ** 2)) 
                          - dx_normal_sq / (2.0 * (sigma_normal ** 2)))
        
        smoothed_probs = weights.dot(active_probs)
        best_idx_blend = np.argmax(smoothed_probs)
        blend_coords = cands[best_idx_blend]
        blend_coords_list.append(blend_coords)
        
    raw_coords_list = np.array(raw_coords_list)
    blend_coords_list = np.array(blend_coords_list)
    
    diff_dists = np.linalg.norm(raw_coords_list - blend_coords_list, axis=1)
    print("\n=== Raw vs Blended Coordinate Differences (Test Sample) ===")
    print(f"Mean Difference: {diff_dists.mean()*100:.4f} cm")
    print(f"Median Difference: {np.median(diff_dists)*100:.4f} cm")
    print(f"Max Difference: {diff_dists.max()*100:.4f} cm")
    print(f"Fraction > 0.5cm: {(diff_dists > 0.005).mean():.2%}")
    print(f"Fraction > 1.0cm: {(diff_dists > 0.01).mean():.2%}")

if __name__ == "__main__":
    test_raw_vs_blend()
