import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
import sys
import pickle
import traceback
import torch
torch.set_num_threads(4)

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification
from step45_hybrid_ranker.physics import (
    CANDIDATES_SLOW, CANDIDATES_FAST_2D, CANDIDATES_FAST_3D, CANDIDATES_RAW_ACCEL_TURNING, CandidateSpec, get_damping_factors, extract_multi_scale_derivatives,
    PINV_W5_QUAD, PINV_W3_QUAD, EPS
)
from step45_hybrid_ranker.prepare_data import extract_context_features, CLUSTER_FEATURES, REGIME_MAPPING

# Precompute multiplier matrices for vectorized candidate generation
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
        col4 = spec.binormal * (acc_scale * ts2 * fa)
        col5 = spec.jerk * (jerk_scale * ts3 * fj)
        M.append([col0, col1, col2, col3, col4, col5])
        
    return np.array(M, dtype=np.float32)

fallback_specs = [
    CandidateSpec(name="straight_fallback_d0.2", d1=1.0, par=0.0, perp=0.0, binormal=0.0, damping=0.2),
    CandidateSpec(name="straight_fallback_d0.5", d1=1.0, par=0.0, perp=0.0, binormal=0.0, damping=0.5),
    CandidateSpec(name="straight_fallback_d0.8", d1=1.0, par=0.0, perp=0.0, binormal=0.0, damping=0.8),
]
CANDS_FAST_TURNING_3D = CANDIDATES_FAST_3D + fallback_specs

M_SLOW = compute_multiplier_matrix(CANDIDATES_SLOW)
M_FAST_2D = compute_multiplier_matrix(CANDIDATES_FAST_2D)
M_FAST_3D = compute_multiplier_matrix(CANDIDATES_FAST_3D)
M_FAST_TURNING_3D = compute_multiplier_matrix(CANDS_FAST_TURNING_3D)
M_RAW_ACCEL = compute_multiplier_matrix(CANDIDATES_RAW_ACCEL_TURNING)

def build_spec_dict(specs_list):
    return {
        "cand_idx": np.arange(len(specs_list) + 2, dtype=np.int32),
        "spec_par": np.array([s.par for s in specs_list] + [0.0, 0.0], dtype=np.float32),
        "spec_perp": np.array([s.perp for s in specs_list] + [0.0, 0.0], dtype=np.float32),
        "spec_binormal": np.array([s.binormal for s in specs_list] + [0.0, 0.0], dtype=np.float32),
        "spec_ts": np.array([s.time_scale for s in specs_list] + [1.0, 1.0], dtype=np.float32),
        "spec_dmp": np.array([s.damping for s in specs_list] + [0.0, 0.0], dtype=np.float32),
        "spec_jerk": np.array([s.jerk for s in specs_list] + [0.0, 0.0], dtype=np.float32),
        "is_prior": np.array([0]*len(specs_list) + [1, 1], dtype=np.int32)
    }

CANDS_SLOW_SPECS = build_spec_dict(CANDIDATES_SLOW)
CANDS_FAST_2D_SPECS = build_spec_dict(CANDIDATES_FAST_2D)
CANDS_FAST_3D_SPECS = build_spec_dict(CANDIDATES_FAST_3D)
CANDS_FAST_TURNING_3D_SPECS = build_spec_dict(CANDS_FAST_TURNING_3D)
CANDS_FAST_EXTREME_3D_SPECS = build_spec_dict(CANDS_FAST_TURNING_3D + CANDIDATES_RAW_ACCEL_TURNING)
CANDS_SLOW_EXTREME_2D_SPECS = build_spec_dict(CANDIDATES_FAST_2D + CANDIDATES_RAW_ACCEL_TURNING)

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
    
    if regime == "fast_straight_low":
        acc_smooth = acc
        if len(x_sliced) >= 5:
            x_w5 = x_sliced[-5:]
            coeffs_w5 = PINV_W5_QUAD @ x_w5
            acc_smooth = 2.0 * coeffs_w5[0]
            
        M = M_FAST_2D.copy()
        spec_arr = CANDS_FAST_2D_SPECS
        S_grid = float(np.clip(1.0 + 0.4 * p_sacc, 1.0, 1.8))
        
    elif regime == "slow_moderate_turning":
        acc_smooth = acc
        if len(x_sliced) >= 5:
            x_w5 = x_sliced[-5:]
            coeffs_w5 = PINV_W5_QUAD @ x_w5
            acc_smooth = 2.0 * coeffs_w5[0]
            
        M = M_SLOW.copy()
        spec_arr = CANDS_SLOW_SPECS
        S_grid = float(np.clip(1.0 + 0.1 * ctx_curv, 1.0, 1.5))
        
    elif regime == "fast_moderate_turning":
        acc_smooth = acc
        if len(x_sliced) >= 5:
            x_w5 = x_sliced[-5:]
            coeffs_w5 = PINV_W5_QUAD @ x_w5
            acc_smooth = 2.0 * coeffs_w5[0]
            
        M = M_FAST_2D.copy()
        spec_arr = CANDS_FAST_2D_SPECS
        S_grid = float(np.clip(1.2 + 0.1 * ctx_curv, 1.2, 2.0))
        
    elif regime == "fast_straight_high":
        acc_smooth = acc
        if len(x_sliced) >= 5:
            x_w5 = x_sliced[-5:]
            coeffs_w5 = PINV_W5_QUAD @ x_w5
            acc_smooth = 2.0 * coeffs_w5[0]
            
        M = M_FAST_2D.copy()
        spec_arr = CANDS_FAST_2D_SPECS
        S_grid = float(np.clip(1.0 + 0.6 * p_sacc, 1.0, 3.2))
        
    elif regime == "fast_extreme_turning":
        acc_smooth = acc
        if len(x_sliced) >= 3:
            x_w3 = x_sliced[-3:]
            coeffs_w3 = PINV_W3_QUAD @ x_w3
            acc_smooth = 2.0 * coeffs_w3[0]
            
        M = M_FAST_TURNING_3D.copy()
        spec_arr = CANDS_FAST_EXTREME_3D_SPECS
        S_grid = float(np.clip(1.8 + 0.8 * p_sacc, 1.8, 4.2))
        
    else: # slow_extreme_turning
        acc_smooth = acc
        if len(x_sliced) >= 3:
            x_w3 = x_sliced[-3:]
            coeffs_w3 = PINV_W3_QUAD @ x_w3
            acc_smooth = 2.0 * coeffs_w3[0]
            
        M = M_FAST_2D.copy()
        spec_arr = CANDS_SLOW_EXTREME_2D_SPECS
        S_grid = float(np.clip(1.8 + 0.15 * ctx["smooth_curv_w3"], 1.8, 4.0))
        
    acc_par_scalar = np.sum(acc_smooth * tangent)
    acc_par = acc_par_scalar * tangent
    acc_perp_vec = acc_smooth - acc_par
    
    acc_perp_magnitude = np.linalg.norm(acc_perp_vec)
    if acc_perp_magnitude > 1e-6:
        n_hat = acc_perp_vec / acc_perp_magnitude
    else:
        if abs(tangent[0]) > 0.9:
            n_hat = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            n_hat = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        n_hat = n_hat - np.sum(n_hat * tangent) * tangent
        n_hat = n_hat / (np.linalg.norm(n_hat) + EPS)
        
    b_hat = np.cross(tangent, n_hat)
    b_hat = b_hat / (np.linalg.norm(b_hat) + EPS)
    
    acc_binormal_vec = acc_perp_magnitude * b_hat
    
    D = np.vstack([d1, d2, acc_par, acc_perp_vec, acc_binormal_vec, jerk])
    
    M[:, 2] *= S_grid
    M[:, 3] *= S_grid
    M[:, 4] *= S_grid
    
    preds = p0 + M @ D
    
    if regime in ["fast_extreme_turning", "slow_extreme_turning"]:
        acc_par_scalar_raw = np.sum(acc * tangent)
        acc_par_raw = acc_par_scalar_raw * tangent
        acc_perp_vec_raw = acc - acc_par_raw
        
        D_raw = np.vstack([d1, d2, acc_par_raw, acc_perp_vec_raw, np.zeros_like(acc_perp_vec_raw), jerk])
        
        M_raw = M_RAW_ACCEL.copy()
        M_raw[:, 2] *= S_grid
        M_raw[:, 3] *= S_grid
        M_raw[:, 4] *= S_grid
        
        preds_raw = p0 + M_raw @ D_raw
        preds = np.vstack([preds, preds_raw])
        
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
    
    c_displacement_normal = np.sum(d_cands * n_hat, axis=-1)
    c_displacement_binormal = np.sum(d_cands * b_hat, axis=-1)
    c_accel_binormal = np.sum(c_acc * b_hat, axis=-1)
    
    c_features = {
        "grid_scale": np.full(len(all_cands), S_grid, dtype=np.float32),
        "cand_speed": c_speeds,
        "cand_speed_ratio": c_speed_ratio,
        "cand_turn_angle": c_turn_angles,
        "cand_turn_rate": c_turn_rates,
        "cand_accel": c_accels,
        "cand_lat_accel": c_lat_accels,
        "cand_disp_normal": c_displacement_normal,
        "cand_disp_binormal": c_displacement_binormal,
        "cand_acc_binormal": c_accel_binormal
    }
    
    return all_cands, spec_arr, speed, tangent, S_grid, ctx_curv, c_features

def run_step45_inference(batch_size=250, blend_priors=True):
    try:
        msg = f"🚀 Starting: [Step 45 inference.py] Batched test inference (blend_priors={blend_priors})..."
        send_discord_notification(None, msg)
        print(msg)
        
        data_dir = Path("data/open")
        test_dir = data_dir / "test"
        
        print("Loading EqMotion test predictions...")
        s4_preds_df = pd.read_csv("experiments/step12/step4_preds_test.csv").set_index('id')
        
        print("Loading Scaler, GMM-6 config...")
        gmm_models_dir = Path("step39_six_regime/models")
        with open(gmm_models_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(gmm_models_dir / "gmm_model.pkl", "rb") as f:
            gmm = pickle.load(f)
            
        print("Loading BGM-12 components...")
        bgm_models_dir = Path("step40_dual_specialized/models/bgm")
        with open(bgm_models_dir / "scaler.pkl", "rb") as f:
            bgm_scaler = pickle.load(f)
        with open(bgm_models_dir / "bgm_model.pkl", "rb") as f:
            bgm = pickle.load(f)
            
        print("Loading 4 Specialized AutoGluon Predictors...")
        models_dir = Path("step45_hybrid_ranker/models")
        pred_fsl = TabularPredictor.load(str(models_dir / "final_fast_straight_low"))
        pred_smt = TabularPredictor.load(str(models_dir / "final_slow_moderate_turning"))
        pred_fsh = TabularPredictor.load(str(models_dir / "final_fast_straight_high"))
        pred_steer = TabularPredictor.load(str(models_dir / "final_steering"))
        
        submission_df = pd.read_csv(data_dir / "sample_submission.csv")
        test_ids = submission_df['id'].values
        
        predictions = []
        displacements = []
        
        print(f"Running batched inference with GMM-6 Hybrid Classifier-Regressor Routing...")
        
        for i in range(0, len(test_ids), batch_size):
            batch_ids = test_ids[i : i + batch_size]
            
            batch_cand_idx = []
            batch_regime = []
            batch_spec_par = []
            batch_spec_perp = []
            batch_spec_binormal = []
            batch_spec_ts = []
            batch_spec_dmp = []
            batch_spec_jerk = []
            batch_is_prior = []
            
            batch_gmm_probs = [[] for _ in range(6)]
            batch_gmm_cluster = []
            batch_bgm_probs = [[] for _ in range(12)]
            batch_bgm_cluster = []
            
            batch_grid_scale = []
            batch_cand_speed = []
            batch_cand_speed_ratio = []
            batch_cand_turn_angle = []
            batch_cand_turn_rate = []
            batch_cand_accel = []
            batch_cand_lat_accel = []
            batch_cand_disp_normal = []
            batch_cand_disp_binormal = []
            batch_cand_acc_binormal = []
            
            batch_ctx = None
            
            batch_dist_p0 = []
            batch_dist_s7 = []
            batch_dist_s4 = []
            
            batch_info = []
            
            start_idx = 0
            for fid in batch_ids:
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
                
                probs_6 = gmm.predict_proba(feat_scaled)[0]
                cluster_idx = np.argmax(probs_6)
                regime_name = REGIME_MAPPING[cluster_idx]
                
                bgm_feat_scaled = bgm_scaler.transform(feat_vector)
                probs_bgm = bgm.predict_proba(bgm_feat_scaled)[0]
                cluster_idx_bgm = bgm.predict(bgm_feat_scaled)[0]
                
                cands, spec_arr, speed, tangent, S_grid, ctx_curv, c_features = make_candidates_vectorized(
                    xyz, priors=priors, end_idx=-1, horizon=2, regime=regime_name
                )
                
                N_c = len(cands)
                
                batch_cand_idx.append(spec_arr["cand_idx"])
                batch_regime.append([regime_name] * N_c)
                batch_spec_par.append(spec_arr["spec_par"])
                batch_spec_perp.append(spec_arr["spec_perp"])
                batch_spec_binormal.append(spec_arr["spec_binormal"])
                batch_spec_ts.append(spec_arr["spec_ts"])
                batch_spec_dmp.append(spec_arr["spec_dmp"])
                batch_spec_jerk.append(spec_arr["spec_jerk"])
                batch_is_prior.append(spec_arr["is_prior"])
                
                batch_grid_scale.append(c_features["grid_scale"])
                batch_cand_speed.append(c_features["cand_speed"])
                batch_cand_speed_ratio.append(c_features["cand_speed_ratio"])
                batch_cand_turn_angle.append(c_features["cand_turn_angle"])
                batch_cand_turn_rate.append(c_features["cand_turn_rate"])
                batch_cand_accel.append(c_features["cand_accel"])
                batch_cand_lat_accel.append(c_features["cand_lat_accel"])
                batch_cand_disp_normal.append(c_features["cand_disp_normal"])
                batch_cand_disp_binormal.append(c_features["cand_disp_binormal"])
                batch_cand_acc_binormal.append(c_features["cand_acc_binormal"])
                
                for j in range(6):
                    batch_gmm_probs[j].append(np.full(N_c, probs_6[j], dtype=np.float32))
                batch_gmm_cluster.append(np.full(N_c, cluster_idx, dtype=np.int32))
                
                for k in range(12):
                    batch_bgm_probs[k].append(np.full(N_c, probs_bgm[k], dtype=np.float32))
                batch_bgm_cluster.append(np.full(N_c, cluster_idx_bgm, dtype=np.int32))
                
                if batch_ctx is None:
                    batch_ctx = {k: [] for k in ctx.keys()}
                for k, v in ctx.items():
                    batch_ctx[k].append(np.full(N_c, v, dtype=np.float32))
                    
                d_p0 = np.linalg.norm(cands - p0, axis=-1)
                d_s7 = np.linalg.norm(cands - s7_pos, axis=-1)
                d_s4 = np.linalg.norm(cands - s4_pos, axis=-1)
                
                batch_dist_p0.append(d_p0)
                batch_dist_s7.append(d_s7)
                batch_dist_s4.append(d_s4)
                
                end_idx = start_idx + N_c
                batch_info.append({
                    "id": fid,
                    "start": start_idx,
                    "end": end_idx,
                    "cands": cands,
                    "p0": p0,
                    "s7_pos": s7_pos,
                    "s4_pos": s4_pos,
                    "regime": regime_name,
                    "p_0": float(probs_6[0]),
                    "p_1": float(probs_6[1]),
                    "p_2": float(probs_6[2]),
                    "p_3": float(probs_6[3]),
                    "p_4": float(probs_6[4]),
                    "p_5": float(probs_6[5]),
                    "speed": speed,
                    "tangent": tangent,
                    "p_saccade": ctx["ctx_p_saccade"],
                    "S_grid": S_grid,
                    "ctx_curv": ctx_curv
                })
                start_idx = end_idx
                
            pred_dict = {
                "cand_idx": np.concatenate(batch_cand_idx),
                "regime": np.concatenate(batch_regime),
                "spec_par": np.concatenate(batch_spec_par),
                "spec_perp": np.concatenate(batch_spec_perp),
                "spec_binormal": np.concatenate(batch_spec_binormal),
                "spec_ts": np.concatenate(batch_spec_ts),
                "spec_dmp": np.concatenate(batch_spec_dmp),
                "spec_jerk": np.concatenate(batch_spec_jerk),
                "is_prior": np.concatenate(batch_is_prior),
                
                "grid_scale": np.concatenate(batch_grid_scale),
                "cand_speed": np.concatenate(batch_cand_speed),
                "cand_speed_ratio": np.concatenate(batch_cand_speed_ratio),
                "cand_turn_angle": np.concatenate(batch_cand_turn_angle),
                "cand_turn_rate": np.concatenate(batch_cand_turn_rate),
                "cand_accel": np.concatenate(batch_cand_accel),
                "cand_lat_accel": np.concatenate(batch_cand_lat_accel),
                "cand_disp_normal": np.concatenate(batch_cand_disp_normal),
                "cand_disp_binormal": np.concatenate(batch_cand_disp_binormal),
                "cand_acc_binormal": np.concatenate(batch_cand_acc_binormal),
                
                "dist_to_p0": np.concatenate(batch_dist_p0),
                "dist_to_s7": np.concatenate(batch_dist_s7),
                "dist_to_s4": np.concatenate(batch_dist_s4)
            }
            for k in batch_ctx.keys():
                pred_dict[k] = np.concatenate(batch_ctx[k])
                
            for j in range(6):
                pred_dict[f"gmm_p{j}"] = np.concatenate(batch_gmm_probs[j])
            pred_dict["gmm_cluster"] = np.concatenate(batch_gmm_cluster)
            
            for k in range(12):
                pred_dict[f"bgm_p{k}"] = np.concatenate(batch_bgm_probs[k])
            pred_dict["bgm_cluster"] = np.concatenate(batch_bgm_cluster)
            
            pred_data = pd.DataFrame(pred_dict)
            
            pred_data['norm_cand_speed'] = pred_data['cand_speed'] / (pred_data['ctx_speed'] + 1e-6)
            pred_data['norm_cand_accel'] = pred_data['cand_accel'] / (pred_data['ctx_acc'] + 1e-6)
            pred_data['s7_s4_dist'] = np.abs(pred_data['dist_to_s7'] - pred_data['dist_to_s4'])
            
            pred_data['norm_cand_disp_binormal'] = pred_data['cand_disp_binormal'] / (pred_data['ctx_speed'] + 1e-6)
            pred_data['norm_cand_acc_binormal'] = pred_data['cand_acc_binormal'] / (pred_data['ctx_acc'] + 1e-6)
            
            # Predict for the 4 models
            scores_0 = np.zeros(len(pred_data), dtype=np.float32)
            scores_1 = np.zeros(len(pred_data), dtype=np.float32)
            scores_3 = np.zeros(len(pred_data), dtype=np.float32)
            scores_steer = np.zeros(len(pred_data), dtype=np.float32)
            
            mask_0 = np.zeros(len(pred_data), dtype=bool)
            mask_1 = np.zeros(len(pred_data), dtype=bool)
            mask_3 = np.zeros(len(pred_data), dtype=bool)
            mask_steer = np.zeros(len(pred_data), dtype=bool)
            
            for info in batch_info:
                if info['p_0'] > 0.001:
                    mask_0[info['start'] : info['end']] = True
                if info['p_1'] > 0.001:
                    mask_1[info['start'] : info['end']] = True
                if info['p_3'] > 0.001:
                    mask_3[info['start'] : info['end']] = True
                if (info['p_2'] + info['p_4'] + info['p_5']) > 0.001:
                    mask_steer[info['start'] : info['end']] = True
                    
            if np.any(mask_0):
                scores_0[mask_0] = pred_fsl.predict_proba(pred_data[mask_0])[1].values
            if np.any(mask_1):
                scores_1[mask_1] = pred_smt.predict_proba(pred_data[mask_1])[1].values
            if np.any(mask_3):
                scores_3[mask_3] = pred_fsh.predict_proba(pred_data[mask_3])[1].values
            if np.any(mask_steer):
                dists_steer = pred_steer.predict(pred_data[mask_steer]).values
                scores_steer[mask_steer] = np.exp(-dists_steer / 0.015)
            
            # Combine scores and run anisotropic smoothing
            for info in batch_info:
                sub_s0 = scores_0[info['start'] : info['end']]
                sub_s1 = scores_1[info['start'] : info['end']]
                sub_s3 = scores_3[info['start'] : info['end']]
                sub_s_steer = scores_steer[info['start'] : info['end']]
                
                probs = (
                    info['p_0'] * sub_s0 +
                    info['p_1'] * sub_s1 +
                    info['p_2'] * sub_s_steer +
                    info['p_3'] * sub_s3 +
                    info['p_4'] * sub_s_steer +
                    info['p_5'] * sub_s_steer
                )
                
                cands = info['cands']
                t = info['tangent']
                p_sacc = info['p_saccade']
                speed = info['speed']
                speed_m_s = speed / 0.01
                
                S_grid = info['S_grid']
                S_scale = S_grid ** 1.5
                
                sigma_tangential = np.clip(0.0035 + 0.005 * speed_m_s * p_sacc, 0.003, 0.011) * S_scale
                sigma_normal = np.clip(0.0035 + 0.0015 * speed_m_s * p_sacc, 0.003, 0.006) * S_scale
                
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
                best_idx = np.argmax(smoothed_probs)
                final_coords = cands[best_idx]
                
                # Apply optimal coordinate blending weights
                if blend_priors:
                    # These weights are defaults; optimize_blending will output actual optimal weights
                    optimal_weights = {
                        "fast_straight_low": (0.60, 0.40, 0.00),
                        "slow_moderate_turning": (0.80, 0.10, 0.10),
                        "fast_moderate_turning": (0.65, 0.00, 0.35),
                        "fast_straight_high": (0.95, 0.00, 0.05),
                        "fast_extreme_turning": (1.00, 0.00, 0.00),
                        "slow_extreme_turning": (0.95, 0.00, 0.05)
                    }
                    w_m, w_4, w_7 = optimal_weights[info['regime']]
                    final_coords = w_m * final_coords + w_4 * info['s4_pos'] + w_7 * info['s7_pos']
                    
                predictions.append({
                    "id": info['id'],
                    "x": final_coords[0],
                    "y": final_coords[1],
                    "z": final_coords[2]
                })
                
                disp = np.linalg.norm(final_coords - info['p0'])
                displacements.append(disp)
                
            print(f"Processed batch {i//batch_size + 1}/{(len(test_ids)-1)//batch_size + 1}...")
            
        out_df = pd.DataFrame(predictions)
        suffix = "_blended" if blend_priors else ""
        out_path = Path(f"outputs/step45_hybrid_ranker/submission{suffix}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Step 45 submission saved to {out_path}")
        
        displacements = np.array(displacements)
        mean_disp_cm = displacements.mean() * 100
        max_disp_cm = displacements.max() * 100
        
        success_msg = (
            f"✅ Finished: [Step 45 inference.py] Submission created successfully!\n"
            f"Saved to: `{out_path}`\n"
            f"Physical Displacement from p0:\n"
            f"- **Mean**: **{mean_disp_cm:.4f} cm**\n"
            f"- **Max**: **{max_disp_cm:.4f} cm**"
        )
        send_discord_notification(None, success_msg)
        print(success_msg)
        
    except BaseException as e:
        error_msg = f"❌ Failed: [Step 45 inference.py] Inference ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(None, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 45 Hybrid Inference")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size of test cases to process (default: 250)")
    parser.add_argument("--no_blend", action="store_true", help="Disable coordinate blending")
    args = parser.parse_args()
    
    run_step45_inference(batch_size=args.batch_size, blend_priors=not args.no_blend)
