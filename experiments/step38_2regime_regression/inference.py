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
from step38_2regime_regression.physics import (
    CANDIDATES_SLOW, CANDIDATES_FAST, CandidateSpec, get_damping_factors, extract_multi_scale_derivatives,
    PINV_W5_QUAD, PINV_W3_QUAD, EPS
)
from step38_2regime_regression.prepare_data import extract_context_features, CLUSTER_FEATURES
from utils.notifier import send_discord_notification

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
    
    is_turning = (ctx_curv > 12.0) or (ctx_lat_accel > 0.0020)
    
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
        
    else:
        # Fallback to physical threshold
        if speed <= 0.0234 and not is_turning:
            acc_smooth = acc
            if len(x_sliced) >= 5:
                x_w5 = x_sliced[-5:]
                coeffs_w5 = PINV_W5_QUAD @ x_w5
                acc_smooth = 2.0 * coeffs_w5[0]
            M = M_SLOW_STRAIGHT.copy()
            spec_arr = CANDS_SLOW_SPECS
            S_grid = 1.0
        else:
            acc_smooth = acc
            if len(x_sliced) >= 5:
                x_w5 = x_sliced[-5:]
                coeffs_w5 = PINV_W5_QUAD @ x_w5
                acc_smooth = 2.0 * coeffs_w5[0]
            M = M_FAST_STRAIGHT.copy()
            spec_arr = CANDS_FAST_SPECS
            S_grid = float(np.clip(1.0 + 0.6 * p_sacc, 1.0, 2.5))
            
    acc_par_scalar = np.sum(acc_smooth * tangent)
    acc_par = acc_par_scalar * tangent
    acc_perp_vec = acc_smooth - acc_par
    
    D = np.vstack([d1, d2, acc_par, acc_perp_vec, jerk])
    
    M[:, 2] *= S_grid
    M[:, 3] *= S_grid
    
    preds = p0 + M @ D
    
    s7_pos, s4_pos = priors
    all_cands = np.vstack([preds, s7_pos, s4_pos])
    
    # Compute candidate-level physical features
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
    
    return all_cands, spec_arr, speed, tangent, is_turning, S_grid, ctx_curv, c_features

def run_step38_inference(batch_size=250, model_type="l3"):
    try:
        msg = f"🚀 [Step 38] GMM 4-Regime Regression Inference Started (batch_size={batch_size}, model_type={model_type})..."
        send_discord_notification(None, msg)
        print(msg)
        
        data_dir = Path("data/open")
        test_dir = data_dir / "test"
        
        # Load EqMotion test predictions
        print("Loading EqMotion test predictions...")
        s4_preds_df = pd.read_csv("experiments/step12/step4_preds_test.csv").set_index('id')
        
        # Load Scaler, GMM model, and regime mapping from step35
        print("Loading Scaler, GMM, and mapping config...")
        models_dir = Path("experiments/step35_four_regime/models")
        with open(models_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(models_dir / "gmm_model.pkl", "rb") as f:
            gmm = pickle.load(f)
        with open(models_dir / "regime_mapping.pkl", "rb") as f:
            mapping = pickle.load(f)
            
        # Load the 4 TabularPredictors from step38
        predictors = {}
        for regime in ["slow_straight", "fast_straight", "slow_extreme_turning", "fast_turning"]:
            model_path = Path(f"step38_2regime_regression/models/ranker_v38_{regime}")
            print(f"Loading AutoGluon Predictor for {regime} from {model_path}...")
            predictors[regime] = TabularPredictor.load(str(model_path))
            
        submission_df = pd.read_csv(data_dir / "sample_submission.csv")
        test_ids = submission_df['id'].values
        
        predictions = []
        displacements = []
        
        print(f"Running batched inference (batch_size={batch_size}, model_type={model_type}) with Regression Consensus Blending...")

        for i in range(0, len(test_ids), batch_size):
            batch_ids = test_ids[i : i + batch_size]
            
            batch_cand_idx = []
            batch_spec_par = []
            batch_spec_perp = []
            batch_spec_ts = []
            batch_spec_dmp = []
            batch_spec_jerk = []
            batch_is_prior = []
            
            batch_grid_scale = []
            batch_cand_speed = []
            batch_cand_speed_ratio = []
            batch_cand_turn_angle = []
            batch_cand_turn_rate = []
            batch_cand_accel = []
            batch_cand_lat_accel = []
            
            batch_ctx = None
            
            batch_dist_p0 = []
            batch_dist_s7 = []
            batch_dist_s4 = []
            
            batch_info = []
            fid_to_regime = {}
            
            start_idx = 0
            for fid in batch_ids:
                fpath = test_dir / f"{fid}.csv"
                df = pd.read_csv(fpath)
                xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
                
                # Calculate Priors
                p0 = xyz[-1]
                last_vel = xyz[-1] - xyz[-2]
                s7_pos = p0 + 2.0 * last_vel
                s4_pos = s4_preds_df.loc[fid].to_numpy()
                priors = [s7_pos, s4_pos]
                
                # Extract features and predict GMM regime
                ctx = extract_context_features(xyz)
                
                feat_vector = np.array([[ctx[feat] for feat in CLUSTER_FEATURES]], dtype=np.float32)
                feat_scaled = scaler.transform(feat_vector)
                cluster_idx = gmm.predict(feat_scaled)[0]
                regime_name = mapping[cluster_idx]
                fid_to_regime[fid] = regime_name
                
                # Generate candidates
                cands, spec_arr, speed, tangent, is_turning, S_grid, ctx_curv, c_features = make_candidates_vectorized(xyz, priors=priors, end_idx=-1, horizon=2, regime=regime_name)
                
                N_c = len(cands)
                
                batch_cand_idx.append(spec_arr["cand_idx"])
                batch_spec_par.append(spec_arr["spec_par"])
                batch_spec_perp.append(spec_arr["spec_perp"])
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
                    "speed": speed,
                    "tangent": tangent,
                    "p_saccade": ctx["ctx_p_saccade"],
                    "is_turning": is_turning,
                    "S_grid": S_grid,
                    "ctx_curv": ctx_curv
                })
                start_idx = end_idx
                
            pred_dict = {
                "cand_idx": np.concatenate(batch_cand_idx),
                "spec_par": np.concatenate(batch_spec_par),
                "spec_perp": np.concatenate(batch_spec_perp),
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
                
                "dist_to_p0": np.concatenate(batch_dist_p0),
                "dist_to_s7": np.concatenate(batch_dist_s7),
                "dist_to_s4": np.concatenate(batch_dist_s4)
            }
            for k in batch_ctx.keys():
                pred_dict[k] = np.concatenate(batch_ctx[k])
                
            pred_data = pd.DataFrame(pred_dict)
            
            # Predict distances using the routed GMM regimes
            batch_predicted_dists = np.zeros(len(pred_data), dtype=np.float32)
            
            regime_masks = {
                "slow_straight": np.zeros(len(pred_data), dtype=bool),
                "slow_extreme_turning": np.zeros(len(pred_data), dtype=bool),
                "fast_straight": np.zeros(len(pred_data), dtype=bool),
                "fast_turning": np.zeros(len(pred_data), dtype=bool)
            }
            
            for info in batch_info:
                fid = info['id']
                regime = fid_to_regime[fid]
                regime_masks[regime][info['start'] : info['end']] = True
                
            for regime, mask in regime_masks.items():
                if np.any(mask):
                    pred_data_reg = pred_data[mask]
                    predictor = predictors[regime]
                    
                    if model_type == "l3":
                        pred_dists = predictor.predict(pred_data_reg)
                    elif model_type == "l2":
                        pred_dists = predictor.predict(pred_data_reg, model="WeightedEnsemble_L2")
                    elif model_type == "fast_tree":
                        pred_cat = predictor.predict(pred_data_reg, model="CatBoost_BAG_L1")
                        pred_xgb = predictor.predict(pred_data_reg, model="XGBoost_BAG_L1")
                        pred_dists = 0.6 * pred_cat + 0.4 * pred_xgb
                    elif model_type == "ultra_fast":
                        pred_dists = predictor.predict(pred_data_reg, model="CatBoost_BAG_L1")
                    else:
                        pred_dists = predictor.predict(pred_data_reg)
                        
                    batch_predicted_dists[mask] = pred_dists.values

            # Post-process using Anisotropic Spatial Consensus Blending
            for info in batch_info:
                pred_dists = batch_predicted_dists[info['start'] : info['end']]
                
                # Clip distances to minimum of 0 to avoid potential negative extrapolation from linear ensembles
                pred_dists = np.clip(pred_dists, 0.0, None)
                
                # Convert predicted distance to a probability-like score (smaller = higher score)
                # We use a scale of 0.005 m (0.5 cm) as the normalizer.
                probs = np.exp(- pred_dists / 0.005)
                
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
        out_path = Path("outputs/step38_2regime_regression/submission.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Step 38 submission saved to {out_path}")
        
        displacements = np.array(displacements)
        mean_disp_cm = displacements.mean() * 100
        max_disp_cm = displacements.max() * 100
        
        success_msg = (
            f"✅ [Step 38] GMM 4-Regime Regression Inference Finished!\n"
            f"Saved to: `{out_path}`\n"
            f"Physical Displacement from Last Observed Point (p0):\n"
            f"- **Mean**: **{mean_disp_cm:.4f} cm**\n"
            f"- **Max**: **{max_disp_cm:.4f} cm**"
        )
        send_discord_notification(None, success_msg)
        print(success_msg)
        
    except BaseException as e:
        error_msg = f"❌ [Step 38] Inference ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(None, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 38 GMM 4-Regime Regression Inference")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size of test cases to process (default: 250)")
    parser.add_argument("--model_type", type=str, default="l3", choices=["l3", "l2", "fast_tree", "ultra_fast"],
                        help="Model prediction type: 'l3' (WeightedEnsemble_L3), 'l2' (WeightedEnsemble_L2), 'fast_tree' (0.6 CatBoost + 0.4 XGBoost), 'ultra_fast' (CatBoost L1 only)")
    args = parser.parse_args()
    
    run_step38_inference(batch_size=args.batch_size, model_type=args.model_type)
