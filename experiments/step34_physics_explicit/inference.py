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
import traceback
import torch
torch.set_num_threads(4)

sys.path.append(os.getcwd())
from step34_physics_explicit.physics import (
    CANDIDATES_SLOW, CANDIDATES_FAST, get_damping_factors, extract_multi_scale_derivatives, PINV_W5_QUAD, EPS
)
from step34_physics_explicit.prepare_data import extract_context_features
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

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

M_SLOW = compute_multiplier_matrix(CANDIDATES_SLOW)
M_FAST = compute_multiplier_matrix(CANDIDATES_FAST)

# Precomputed candidate spec arrays for vectorized DataFrame creation
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

def make_candidates_vectorized(x, priors, end_idx=-1, horizon=2):
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
    
    # Extract smooth kinematics and dynamic grid scaling
    ctx = extract_multi_scale_derivatives(x_sliced)
    p_sacc = ctx["ctx_p_saccade"]
    ctx_lat_accel = ctx["ctx_lat_accel"]
    
    # Calculate curvature at last step (raw curvature)
    cross_va = np.cross(d1, acc)
    ctx_curv = np.linalg.norm(cross_va) / (speed**3 + 1e-6)
    
    # Check if this slow trajectory is turning
    is_turning = (ctx_curv > 12.0) or (ctx_lat_accel > 0.0020)
    
    if len(x_sliced) >= 5:
        x_w5 = x_sliced[-5:]
        coeffs_w5 = PINV_W5_QUAD @ x_w5
        acc_smooth = 2.0 * coeffs_w5[0]
    else:
        acc_smooth = acc
        
    acc_par_scalar = np.sum(acc_smooth * tangent)
    acc_par = acc_par_scalar * tangent
    acc_perp_vec = acc_smooth - acc_par
    
    D = np.vstack([d1, d2, acc_par, acc_perp_vec, jerk]) # shape (5, 3)
    
    if speed <= 0.0234 and not is_turning:
        M = M_SLOW.copy()
        spec_arr = CANDS_SLOW_SPECS
        S_grid = 1.0
    else:
        M = M_FAST.copy()
        spec_arr = CANDS_FAST_SPECS
        S_grid = float(np.clip(1.0 + 0.6 * p_sacc + 0.05 * (ctx_curv - 6.0), 1.0, 2.5))
        
    # Apply dynamic scaling to the multiplier matrix columns 2 and 3 (par & perp)
    M[:, 2] *= S_grid
    M[:, 3] *= S_grid
    
    preds = p0 + M @ D # shape (N, 3)
    
    s7_pos, s4_pos = priors
    all_cands = np.vstack([preds, s7_pos, s4_pos])
    
    # Compute candidate-level physical features vectorially
    d_cands = all_cands - p0  # shape (N_c, 3)
    c_speeds = np.linalg.norm(d_cands, axis=-1) / 2.0  # displacement speed per step (40ms)
    
    v0_norm = np.linalg.norm(d1)
    v0_hat = d1 / (v0_norm + EPS)
    d_cands_norm = np.linalg.norm(d_cands, axis=-1)
    d_cands_hat = d_cands / (d_cands_norm[:, None] + EPS)
    
    cos_theta = np.sum(d_cands_hat * v0_hat, axis=-1)
    c_turn_angles = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180.0 / np.pi)
    
    hist_turn_deg = float(ctx["smooth_turn_w5"]) * (180.0 / np.pi)
    c_turn_rates = c_turn_angles - hist_turn_deg
    
    # candidate acceleration: (c_i - p0 - 2 * v0) / 2
    c_acc = (all_cands - p0 - 2.0 * d1) / 2.0
    c_accels = np.linalg.norm(c_acc, axis=-1)
    c_acc_par = np.sum(c_acc * v0_hat, axis=-1)[:, None] * v0_hat
    c_acc_perp = c_acc - c_acc_par
    c_lat_accels = np.linalg.norm(c_acc_perp, axis=-1)
    
    c_features = {
        "grid_scale": np.full(len(all_cands), S_grid, dtype=np.float32),
        "cand_speed": c_speeds,
        "cand_turn_angle": c_turn_angles,
        "cand_turn_rate": c_turn_rates,
        "cand_accel": c_accels,
        "cand_lat_accel": c_lat_accels
    }
    
    spec_arr_dynamic = spec_arr
    
    return all_cands, spec_arr_dynamic, speed, tangent, is_turning, S_grid, ctx_curv, c_features

def run_step34_inference():
    try:
        send_discord_notification(URL, "🚀 [Step 34] Physics-Explicit Inference Started...")
        
        data_dir = Path("data/open")
        test_dir = data_dir / "test"
        
        # Load EqMotion test predictions
        print("Loading EqMotion test predictions...")
        s4_preds_df = pd.read_csv("step12/step4_preds_test.csv").set_index('id')
        
        # Load Predictors
        model_path_slow = 'step34_physics_explicit/models/ranker_v34_slow'
        model_path_fast = 'step34_physics_explicit/models/ranker_v34_fast'
        print("Loading AutoGluon Rankers...")
        predictor_slow = TabularPredictor.load(model_path_slow)
        predictor_fast = TabularPredictor.load(model_path_fast)
        
        submission_df = pd.read_csv(data_dir / "sample_submission.csv")
        test_ids = submission_df['id'].values
        
        predictions = []
        displacements = []
        
        batch_size = 20
        print(f"Running batched inference (batch_size={batch_size}) with Anisotropic Gaussian Blending...")
        
        for i in range(0, len(test_ids), batch_size):
            batch_ids = test_ids[i : i + batch_size]
            
            # Feature lists for batch construction
            batch_cand_idx = []
            batch_spec_par = []
            batch_spec_perp = []
            batch_spec_ts = []
            batch_spec_dmp = []
            batch_spec_jerk = []
            batch_is_prior = []
            
            # Candidate physics features
            batch_grid_scale = []
            batch_cand_speed = []
            batch_cand_turn_angle = []
            batch_cand_turn_rate = []
            batch_cand_accel = []
            batch_cand_lat_accel = []
            
            # We initialize dictionary for contexts
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
                
                # Calculate Priors
                p0 = xyz[-1]
                last_vel = xyz[-1] - xyz[-2]
                s7_pos = p0 + 2.0 * last_vel
                s4_pos = s4_preds_df.loc[fid].to_numpy()
                priors = [s7_pos, s4_pos]
                
                # Extract features and candidates
                ctx = extract_context_features(xyz)
                cands, spec_arr, speed, tangent, is_turning, S_grid, ctx_curv, c_features = make_candidates_vectorized(xyz, priors=priors, end_idx=-1, horizon=2)
                
                N_c = len(cands)
                
                # Append to lists
                batch_cand_idx.append(spec_arr["cand_idx"])
                batch_spec_par.append(spec_arr["spec_par"])
                batch_spec_perp.append(spec_arr["spec_perp"])
                batch_spec_ts.append(spec_arr["spec_ts"])
                batch_spec_dmp.append(spec_arr["spec_dmp"])
                batch_spec_jerk.append(spec_arr["spec_jerk"])
                batch_is_prior.append(spec_arr["is_prior"])
                
                # Physical features
                batch_grid_scale.append(c_features["grid_scale"])
                batch_cand_speed.append(c_features["cand_speed"])
                batch_cand_turn_angle.append(c_features["cand_turn_angle"])
                batch_cand_turn_rate.append(c_features["cand_turn_rate"])
                batch_cand_accel.append(c_features["cand_accel"])
                batch_cand_lat_accel.append(c_features["cand_lat_accel"])
                
                if batch_ctx is None:
                    batch_ctx = {k: [] for k in ctx.keys()}
                for k, v in ctx.items():
                    batch_ctx[k].append(np.full(N_c, v, dtype=np.float32))
                    
                # Compute distances vectorially
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
                
            # Concatenate lists to create features DataFrame
            pred_dict = {
                "cand_idx": np.concatenate(batch_cand_idx),
                "spec_par": np.concatenate(batch_spec_par),
                "spec_perp": np.concatenate(batch_spec_perp),
                "spec_ts": np.concatenate(batch_spec_ts),
                "spec_dmp": np.concatenate(batch_spec_dmp),
                "spec_jerk": np.concatenate(batch_spec_jerk),
                "is_prior": np.concatenate(batch_is_prior),
                
                # Physical features
                "grid_scale": np.concatenate(batch_grid_scale),
                "cand_speed": np.concatenate(batch_cand_speed),
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
            
            # Predict Hit Probability routing cruising vs turning dynamically
            batch_scores = np.zeros(len(pred_data), dtype=np.float32)
            
            # Cruise / Turn mask matching data prep
            is_turning_mask = (pred_data['ctx_curv'] > 12.0) | (pred_data['ctx_lat_accel'] > 0.0020)
            slow_cruising_mask = (pred_data['ctx_speed'] <= 0.0234) & (~is_turning_mask)
            
            pred_data_slow = pred_data[slow_cruising_mask]
            pred_data_fast = pred_data[~slow_cruising_mask]
            
            if len(pred_data_slow) > 0:
                proba_slow = predictor_slow.predict_proba(pred_data_slow)
                score_col_slow = 1 if 1 in proba_slow.columns else proba_slow.columns[0]
                batch_scores[slow_cruising_mask] = proba_slow[score_col_slow].values
                
            if len(pred_data_fast) > 0:
                proba_fast = predictor_fast.predict_proba(pred_data_fast)
                score_col_fast = 1 if 1 in proba_fast.columns else proba_fast.columns[0]
                batch_scores[~slow_cruising_mask] = proba_fast[score_col_fast].values
            
            # Post-process each trajectory in the batch using Anisotropic Spatial Probability Blending
            for info in batch_info:
                probs = batch_scores[info['start'] : info['end']]
                cands = info['cands']
                t = info['tangent']
                p_sacc = info['p_saccade']
                speed = info['speed'] # displacement per 10ms
                speed_m_s = speed / 0.01
                
                S_grid = info['S_grid']
                S_scale = S_grid ** 1.5
                
                sigma_tangential = np.clip(0.0035 + 0.005 * speed_m_s * p_sacc, 0.003, 0.011) * S_scale
                sigma_normal = np.clip(0.0035 + 0.0015 * speed_m_s * p_sacc, 0.003, 0.006) * S_scale
                
                # Anisotropic Spatial Probability Blending
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
                
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed batch {i//batch_size + 1}/{(len(test_ids)-1)//batch_size + 1}...")
            
        out_df = pd.DataFrame(predictions)
        out_path = Path("outputs/step34_physics_explicit/submission.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Step 34 submission saved to {out_path}")
        
        displacements = np.array(displacements)
        mean_disp_cm = displacements.mean() * 100
        max_disp_cm = displacements.max() * 100
        
        success_msg = (
            f"✅ [Step 34] Physics-Explicit Inference Finished!\n"
            f"Saved to: `{out_path}`\n"
            f"Physical Displacement from Last Observed Point (p0):\n"
            f"- **Mean**: **{mean_disp_cm:.4f} cm**\n"
            f"- **Max**: **{max_disp_cm:.4f} cm**"
        )
        send_discord_notification(URL, success_msg)
        print(success_msg)
        
    except BaseException as e:
        error_msg = f"❌ [Step 34] Inference ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    run_step34_inference()
