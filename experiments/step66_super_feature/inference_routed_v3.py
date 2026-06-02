import os
import sys
import json
import csv
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor
import torch

# Add project root and step55 to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "step55_sota_2026")))
from step55_sota_2026.model import Sota2026TrajectoryModel, extract_features as extract_s55_features
from utils.notifier import send_discord_notification

EPS = 1e-8

def classify_regimes(X):
    N = X.shape[0]
    last_v = (X[:, -1] - X[:, -2]) / 0.04
    speeds = np.linalg.norm(last_v, axis=1)
    prev_v = (X[:, -2] - X[:, -3]) / 0.04
    last_a = (last_v - prev_v) / 0.04
    t_dir = last_v / (speeds[:, None] + EPS)
    acc_par_scalar = np.sum(last_a * t_dir, axis=1)
    acc_perp = last_a - acc_par_scalar[:, None] * t_dir
    acc_perp_norm = np.linalg.norm(acc_perp, axis=1)
    cross_prod = np.cross(last_v, last_a, axis=1)
    cross_norm = np.linalg.norm(cross_prod, axis=1)
    curvature = cross_norm / (speeds ** 3 + EPS)
    is_steering = (curvature > 6.0) | (acc_perp_norm > 1.8)
    regimes = np.zeros(N, dtype=int)
    for i in range(N):
        if is_steering[i]:
            regimes[i] = 2
        elif speeds[i] <= 0.50:
            regimes[i] = 0
        else:
            regimes[i] = 1
    return regimes, speeds, curvature, acc_perp_norm, acc_par_scalar

def main():
    send_discord_notification(None, "🚀 Started: [Step 67 inference_routed_v3.py] Running Track B Hybrid AutoML-Routing Inference...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Setup paths
        data_dir_s52 = Path("step52_focal_ode/data")
        data_dir_s53 = Path("step53_adaptive_geometry/data")
        models_dir_s53 = Path("step53_adaptive_geometry/models")
        data_dir_s55 = Path("step55_sota_2026/data")
        data_dir_s57 = Path("step57_spatiotemporal_ai/data")
        data_dir_s62 = Path("step62_spatiotemporal_ai/data")
        data_dir_s63 = Path("step63_spatiotemporal_ai/data")
        data_dir_s64 = Path("step64_spatiotemporal_ai/data")
        data_dir_s65 = Path("step65_spatiotemporal_ai/data")
        
        data_dir_active = Path("step66_super_feature/data")
        models_dir_active = Path("step66_super_feature/models")
        out_dir = Path("outputs/step66_super_feature")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading core test datasets...")
        test_x = np.load(data_dir_s55 / "test_x.npy")
        test_candidates = np.load(data_dir_s55 / "test_candidates.npy")
        test_candidates_hybrid = np.load(data_dir_active / "test_candidates_hybrid_v3.npy") # [10000, 43, 3]
        
        with open(data_dir_s55 / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        # ------------------ RECONSTRUCT ULTIMATE BLEND V12 CONSENSUS FOR EASY CASES ------------------
        print("Reconstructing Ultimate Blend V12 Consensus for Easy snap-routing...")
        
        # Load individual predictions for test set
        s47_soft_test = np.load(data_dir_s52 / "step47_test_soft.npy")
        s47_argmax_test = np.load(data_dir_s52 / "step47_test_argmax.npy")
        s53_soft_test = np.load(data_dir_s53 / "step52_test.npy")
        
        s57_test_1step = np.load(data_dir_s57 / "test_preds_cfm_1step.npy")
        s57_test_2step = np.load(data_dir_s57 / "test_preds_cfm_2step.npy")
        s62_test_1step = np.load(data_dir_s62 / "test_preds_cfm_1step.npy")
        s62_test_2step = np.load(data_dir_s62 / "test_preds_cfm_2step.npy")
        s63_test_1step = np.load(data_dir_s63 / "test_preds_cfm_1step.npy")
        s63_test_2step = np.load(data_dir_s63 / "test_preds_cfm_2step.npy")
        s64_test_1step = np.load(data_dir_s64 / "test_preds_cfm_1step.npy")
        s64_test_2step = np.load(data_dir_s64 / "test_preds_cfm_2step.npy")
        s65_test_1step = np.load(data_dir_s65 / "test_preds_cfm_1step.npy")
        s65_test_2step = np.load(data_dir_s65 / "test_preds_cfm_2step.npy")
        
        # Reconstruct S55 test predictions (needs torch loading)
        s55_test_preds = np.zeros((len(test_x), 3))
        for fold in range(5):
            stats = torch.load(Path("step55_sota_2026/models") / f"stats_fold_{fold}.pt", map_location=device)
            tr_mean = stats["mean"]
            tr_std = stats["std"]
            model_s55 = Sota2026TrajectoryModel(feature_dim=38, latent_dim=64, num_candidates=36).to(device)
            model_s55.load_state_dict(torch.load(Path("step55_sota_2026/models") / f"model_fold_{fold}.pt", map_location=device))
            model_s55.eval()
            with torch.no_grad():
                test_tensor_dev = torch.tensor(test_x, dtype=torch.float32).to(device)
                ft_te, df_te, _, _, _ = extract_s55_features(test_tensor_dev, tr_mean.to(device), tr_std.to(device))
                candidates_te = torch.tensor(test_candidates, dtype=torch.float32).to(device)
                fold_pred, _ = model_s55(ft_te, df_te, candidates_te)
                s55_test_preds += fold_pred.cpu().numpy() / 5.0
                
        regimes_te, speeds_te, curvature_te, acc_perp_te, acc_par_te = classify_regimes(test_x)
        cruising_te_idx = np.where(regimes_te == 0)[0]
        gliding_te_idx = np.where(regimes_te == 1)[0]
        steering_te_idx = np.where(regimes_te == 2)[0]
        
        # Blend Weights
        w_cr = [0.5, 0.05, 0.0, 0.0, 0.05, 0.25, 0.15, 0.0]
        w_gl = [0.30, 0.0, 0.35, 0.25, 0.0, 0.1, 0.0, 0.0]
        w_st = [0.55, 0.20, 0.05, 0.10, 0.0, 0.0, 0.10, 0.0]
        
        v_cr = ["soft", "argmax", "1step", "1step", "1step", "1step", "1step"]
        v_gl = ["argmax", "soft", "1step", "1step", "1step", "1step", "1step"]
        v_st = ["soft", "soft", "1step", "1step", "1step", "1step", "1step"]
        
        blended_test = np.zeros_like(s55_test_preds)
        
        def reconstruct_blended_test(idx, w, variants):
            s47_te = s47_soft_test if variants[0] == "soft" else s47_argmax_test
            s53_te = s53_soft_test
            s57_te = s57_test_1step if variants[2] == "1step" else s57_test_2step
            s62_te = s62_test_1step if variants[3] == "1step" else s62_test_2step
            s63_te = s63_test_1step if variants[4] == "1step" else s63_test_2step
            s64_te = s64_test_1step if variants[5] == "1step" else s64_test_2step
            s65_te = s65_test_1step if variants[6] == "1step" else s65_test_2step
            
            preds = (
                w[0] * s47_te[idx] +
                w[1] * s53_te[idx] +
                w[2] * s55_test_preds[idx] +
                w[3] * s57_te[idx] +
                w[4] * s62_te[idx] +
                w[5] * s63_te[idx] +
                w[6] * s64_te[idx] +
                w[7] * s65_te[idx]
            )
            return preds
            
        blended_test[cruising_te_idx] = reconstruct_blended_test(cruising_te_idx, w_cr, v_cr)
        blended_test[gliding_te_idx] = reconstruct_blended_test(gliding_te_idx, w_gl, v_gl)
        blended_test[steering_te_idx] = reconstruct_blended_test(steering_te_idx, w_st, v_st)
        
        # ------------------ PREDICT DISTANCES USING HIGH-QUALITY AUTOML FOR HARD CASES ------------------
        print("Loading test tabular dataset for AutoML predictions...")
        test_df = pd.read_parquet(data_dir_active / "test_tabular_v3.parquet")
        test_data = test_df.drop(columns=['sample_idx', 'cand_idx'])
        
        folds = 5
        pred_dists_list = []
        
        print("Predicting with 5-Fold High-Quality AutoML models...")
        for fold in range(folds):
            fold_model_path = models_dir_active / f"autogluon_fold_v3_{fold}"
            print(f"  Loading Fold {fold+1} Predictor...")
            predictor = TabularPredictor.load(fold_model_path)
            fold_preds = predictor.predict(test_data)
            pred_dists_list.append(fold_preds)
            
        print("Averaging predicted distances...")
        avg_pred_dists = np.mean(pred_dists_list, axis=0)
        test_df['pred_dist'] = avg_pred_dists
        
        # Frenet Vector Calculations for Anisotropic Spatial Blending
        print("Calculating Frenet vectors for test set (Hard cases)...")
        v_te = (test_x[:, 1:] - test_x[:, :-1]) / 0.04
        a_te = (v_te[:, 1:] - v_te[:, :-1]) / 0.04
        v_last_te = v_te[:, -1]
        a_last_te = a_te[:, -1]
        
        speeds = np.linalg.norm(v_last_te, axis=1)
        T_te = v_last_te / (speeds[:, None] + 1e-8)
        a_par = np.sum(a_last_te * T_te, axis=1)
        a_perp = a_last_te - a_par[:, None] * T_te
        a_perp_norm = np.linalg.norm(a_perp, axis=1)
        N_te = a_perp / (a_perp_norm[:, None] + 1e-8)
        
        fallback = np.zeros_like(N_te)
        axis = np.argmin(np.abs(T_te), axis=1)
        fallback[np.arange(N_te.shape[0]), axis] = 1.0
        fallback = fallback - np.sum(fallback * T_te, axis=1)[:, None] * T_te
        fallback = fallback / (np.linalg.norm(fallback, axis=1)[:, None] + 1e-8)
        N_te = np.where(a_perp_norm[:, None] > 1e-6, N_te, fallback)
        
        B_te = np.cross(T_te, N_te, axis=1)
        B_te = B_te / (np.linalg.norm(B_te, axis=1)[:, None] + 1e-8)
        
        # ------------------ APPLY HYBRID SNAP-ROUTING DECISION ------------------
        print("Performing Hybrid snap-routing on Test set...")
        
        cand_std = np.std(test_candidates_hybrid, axis=1) # [10000, 3]
        cand_spread = np.mean(cand_std, axis=1) # [10000]
        
        easy_thresh = 0.0038 # 0.38cm in meters
        easy_mask = cand_spread <= easy_thresh
        num_easy = np.sum(easy_mask)
        print(f"Test Set Routing: Easy (Snap-routing)={num_easy} ({num_easy/len(test_ids)*100.0:.2f}%), Hard (AutoML Blended)={len(test_ids)-num_easy} ({(len(test_ids)-num_easy)/len(test_ids)*100.0:.2f}%)")
        
        submission_coords = np.zeros((len(test_ids), 3))
        
        # Hyperparameters for AutoML Anisotropic Spatial Blending in Hard cases
        sigma_L = 0.015 # 1.5 cm
        sigma_T = 0.025 # 2.5 cm
        temp = 0.3
        
        easy_snapped_count = 0
        hard_blended_count = 0
        
        for idx in range(len(test_ids)):
            if easy_mask[idx]:
                # Easy Case: Snap Ultimate Blend V12 Consensus to the nearest candidate
                cands = test_candidates_hybrid[idx]
                raw_pt = blended_test[idx]
                dists = np.linalg.norm(cands - raw_pt, axis=1)
                best_cand_idx = np.argmin(dists)
                submission_coords[idx] = cands[best_cand_idx]
                easy_snapped_count += 1
            else:
                # Hard Case: Use High-Quality AutoML with Anisotropic Spatial Blending
                sample_preds = test_df[test_df['sample_idx'] == idx].sort_values('cand_idx')
                cands = test_candidates_hybrid[idx]
                preds = sample_preds['pred_dist'].values # in cm
                
                w = np.exp(-preds / temp)
                diffs = cands[:, None, :] - cands[None, :, :]
                
                T = T_te[idx]
                N = N_te[idx]
                B = B_te[idx]
                
                d_T = np.sum(diffs * T, axis=2)
                d_N = np.sum(diffs * N, axis=2)
                d_B = np.sum(diffs * B, axis=2)
                
                exponent = -0.5 * ( (d_T**2 / (sigma_L**2)) + (d_N**2 / (sigma_T**2)) + (d_B**2 / (sigma_T**2)) )
                kernel_val = np.exp(exponent)
                
                scores = np.sum(w[None, :] * kernel_val, axis=1)
                blended_best_idx = np.argmax(scores)
                submission_coords[idx] = cands[blended_best_idx]
                hard_blended_count += 1
                
        # Write routed final submission to CSV
        sub_path = out_dir / "submission_routed_v3.csv"
        print(f"Writing hybrid routed submission to {sub_path}...")
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, coord in zip(test_ids, submission_coords):
                writer.writerow([sample_id, f"{coord[0]:.9f}", f"{coord[1]:.9f}", f"{coord[2]:.9f}"])
                
        # Displacement verification from last point
        p_last_te_arr = test_x[:, -1]
        diffs = np.linalg.norm(submission_coords - p_last_te_arr, axis=1)
        disp_stats = f"Displacement stats (cm) -> Mean: {diffs.mean()*100:.3f}, Max: {diffs.max()*100:.3f}, Std: {diffs.std()*100:.3f}"
        print(f"\n=================== Hybrid Routed V3 Summary ===================")
        print(f"Submission: {sub_path}")
        print(f"Easy Snapped: {easy_snapped_count}  |  Hard AutoML Blended: {hard_blended_count}")
        print(f"Displacement: {disp_stats}")
        print("================================================================")
        
        success_msg = f"✅ Finished: [Step 67 Routed V3] Final submission generated. Mean: {diffs.mean()*100:.3f}cm, Easy: {easy_snapped_count}, Hard: {hard_blended_count}"
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 67 Routed V3] Hybrid Routed Inference failed.\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
