import os
import sys
import json
import csv
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
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

def extract_expanded_features(X, speeds, curvature, acc_perp, acc_par, pred_disp_norm, blended_coords):
    p_last = X[:, -1]
    
    # 1. 3D Jerk and Saccadic dynamics
    v = (X[:, 1:] - X[:, :-1]) / 0.04
    a = (v[:, 1:] - v[:, :-1]) / 0.04
    j = (a[:, 1:] - a[:, :-1]) / 0.04
    j_last = j[:, -1]
    jerk_norm = np.linalg.norm(j_last, axis=1)
    
    # 2. Polaris spherical coordinates
    rho = np.linalg.norm(X, axis=2)
    theta = np.arctan2(X[:, :, 1], X[:, :, 0])
    phi = np.arccos(np.clip(X[:, :, 2] / (rho + EPS), -1.0, 1.0))
    d_rho = (rho[:, 1:] - rho[:, :-1]) / 0.04
    d_theta = (theta[:, 1:] - theta[:, :-1]) / 0.04
    d_phi = (phi[:, 1:] - phi[:, :-1]) / 0.04
    d_theta = np.remainder(d_theta + np.pi, 2 * np.pi) - np.pi
    d_phi = np.remainder(d_phi + np.pi, 2 * np.pi) - np.pi
    df_sph = np.stack([d_rho, d_theta, d_phi], axis=-1)
    a_sph = df_sph[:, 1:] - df_sph[:, :-1]
    a_sph_last = a_sph[:, -1]
    
    # 3. Z displacement vertical drift
    z_diff = np.abs(blended_coords[:, 2] - p_last[:, 2])
    
    feat = np.column_stack([
        speeds, curvature, acc_perp, acc_par, pred_disp_norm, z_diff,
        jerk_norm, a_sph_last[:, 0], a_sph_last[:, 1], a_sph_last[:, 2]
    ])
    return feat

def main():
    send_discord_notification(None, "🚀 Started: [Step 66 inference_s67_final.py] Generating Cross-Attention CFM & s67-Guided Ensemble submissions...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load directories
        data_dir_s52 = Path("step52_focal_ode/data")
        data_dir_s53 = Path("step53_adaptive_geometry/data")
        models_dir_s53 = Path("step53_adaptive_geometry/models")
        data_dir_s55 = Path("step55_sota_2026/data")
        data_dir_s57 = Path("step57_spatiotemporal_ai/data")
        data_dir_s62 = Path("step62_spatiotemporal_ai/data")
        data_dir_s63 = Path("step63_spatiotemporal_ai/data")
        data_dir_s64 = Path("step64_spatiotemporal_ai/data")
        data_dir_s65 = Path("step65_spatiotemporal_ai/data")
        models_dir_s67 = Path("step66_super_feature/models/s67")
        out_dir = Path("outputs/step66_super_feature")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        train_x = np.load(data_dir_s55 / "train_x.npy")
        train_y = np.load(data_dir_s55 / "train_y.npy")
        test_x = np.load(data_dir_s55 / "test_x.npy")
        test_candidates = np.load(data_dir_s55 / "test_candidates.npy")
        
        with open(data_dir_s55 / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        p_last_te = test_x[:, -1]
        
        # Load s67 predictions
        print("Loading s67 predictions...")
        s67_test_1step = np.load(models_dir_s67 / "test_preds_cfm_1step.npy")
        s67_test_2step = np.load(models_dir_s67 / "test_preds_cfm_2step.npy")
        
        # 1. Option E: Continuous Solo s67 CFM
        sub_solo_path = out_dir / "submission_s67_continuous_solo.csv"
        print(f"Writing s67 continuous solo to {sub_solo_path}...")
        with sub_solo_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, coord in zip(test_ids, s67_test_1step):
                writer.writerow([sample_id, f"{coord[0]:.9f}", f"{coord[1]:.9f}", f"{coord[2]:.9f}"])
                
        diffs_solo = np.linalg.norm(s67_test_1step - p_last_te, axis=1)
        print(f"Option s67_solo | Mean: {diffs_solo.mean()*100:.3f}cm, Max: {diffs_solo.max()*100:.3f}cm")
        send_discord_notification(None, f"✅ Option [s67_solo] generated. Mean: {diffs_solo.mean()*100:.3f}cm")
        
        # 2. Option F: s67-Guided Powell Ensemble (th=0.80, shrink=0.80, gamma=0.30)
        print("Reconstructing blended OOF coordinates with Powell weights...")
        s47_soft_oof = np.load(data_dir_s52 / "step47_oof_soft.npy")
        s47_argmax_oof = np.load(data_dir_s52 / "step47_oof_argmax.npy")
        s53_soft_oof = np.load(models_dir_s53 / "oof_preds_soft.npy")
        s53_argmax_oof = np.load(models_dir_s53 / "oof_preds_argmax.npy")
        s55_soft_oof = np.load(data_dir_s55 / "oof_preds_soft.npy")
        
        s57_cfm_2step_oof = np.load(data_dir_s57 / "oof_preds_cfm_2step.npy")
        s57_cfm_1step_oof = np.load(data_dir_s57 / "oof_preds_cfm_1step.npy")
        s62_cfm_2step_oof = np.load(data_dir_s62 / "oof_preds_cfm_2step.npy")
        s62_cfm_1step_oof = np.load(data_dir_s62 / "oof_preds_cfm_1step.npy")
        s63_cfm_2step_oof = np.load(data_dir_s63 / "oof_preds_cfm_2step.npy")
        s63_cfm_1step_oof = np.load(data_dir_s63 / "oof_preds_cfm_1step.npy")
        s64_cfm_2step_oof = np.load(data_dir_s64 / "oof_preds_cfm_2step.npy")
        s64_cfm_1step_oof = np.load(data_dir_s64 / "oof_preds_cfm_1step.npy")
        s65_cfm_2step_oof = np.load(data_dir_s65 / "oof_preds_cfm_2step.npy")
        s65_cfm_1step_oof = np.load(data_dir_s65 / "oof_preds_cfm_1step.npy")
        s67_cfm_1step_oof = np.load(models_dir_s67 / "oof_preds_cfm_1step.npy")
        s67_cfm_2step_oof = np.load(models_dir_s67 / "oof_preds_cfm_2step.npy")
        
        regimes_tr, speeds_tr, curvature_tr, acc_perp_tr, acc_par_tr = classify_regimes(train_x)
        cruising_idx = np.where(regimes_tr == 0)[0]
        gliding_idx = np.where(regimes_tr == 1)[0]
        steering_idx = np.where(regimes_tr == 2)[0]
        
        # Load best weights if available, else use defaults
        weights_file = models_dir_s67 / "best_ensemble_weights.npz"
        if weights_file.exists():
            print(f"Loading optimized weights from {weights_file}...")
            weights_data = np.load(weights_file)
            w_cr = weights_data["w_cr"]
            w_gl = weights_data["w_gl"]
            w_st = weights_data["w_st"]
        else:
            print("Optimized weights file not found. Using default 9-model weights...")
            w_cr = np.array([0.5065, 0.0510, 0.0, 0.0, 0.0510, 0.2386, 0.1528, 0.0, 0.0])
            w_gl = np.array([0.2964, 0.0, 0.3518, 0.2513, 0.0, 0.1005, 0.0, 0.0, 0.0])
            w_st = np.array([0.5479, 0.1999, 0.0498, 0.1008, 0.0, 0.0, 0.1015, 0.0, 0.0])
            
        v_cr = ["soft", "argmax", "1step", "1step", "1step", "1step", "1step", "1step"]
        v_gl = ["argmax", "soft", "1step", "1step", "1step", "1step", "1step", "1step"]
        v_st = ["soft", "soft", "1step", "1step", "1step", "1step", "1step", "1step"]
        
        blended_oof = np.zeros_like(train_y)
        
        def reconstruct_blended_regime(idx, w, variants):
            s47_arr = s47_soft_oof if variants[0] == "soft" else s47_argmax_oof
            s53_arr = s53_soft_oof if variants[1] == "soft" else s53_argmax_oof
            s57_arr = s57_cfm_1step_oof if variants[2] == "1step" else s57_cfm_2step_oof
            s62_arr = s62_cfm_1step_oof if variants[3] == "1step" else s62_cfm_2step_oof
            s63_arr = s63_cfm_1step_oof if variants[4] == "1step" else s63_cfm_2step_oof
            s64_arr = s64_cfm_1step_oof if variants[5] == "1step" else s64_cfm_2step_oof
            s65_arr = s65_cfm_1step_oof if variants[6] == "1step" else s65_cfm_2step_oof
            s67_arr = s67_cfm_1step_oof if variants[7] == "1step" else s67_cfm_2step_oof
            
            preds = (
                w[0] * s47_arr[idx] +
                w[1] * s53_arr[idx] +
                w[2] * s55_soft_oof[idx] +
                w[3] * s57_arr[idx] +
                w[4] * s62_arr[idx] +
                w[5] * s63_arr[idx] +
                w[6] * s64_arr[idx] +
                w[7] * s65_arr[idx] +
                w[8] * s67_arr[idx]
            )
            return preds
            
        blended_oof[cruising_idx] = reconstruct_blended_regime(cruising_idx, w_cr, v_cr)
        blended_oof[gliding_idx] = reconstruct_blended_regime(gliding_idx, w_gl, v_gl)
        blended_oof[steering_idx] = reconstruct_blended_regime(steering_idx, w_st, v_st)
        
        # Fit RF Outlier Predictor
        print("Fitting RF outlier detector...")
        p_last_tr = train_x[:, -1]
        pred_disp_tr = blended_oof - p_last_tr
        pred_disp_norm_tr = np.linalg.norm(pred_disp_tr, axis=1)
        errors_tr = np.linalg.norm(blended_oof - train_y, axis=1)
        is_miss_tr = (errors_tr > 0.01).astype(int)
        
        features_tr = extract_expanded_features(train_x, speeds_tr, curvature_tr, acc_perp_tr, acc_par_tr, pred_disp_norm_tr, blended_oof)
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf_rf.fit(features_tr, is_miss_tr)
        
        # Reconstruct blended Test coordinates
        print("Reconstructing test predictions of individual models...")
        s47_soft_test = np.load(data_dir_s52 / "step47_test_soft.npy")
        s47_argmax_test = np.load(data_dir_s52 / "step47_test_argmax.npy")
        s53_soft_test = np.load(data_dir_s53 / "step52_test.npy")
        
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
        
        regimes_te, speeds_te, curvature_te, acc_perp_te, acc_par_te = classify_regimes(test_x)
        cruising_te_idx = np.where(regimes_te == 0)[0]
        gliding_te_idx = np.where(regimes_te == 1)[0]
        steering_te_idx = np.where(regimes_te == 2)[0]
        
        blended_test = np.zeros_like(s55_test_preds)
        
        def reconstruct_blended_test(idx, w, variants):
            s47_te = s47_soft_test if variants[0] == "soft" else s47_argmax_test
            s53_te = s53_soft_test
            s57_te = s57_test_1step if variants[2] == "1step" else s57_test_2step
            s62_te = s62_test_1step if variants[3] == "1step" else s62_test_2step
            s63_te = s63_test_1step if variants[4] == "1step" else s63_test_2step
            s64_te = s64_test_1step if variants[5] == "1step" else s64_test_2step
            s65_te = s65_test_1step if variants[6] == "1step" else s65_test_2step
            s67_te = s67_test_1step if variants[7] == "1step" else s67_test_2step
            
            preds = (
                w[0] * s47_te[idx] +
                w[1] * s53_te[idx] +
                w[2] * s55_test_preds[idx] +
                w[3] * s57_te[idx] +
                w[4] * s62_te[idx] +
                w[5] * s63_te[idx] +
                w[6] * s64_te[idx] +
                w[7] * s65_te[idx] +
                w[8] * s67_te[idx]
            )
            return preds
            
        blended_test[cruising_te_idx] = reconstruct_blended_test(cruising_te_idx, w_cr, v_cr)
        blended_test[gliding_te_idx] = reconstruct_blended_test(gliding_te_idx, w_gl, v_gl)
        blended_test[steering_te_idx] = reconstruct_blended_test(steering_te_idx, w_st, v_st)
        
        pred_disp_te = blended_test - p_last_te
        pred_disp_norm_te = np.linalg.norm(pred_disp_te, axis=1)
        features_te = extract_expanded_features(test_x, speeds_te, curvature_te, acc_perp_te, acc_par_te, pred_disp_norm_te, blended_test)
        prob_miss_te = clf_rf.predict_proba(features_te)[:, 1]
        
        # Apply th=0.70, shrink=0.95, gamma=0.60 using s67 as guidance
        th = 0.70
        shrink = 0.95
        gamma = 0.60
        
        submission_coords = blended_test.copy()
        damped_count = 0
        for idx in range(len(test_x)):
            if prob_miss_te[idx] > th and pred_disp_norm_te[idx] > 0.015:
                damped_coord = p_last_te[idx] + shrink * pred_disp_te[idx]
                submission_coords[idx] = (1.0 - gamma) * damped_coord + gamma * s67_test_1step[idx]
                damped_count += 1
                
        sub_guided_path = out_dir / "submission_continuous_s67_moderate.csv"
        print(f"Writing s67 guided continuous to {sub_guided_path}...")
        with sub_guided_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, coord in zip(test_ids, submission_coords):
                writer.writerow([sample_id, f"{coord[0]:.9f}", f"{coord[1]:.9f}", f"{coord[2]:.9f}"])
                
        diffs_guided = np.linalg.norm(submission_coords - p_last_te, axis=1)
        print(f"Option s67_guided | Mean: {diffs_guided.mean()*100:.3f}cm, Max: {diffs_guided.max()*100:.3f}cm, Damped: {damped_count}")
        send_discord_notification(None, f"✅ Option [s67_guided] generated. Mean: {diffs_guided.mean()*100:.3f}cm, Damped: {damped_count}")
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 66 inference_s67_final.py] S67 inference failed.\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
