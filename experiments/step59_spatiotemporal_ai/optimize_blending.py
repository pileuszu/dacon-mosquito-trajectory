import os
import sys
import json
import csv
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from step55_sota_2026.model import extract_features, Sota2026TrajectoryModel

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
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_dir_s52 = Path("step52_focal_ode/data")
        data_dir_s53 = Path("step53_adaptive_geometry/data")
        models_dir_s53 = Path("step53_adaptive_geometry/models")
        data_dir_s55 = Path("step55_sota_2026/data")
        models_dir_s55 = Path("step55_sota_2026/models")
        data_dir_s56 = Path("step56_flow_matching/data")
        data_dir_s57 = Path("step57_spatiotemporal_ai/data")
        
        # Current active step data
        data_dir_active = Path("step59_spatiotemporal_ai/data")
        out_dir = Path("outputs/step59_spatiotemporal_ai")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        train_x = np.load(data_dir_s55 / "train_x.npy")
        train_y = np.load(data_dir_s55 / "train_y.npy")
        test_x = np.load(data_dir_s55 / "test_x.npy")
        test_candidates = np.load(data_dir_s55 / "test_candidates.npy")
        with open(data_dir_s55 / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        s47_soft_oof = np.load(data_dir_s52 / "step47_oof_soft.npy")
        s47_argmax_oof = np.load(data_dir_s52 / "step47_oof_argmax.npy")
        s53_soft_oof = np.load(models_dir_s53 / "oof_preds_soft.npy")
        s53_argmax_oof = np.load(models_dir_s53 / "oof_preds_argmax.npy")
        s55_soft_oof = np.load(data_dir_s55 / "oof_preds_soft.npy")
        
        # Load active step OOF
        sactive_cfm_1step_oof = np.load(data_dir_active / "oof_preds_cfm_1step.npy")
        sactive_cfm_2step_oof = np.load(data_dir_active / "oof_preds_cfm_2step.npy")
        
        regimes_tr, speeds_tr, curvature_tr, acc_perp_tr, acc_par_tr = classify_regimes(train_x)
        cruising_idx = np.where(regimes_tr == 0)[0]
        gliding_idx = np.where(regimes_tr == 1)[0]
        steering_idx = np.where(regimes_tr == 2)[0]
        
        # Blending search Space
        weights_space = []
        for w1 in np.linspace(0.0, 1.0, 11):
            for w2 in np.linspace(0.0, 1.0 - w1, int(round((1.0 - w1) * 10)) + 1):
                for w3 in np.linspace(0.0, 1.0 - w1 - w2, int(round((1.0 - w1 - w2) * 10)) + 1):
                    w4 = 1.0 - w1 - w2 - w3
                    weights_space.append((w1, w2, w3, w4))
                    
        def optimize_4model_regime(idx_list):
            best_w = (0.0, 0.0, 0.0, 0.0)
            best_hr = 0.0
            best_s47_t, best_s53_t, best_act_t = "soft", "soft", "2step"
            for w1, w2, w3, w4 in weights_space:
                for s47_t, s47_arr in [("soft", s47_soft_oof), ("argmax", s47_argmax_oof)]:
                    for s53_t, s53_arr in [("soft", s53_soft_oof), ("argmax", s53_argmax_oof)]:
                        for act_t, act_arr in [("1step", sactive_cfm_1step_oof), ("2step", sactive_cfm_2step_oof)]:
                            preds = w1 * s47_arr[idx_list] + w2 * s53_arr[idx_list] + w3 * s55_soft_oof[idx_list] + w4 * act_arr[idx_list]
                            hr = np.mean(np.linalg.norm(preds - train_y[idx_list], axis=1) <= 0.01)
                            if hr > best_hr:
                                best_hr = hr
                                best_w = (w1, w2, w3, w4)
                                best_s47_t, best_s53_t, best_act_t = s47_t, s53_t, act_t
            return best_w, best_s47_t, best_s53_t, best_act_t, best_hr
            
        w_cr, s47_t_cr, s53_t_cr, act_t_cr, hr_cr = optimize_4model_regime(cruising_idx)
        w_gl, s47_t_gl, s53_t_gl, act_t_gl, hr_gl = optimize_4model_regime(gliding_idx)
        w_st, s47_t_st, s53_t_st, act_t_st, hr_st = optimize_4model_regime(steering_idx)
        
        blended_oof = np.zeros_like(train_y)
        
        def get_oof_comb(idx, w, s47_t, s53_t, act_t):
            s47_arr = s47_soft_oof if s47_t == "soft" else s47_argmax_oof
            s53_arr = s53_soft_oof if s53_t == "soft" else s53_argmax_oof
            act_arr = sactive_cfm_1step_oof if act_t == "1step" else sactive_cfm_2step_oof
            return w[0]*s47_arr[idx] + w[1]*s53_arr[idx] + w[2]*s55_soft_oof[idx] + w[3]*act_arr[idx]
            
        blended_oof[cruising_idx] = get_oof_comb(cruising_idx, w_cr, s47_t_cr, s53_t_cr, act_t_cr)
        blended_oof[gliding_idx] = get_oof_comb(gliding_idx, w_gl, s47_t_gl, s53_t_gl, act_t_gl)
        blended_oof[steering_idx] = get_oof_comb(steering_idx, w_st, s47_t_st, s53_t_st, act_t_st)
        
        overall_blended_hr = np.mean(np.linalg.norm(blended_oof - train_y, axis=1) <= 0.01)
        
        # Test Reconstruction & Post-correction
        s47_soft_test = np.load(data_dir_s52 / "step47_test_soft.npy")
        s47_argmax_test = np.load(data_dir_s52 / "step47_test_argmax.npy")
        s53_soft_test = np.load(data_dir_s53 / "step52_test.npy")
        
        print("Reconstructing raw Step 55 (SOTA Mamba-Clifford) test predictions...")
        test_tensor = torch.tensor(test_x, dtype=torch.float32)
        s55_test_preds = np.zeros((len(test_x), 3))
        for fold in range(5):
            stats = torch.load(models_dir_s55 / f"stats_fold_{fold}.pt")
            tr_mean = stats["mean"]
            tr_std = stats["std"]
            model_s55 = Sota2026TrajectoryModel(feature_dim=38, latent_dim=64, num_candidates=36).to(device)
            model_s55.load_state_dict(torch.load(models_dir_s55 / f"model_fold_{fold}.pt", map_location=device))
            model_s55.eval()
            with torch.no_grad():
                test_tensor_dev = test_tensor.to(device)
                ft_te, df_te, _, _, _ = extract_features(test_tensor_dev, tr_mean.to(device), tr_std.to(device))
                candidates_te = torch.tensor(test_candidates, dtype=torch.float32).to(device)
                fold_pred, _ = model_s55(ft_te, df_te, candidates_te)
                s55_test_preds += fold_pred.cpu().numpy() / 5.0
                
        # Load active test predictions
        sactive_test_1step = np.load(data_dir_active / "test_preds_cfm_1step.npy")
        sactive_test_2step = np.load(data_dir_active / "test_preds_cfm_2step.npy")
        
        regimes_te, speeds_te, curvature_te, acc_perp_te, acc_par_te = classify_regimes(test_x)
        cruising_te_idx = np.where(regimes_te == 0)[0]
        gliding_te_idx = np.where(regimes_te == 1)[0]
        steering_te_idx = np.where(regimes_te == 2)[0]
        
        blended_test = np.zeros_like(s55_test_preds)
        
        def get_test_comb(idx, w, s47_t, s53_t, act_t):
            s47_te = s47_soft_test if s47_t == "soft" else s47_argmax_test
            s53_te = s53_soft_test
            act_te = sactive_test_1step if act_t == "1step" else sactive_test_2step
            return w[0]*s47_te[idx] + w[1]*s53_te[idx] + w[2]*s55_test_preds[idx] + w[3]*act_te[idx]
            
        blended_test[cruising_te_idx] = get_test_comb(cruising_te_idx, w_cr, s47_t_cr, s53_t_cr, act_t_cr)
        blended_test[gliding_te_idx] = get_test_comb(gliding_te_idx, w_gl, s47_t_gl, s53_t_gl, act_t_gl)
        blended_test[steering_te_idx] = get_test_comb(steering_te_idx, w_st, s47_t_st, s53_t_st, act_t_st)
        
        # RF classifier
        p_last_tr = train_x[:, -1]
        pred_disp_tr = blended_oof - p_last_tr
        pred_disp_norm_tr = np.linalg.norm(pred_disp_tr, axis=1)
        errors_tr = np.linalg.norm(blended_oof - train_y, axis=1)
        is_miss_tr = (errors_tr > 0.01).astype(int)
        
        features_tr = np.column_stack([
            speeds_tr, curvature_tr, acc_perp_tr, acc_par_tr,
            pred_disp_norm_tr, np.abs(blended_oof[:, 2] - p_last_tr[:, 2])
        ])
        
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf_rf.fit(features_tr, is_miss_tr)
        
        p_last_te = test_x[:, -1]
        pred_disp_te = blended_test - p_last_te
        pred_disp_norm_te = np.linalg.norm(pred_disp_te, axis=1)
        
        features_te = np.column_stack([
            speeds_te, curvature_te, acc_perp_te, acc_par_te,
            pred_disp_norm_te, np.abs(blended_test[:, 2] - p_last_te[:, 2])
        ])
        
        prob_miss_te = clf_rf.predict_proba(features_te)[:, 1]
        
        threshold = 0.80
        shrink_factor = 0.93
        corrected_test = blended_test.copy()
        corrected_count = 0
        for idx in range(len(test_x)):
            if prob_miss_te[idx] > threshold and pred_disp_norm_te[idx] > 0.015:
                corrected_test[idx] = p_last_te[idx] + shrink_factor * pred_disp_te[idx]
                corrected_count += 1
                
        sub_path = out_dir / "submission_ultimate_blend_v59.csv"
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, row in zip(test_ids, corrected_test):
                writer.writerow([sample_id, f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"])
                
        result_json = {
            "overall_oof_hr": overall_blended_hr,
            "cruising_hr": hr_cr,
            "gliding_hr": hr_gl,
            "steering_hr": hr_st,
            "w_cr": [float(v) for v in w_cr],
            "w_gl": [float(v) for v in w_gl],
            "w_st": [float(v) for v in w_st],
            "corrected_count": corrected_count
        }
        with open(out_dir / "blend_result.json", "w") as f:
            json.dump(result_json, f, indent=4)
            
        print(f"Step 59 Blending Complete. Blended OOF Hit Rate: {overall_blended_hr*100:.5f}%")
        
    except Exception as e:
        print(f"Step 59 Blending failed: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
