import os
import sys
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from lightgbm import LGBMClassifier
import warnings

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.outlier_classifier import extract_classifier_features, OutlierDampingClassifier

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

EPS = 1e-8

def classify_4regimes(X):
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
        if speeds[i] <= 0.50:
            if is_steering[i]:
                regimes[i] = 1  # Slow-Turning
            else:
                regimes[i] = 0  # Slow-Straight
        else:
            if is_steering[i]:
                regimes[i] = 3  # Fast-Turning
            else:
                regimes[i] = 2  # Fast-Straight
                
    return regimes, speeds, curvature, acc_perp_norm, acc_par_scalar

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def main():
    print("=== Step 4: Powell Blending Weights Optimization ===")
    
    data_dir = Path("data/processed")
    models_trained_dir = Path("models_trained")
    out_dir = Path("outputs_reproduced")
    out_dir.mkdir(exist_ok=True)
    
    # Load raw train data
    train_x = np.load(data_dir / "train_x.npy")
    train_y = np.load(data_dir / "train_y.npy")
    p_last_tr = train_x[:, -1]
    
    # Define paths to all 21 models with fallbacks to historical experiments folder (experiments/).
    # Note on abbreviations:
    # - 'sNN' prefixes (e.g. s47, s53, s65, etc.) refer to OOF (Out-of-Fold) predictions generated 
    #   during the corresponding historical experiment Step NN (e.g., experiments/step52, experiments/step55, etc.).
    # - 'sf_automl' refers to Super-Feature tabular ranking models.
    # - 'cfm' refers to Clifford-Mamba Continuous Flow Matching predictions.
    # - 'ode' refers to Frenet-guided Neural Ordinary Differential Equation predictions.
    # - 'steer/glide/fast_turn' refer to specialized flight regime models.
    model_paths = [
        # s47_soft_oof (Step 47 hybrid grid search probability predictions)
        models_trained_dir / "s47_oof_soft.npy" if (models_trained_dir / "s47_oof_soft.npy").exists() else Path("experiments/step52_focal_ode/data/step47_oof_soft.npy"),
        # s47_argmax_oof (Step 47 hybrid grid search hard argmax predictions)
        models_trained_dir / "s47_oof_argmax.npy" if (models_trained_dir / "s47_oof_argmax.npy").exists() else Path("experiments/step52_focal_ode/data/step47_oof_argmax.npy"),
        # s53_soft_oof
        models_trained_dir / "s53_oof_preds_soft.npy" if (models_trained_dir / "s53_oof_preds_soft.npy").exists() else Path("experiments/step53_adaptive_geometry/models/oof_preds_soft.npy"),
        # s55_soft_oof
        models_trained_dir / "s55_oof_preds_soft.npy" if (models_trained_dir / "s55_oof_preds_soft.npy").exists() else Path("experiments/step55_sota_2026/data/oof_preds_soft.npy"),
        # s57_cfm_oof
        models_trained_dir / "s57_oof_preds_cfm_1step.npy" if (models_trained_dir / "s57_oof_preds_cfm_1step.npy").exists() else Path("experiments/step57_spatiotemporal_ai/data/oof_preds_cfm_1step.npy"),
        # s62_cfm_oof
        models_trained_dir / "s62_oof_preds_cfm_1step.npy" if (models_trained_dir / "s62_oof_preds_cfm_1step.npy").exists() else Path("experiments/step62_spatiotemporal_ai/data/oof_preds_cfm_1step.npy"),
        # s63_cfm_oof
        models_trained_dir / "s63_oof_preds_cfm_1step.npy" if (models_trained_dir / "s63_oof_preds_cfm_1step.npy").exists() else Path("experiments/step63_spatiotemporal_ai/data/oof_preds_cfm_1step.npy"),
        # s64_cfm_oof
        models_trained_dir / "s64_oof_preds_cfm_1step.npy" if (models_trained_dir / "s64_oof_preds_cfm_1step.npy").exists() else Path("experiments/step64_spatiotemporal_ai/data/oof_preds_cfm_1step.npy"),
        # s65_cfm_oof
        models_trained_dir / "s65_oof_preds_cfm_1step.npy" if (models_trained_dir / "s65_oof_preds_cfm_1step.npy").exists() else Path("experiments/step65_spatiotemporal_ai/data/oof_preds_cfm_1step.npy"),
        # s67_cfm_oof
        models_trained_dir / "s67_oof_preds_cfm_1step.npy" if (models_trained_dir / "s67_oof_preds_cfm_1step.npy").exists() else Path("experiments/step66_super_feature/models/s67/oof_preds_cfm_1step.npy"),
        # s67_v2_cfm_oof (Can be newly trained or loaded from experiments)
        models_trained_dir / "cfm_preds_oof_2step.npy" if (models_trained_dir / "cfm_preds_oof_2step.npy").exists() else Path("experiments/step66_super_feature/models/s67_v2/oof_preds_cfm_1step.npy"),
        # sf_automl_v2
        models_trained_dir / "oof_preds_sf_automl_v2.npy" if (models_trained_dir / "oof_preds_sf_automl_v2.npy").exists() else Path("experiments/step66_super_feature/data/oof_preds_sf_automl_v2.npy"),
        # sf_automl_v3 (Can be newly trained or loaded from experiments)
        models_trained_dir / "ranker_preds_oof.npy" if (models_trained_dir / "ranker_preds_oof.npy").exists() else Path("experiments/step66_super_feature/data/oof_preds_sf_automl_v3.npy"),
        # sf_automl_v3_optimized
        models_trained_dir / "oof_preds_sf_automl_v3_optimized.npy" if (models_trained_dir / "oof_preds_sf_automl_v3_optimized.npy").exists() else Path("experiments/step66_super_feature/data/oof_preds_sf_automl_v3_optimized.npy"),
        # s_steer_cfm_oof
        models_trained_dir / "s67_steering_oof_preds.npy" if (models_trained_dir / "s67_steering_oof_preds.npy").exists() else Path("experiments/step66_super_feature/models/s67_steering/oof_preds_cfm_1step.npy"),
        # s48_neural_ode_oof
        models_trained_dir / "s48_oof.npy" if (models_trained_dir / "s48_oof.npy").exists() else Path("experiments/step48_neural_ode/oof_predictions.npy"),
        # s50_frenet_ode_oof
        models_trained_dir / "s50_oof.npy" if (models_trained_dir / "s50_oof.npy").exists() else Path("experiments/step50_frenet_ode/oof_predictions.npy"),
        # s52_focal_ode_oof (Can be newly trained or loaded from experiments)
        models_trained_dir / "ode_preds_oof.npy" if (models_trained_dir / "ode_preds_oof.npy").exists() else Path("experiments/step52_focal_ode/oof_predictions.npy"),
        # s54_diff_phys_oof
        models_trained_dir / "s54_oof.npy" if (models_trained_dir / "s54_oof.npy").exists() else Path("experiments/step54_differentiable_physics/data/oof_preds_soft.npy"),
        # s_glide_cfm_oof
        models_trained_dir / "s67_gliding_oof_preds.npy" if (models_trained_dir / "s67_gliding_oof_preds.npy").exists() else Path("experiments/step66_super_feature/models/s67_gliding/oof_preds_cfm_1step.npy"),
        # s_fast_turn_cfm_oof
        models_trained_dir / "s67_fast_turning_oof_preds.npy" if (models_trained_dir / "s67_fast_turning_oof_preds.npy").exists() else Path("experiments/step66_super_feature/models/s67_fast_turning/oof_preds_cfm_1step.npy")
    ]
    
    # Load all arrays
    oof_arrays = []
    for i, p in enumerate(model_paths):
        if not p.exists():
            print(f"Warning: model path {p} not found. Using zero arrays as dummy.")
            oof_arrays.append(np.zeros((len(train_x), 3)))
        else:
            oof_arrays.append(np.load(p))
            
    oof_stack = np.stack(oof_arrays, axis=1)  # [10000, 21, 3]
    
    regimes_tr, speeds_tr, curvature_tr, acc_perp_tr, acc_par_tr = classify_4regimes(train_x)
    
    # Dynamic guidance CFM model target
    s67_v2_cfm_oof = oof_arrays[10]
    s_steer_cfm_oof = oof_arrays[14]
    
    guidance = np.zeros_like(s67_v2_cfm_oof)
    for idx in range(len(train_x)):
        r = regimes_tr[idx]
        if r == 0 or r == 2:
            guidance[idx] = s67_v2_cfm_oof[idx]
        elif r == 1 or r == 3:
            guidance[idx] = s_steer_cfm_oof[idx]
            
    # Proportions matching test set distribution
    prop_te_r0 = 0.1514
    prop_te_r1 = 0.2795
    prop_te_r2 = 0.2343
    prop_te_r3 = 0.3348
    
    # Initialize uniform weights
    active_w = {
        0: np.ones(21) / 21.0,
        1: np.ones(21) / 21.0,
        2: np.ones(21) / 21.0,
        3: np.ones(21) / 21.0
    }
    
    turn_mask = (regimes_tr == 1) | (regimes_tr == 3)
    str_mask = (regimes_tr == 0) | (regimes_tr == 2)
    
    best_overall_hr = 0.0
    best_active_w = {r: w.copy() for r, w in active_w.items()}
    
    # Train proxy LightGBM Outlier Classifier for Powell loop
    print("Fitting Outlier LightGBM Classifier...")
    # Calculate simple proxy errors to initialize miss labels
    blended_initial = np.mean(oof_stack, axis=1)
    pred_disp_init = blended_initial - p_last_tr
    pred_disp_norm_init = np.linalg.norm(pred_disp_init, axis=1)
    errors_raw_init = np.linalg.norm(blended_initial - train_y, axis=1)
    is_miss_init = ((errors_raw_init > 0.01) & (pred_disp_norm_init > 0.015)).astype(int)
    
    features_tr = extract_classifier_features(
        train_x, speeds_tr, curvature_tr, acc_perp_tr, acc_par_tr,
        pred_disp_norm_init, blended_initial, oof_stack
    )
    
    clf = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, verbosity=-1, random_state=42)
    clf.fit(features_tr, is_miss_init)
    prob_miss_lgb = clf.predict_proba(features_tr)[:, 1]
    
    # Set default damping parameters
    th_t, shrink_t, gamma_t = 0.78, 0.97, 0.35
    th_s, shrink_s, gamma_s = 0.78, 0.90, 0.35
    
    print("\nRunning Powell optimization iteration loops...")
    for iteration in range(5):
        blended = np.zeros_like(train_y)
        for r in [0, 1, 2, 3]:
            mask = regimes_tr == r
            blended[mask] = np.sum(oof_stack[mask] * active_w[r][None, :, None], axis=1)
            
        pred_disp = blended - p_last_tr
        pred_disp_norm = np.linalg.norm(pred_disp, axis=1)
        
        # Current CV Hit Rate evaluation
        damped = blended.copy()
        for idx in range(len(train_x)):
            if turn_mask[idx]:
                if prob_miss_lgb[idx] > th_t and pred_disp_norm[idx] > 0.015:
                    damped_coord = p_last_tr[idx] + shrink_t * pred_disp[idx]
                    damped[idx] = (1.0 - gamma_t) * damped_coord + gamma_t * guidance[idx]
            elif str_mask[idx]:
                if th_s < 999.0 and prob_miss_lgb[idx] > th_s and pred_disp_norm[idx] > 0.015:
                    damped_coord = p_last_tr[idx] + shrink_s * pred_disp[idx]
                    damped[idx] = (1.0 - gamma_s) * damped_coord + gamma_s * guidance[idx]
                    
        errors_damped = np.linalg.norm(damped - train_y, axis=1)
        w_hr = (prop_te_r0 * np.mean(errors_damped[regimes_tr == 0] <= 0.01) +
                prop_te_r1 * np.mean(errors_damped[regimes_tr == 1] <= 0.01) +
                prop_te_r2 * np.mean(errors_damped[regimes_tr == 2] <= 0.01) +
                prop_te_r3 * np.mean(errors_damped[regimes_tr == 3] <= 0.01)) * 100.0
        
        if w_hr > best_overall_hr:
            best_overall_hr = w_hr
            best_active_w = {r: w.copy() for r, w in active_w.items()}
            
        # Run Powell optimization per regime
        for target_regime in [0, 1, 2, 3]:
            init_logits = np.log(active_w[target_regime] + 1e-6)
            r_mask = regimes_tr == target_regime
            stack_r = oof_stack[r_mask]
            prob_miss_r = prob_miss_lgb[r_mask]
            p_last_r = p_last_tr[r_mask]
            guidance_r = guidance[r_mask]
            
            other_blended = np.zeros_like(train_y)
            for r in [0, 1, 2, 3]:
                if r != target_regime:
                    mask = regimes_tr == r
                    other_blended[mask] = np.sum(oof_stack[mask] * active_w[r][None, :, None], axis=1)
                    
            def objective(logits):
                w_temp = softmax(logits)
                blended_r = np.sum(stack_r * w_temp[None, :, None], axis=1)
                
                pred_disp_r = blended_r - p_last_r
                pred_disp_norm_r = np.linalg.norm(pred_disp_r, axis=1)
                
                if target_regime in [1, 3]:  # Turning
                    damped_coord_r = p_last_r + shrink_t * pred_disp_r
                    damped_target_r = (1.0 - gamma_t) * damped_coord_r + gamma_t * guidance_r
                    damp_mask = (prob_miss_r > th_t) & (pred_disp_norm_r > 0.015)
                    damped_r = np.where(damp_mask[:, None], damped_target_r, blended_r)
                else:  # Straight
                    damped_coord_r = p_last_r + shrink_s * pred_disp_r
                    damped_target_r = (1.0 - gamma_s) * damped_coord_r + gamma_s * guidance_r
                    damp_mask = (prob_miss_r > th_s) & (pred_disp_norm_r > 0.015)
                    damped_r = np.where(damp_mask[:, None], damped_target_r, blended_r)
                        
                damped_full = other_blended.copy()
                damped_full[r_mask] = damped_r
                
                errors_eval = np.linalg.norm(damped_full - train_y, axis=1)
                w_hr_eval = (prop_te_r0 * np.mean(errors_eval[regimes_tr == 0] <= 0.01) +
                             prop_te_r1 * np.mean(errors_eval[regimes_tr == 1] <= 0.01) +
                             prop_te_r2 * np.mean(errors_eval[regimes_tr == 2] <= 0.01) +
                             prop_te_r3 * np.mean(errors_eval[regimes_tr == 3] <= 0.01)) * 100.0
                return -w_hr_eval
                
            res = minimize(objective, init_logits, method='Powell', options={'maxiter': 10, 'disp': False})
            active_w[target_regime] = softmax(res.x)
            
    print(f"Optimal Blending Hit Rate: {best_overall_hr:.5f}%")
    
    # Save optimized parameter json
    output_params = {
        "w_r0": best_active_w[0].tolist(),
        "w_r1": best_active_w[1].tolist(),
        "w_r2": best_active_w[2].tolist(),
        "w_r3": best_active_w[3].tolist(),
        "best_th_turn": th_t,
        "best_shrink_turn": shrink_t,
        "best_gamma_turn": gamma_t,
        "best_th_str": th_s,
        "best_shrink_str": shrink_s,
        "best_gamma_str": gamma_s,
        "best_cv_hr": best_overall_hr
    }
    
    params_out_path = out_dir / "opt_params_21m_method3_tuned_v2.json"
    with open(params_out_path, "w") as f:
        json.dump(output_params, f, indent=4)
    print(f"Powell optimal parameters saved to: {params_out_path}")

if __name__ == "__main__":
    main()
