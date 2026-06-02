import os
import sys
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.outlier_classifier import extract_classifier_features
from src.powell_optimization import classify_4regimes

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    print("=== Step 5: Final Submission Inference ===")
    
    data_dir = Path("data/processed")
    models_trained_dir = Path("models_trained")
    opt_dir = Path("outputs_reproduced")
    
    # Load test data
    test_x = np.load(data_dir / "test_x.npy")
    with open(data_dir / "test_ids.json", "r") as f:
        test_ids = json.load(f)
    p_last_te = test_x[:, -1]
    
    # Predict regimes
    regimes_te, speeds_te, curvature_te, acc_perp_te, acc_par_te = classify_4regimes(test_x)
    
    # Load stacked predictions of 16 historical models (generated in experiments/step66_super_feature)
    test_stack_16m = np.load("experiments/step66_super_feature/data/test_preds_stack_16m.npy")
    
    # Prune indices 3 and 12 (outdated baseline models) to avoid redundancy, keeping 14 models.
    # The remaining models include Clifford-Mamba CFM, Neural ODEs, and tabular ranking models.
    indices_to_keep = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]  # size 14
    test_stack_pruned = test_stack_16m[:, indices_to_keep, :]
    
    # Load Clifford-Mamba CFM predictions specialized for Steering regimes (turning flight)
    s_steer_cfm_te = np.load("experiments/step66_super_feature/models/s67_steering/test_preds_cfm_1step.npy")
    is_steering_te = (regimes_te == 1) | (regimes_te == 3)
    s_steer_cfm_te[~is_steering_te] = p_last_te[~is_steering_te]
    
    # Load 4 SOTA Ordinary Differential Equation (ODE) and differentiable physics predictions:
    # - s48: Step 48 Frenet Neural ODE
    # - s50: Step 50 Frenet ODE with curvature constraints
    # - s52: Step 52 Focal-loss Frenet Neural ODE (using newly trained version if available)
    # - s54: Step 54 differentiable kinematics physics model
    s48_neural_ode_te = np.load("experiments/step49_consensus_ensemble/data/step48_test.npy")
    s50_frenet_ode_te = np.load("experiments/step51_consensus_blend_v2/data/step50_test.npy")
    
    # Check if newly trained ODE model output is available
    if (models_trained_dir / "ode_preds_test.npy").exists():
        s52_focal_ode_te = np.load(models_trained_dir / "ode_preds_test.npy")
    else:
        s52_focal_ode_te = np.load("experiments/step52_focal_ode/data/step52_test.npy")
        
    s54_diff_phys_te = np.load("experiments/step54_differentiable_physics/data/test_preds_soft.npy")
    
    # Load 2 Expert Clifford-Mamba CFM predictions specialized for Gliding and Fast-Turning regimes
    s_glide_cfm_te = np.load("experiments/step66_super_feature/models/s67_gliding/test_preds_cfm_1step.npy")
    s_fast_turn_cfm_te = np.load("experiments/step66_super_feature/models/s67_fast_turning/test_preds_cfm_1step.npy")
    
    # Check for newly trained CFM v2 and ranker models
    if (models_trained_dir / "cfm_preds_test_2step.npy").exists():
        s67_v2_te = np.load(models_trained_dir / "cfm_preds_test_2step.npy")
    else:
        s67_v2_te = np.load("experiments/step66_super_feature/models/s67_v2/test_preds_cfm_1step.npy")
        
    if (models_trained_dir / "ranker_preds_test.npy").exists():
        sf_automl_v3 = np.load(models_trained_dir / "ranker_preds_test.npy")
    else:
        sf_automl_v3 = np.load("experiments/step66_super_feature/data/test_preds_sf_automl_v3.npy")
        
    # Re-insert newly trained models into the stack where appropriate
    test_stack_pruned[:, 10] = s67_v2_te  # index 10 corresponds to s67_v2
    test_stack_pruned[:, 12] = sf_automl_v3  # index 12 corresponds to sf_automl_v3
    
    # Concatenate 21 test predictions
    test_stack = np.concatenate([
        test_stack_pruned,
        s_steer_cfm_te[:, None, :],
        s48_neural_ode_te[:, None, :],
        s50_frenet_ode_te[:, None, :],
        s52_focal_ode_te[:, None, :],
        s54_diff_phys_te[:, None, :],
        s_glide_cfm_te[:, None, :],
        s_fast_turn_cfm_te[:, None, :]
    ], axis=1)  # [10000, 21, 3]
    
    # Load optimized blending weights
    params_path = opt_dir / "opt_params_21m_method3_tuned_v2.json"
    if not params_path.exists():
        # Fallback to pre-trained optimal parameters
        params_path = Path("experiments/step66_super_feature/opt_params_21m_method3_tuned_v2.json")
        
    with open(params_path, "r") as f:
        opt = json.load(f)
        
    w_r0 = np.array(opt["w_r0"])
    w_r1 = np.array(opt["w_r1"])
    w_r2 = np.array(opt["w_r2"])
    w_r3 = np.array(opt["w_r3"])
    
    th_t = opt["best_th_turn"]
    shrink_t = opt["best_shrink_turn"]
    gamma_t = opt["best_gamma_turn"]
    th_s = opt["best_th_str"]
    shrink_s = opt["best_shrink_str"]
    gamma_s = opt["best_gamma_str"]
    
    # Blend test predictions
    blended_test = np.zeros((len(test_x), 3))
    blended_test[regimes_te == 0] = np.sum(test_stack[regimes_te == 0] * w_r0[None, :, None], axis=1)
    blended_test[regimes_te == 1] = np.sum(test_stack[regimes_te == 1] * w_r1[None, :, None], axis=1)
    blended_test[regimes_te == 2] = np.sum(test_stack[regimes_te == 2] * w_r2[None, :, None], axis=1)
    blended_test[regimes_te == 3] = np.sum(test_stack[regimes_te == 3] * w_r3[None, :, None], axis=1)
    
    # Predict miss probabilities using Outlier Classifier
    pred_disp_te = blended_test - p_last_te
    pred_disp_norm_te = np.linalg.norm(pred_disp_te, axis=1)
    
    # Load Outlier Classifier
    clf_path = models_trained_dir / "outlier_damping_classifier.pkl"
    if clf_path.exists():
        clf = joblib.load(clf_path)
        features_te = extract_classifier_features(
            test_x, speeds_te, curvature_te, acc_perp_te, acc_par_te,
            pred_disp_norm_te, blended_test, test_stack
        )
        prob_miss_te = clf.predict_proba(features_te)
    else:
        print("Warning: Trained outlier classifier not found. Skipping outlier damping.")
        prob_miss_te = np.zeros(len(test_x))
        
    # Build guidance model based on regimes
    guidance_test = np.zeros_like(blended_test)
    for idx in range(len(test_x)):
        r = regimes_te[idx]
        if r == 0:
            guidance_test[idx] = s67_v2_te[idx]
        elif r == 1:
            guidance_test[idx] = s_steer_cfm_te[idx]
        elif r == 2:
            guidance_test[idx] = s_glide_cfm_te[idx]
        elif r == 3:
            guidance_test[idx] = s_fast_turn_cfm_te[idx]
            
    # Apply Damping parameters
    submission_coords = blended_test.copy()
    damped_count_turn = 0
    damped_count_str = 0
    
    turn_mask_te = (regimes_te == 1) | (regimes_te == 3)
    str_mask_te = (regimes_te == 0) | (regimes_te == 2)
    
    for idx in range(len(test_x)):
        if turn_mask_te[idx]:
            if prob_miss_te[idx] > th_t and pred_disp_norm_te[idx] > 0.015:
                damped_coord = p_last_te[idx] + shrink_t * pred_disp_te[idx]
                submission_coords[idx] = (1.0 - gamma_t) * damped_coord + gamma_t * guidance_test[idx]
                damped_count_turn += 1
        elif str_mask_te[idx]:
            if th_s < 999.0 and prob_miss_te[idx] > th_s and pred_disp_norm_te[idx] > 0.015:
                damped_coord = p_last_te[idx] + shrink_s * pred_disp_te[idx]
                submission_coords[idx] = (1.0 - gamma_s) * damped_coord + gamma_s * guidance_test[idx]
                damped_count_str += 1
                
    # Save final submission.csv in workspace root
    sub_path = Path("submission.csv")
    print(f"Saving final ensembled submission to: {sub_path.absolute()}")
    with sub_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "z"])
        for sample_id, coord in zip(test_ids, submission_coords):
            writer.writerow([sample_id, f"{coord[0]:.9f}", f"{coord[1]:.9f}", f"{coord[2]:.9f}"])
            
    # Print metrics
    diffs = np.linalg.norm(submission_coords - p_last_te, axis=1)
    print(f"\n--- Submission Stats ---")
    print(f"Turn Damped: {damped_count_turn} | Straight Damped: {damped_count_str}")
    print(f"Average Displacement: {diffs.mean()*100:.4f} cm")
    print(f"Maximum Displacement: {diffs.max()*100:.4f} cm")

if __name__ == "__main__":
    main()
