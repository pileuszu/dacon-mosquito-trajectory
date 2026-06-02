import os
import sys
import json
import csv
import torch
import numpy as np
import argparse
import traceback
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification
from model import FrenetNeuralODEModel, extract_features

EPS = 1e-8

def classify_regimes(X):
    N = X.shape[0]
    last_v = (X[:, -1] - X[:, -2]) / 0.04  # (N, 3)
    speeds = np.linalg.norm(last_v, axis=1) # (N,)
    
    prev_v = (X[:, -2] - X[:, -3]) / 0.04
    last_a = (last_v - prev_v) / 0.04 # (N, 3)
    
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
            regimes[i] = 2  # Steering
        elif speeds[i] <= 0.50:
            regimes[i] = 0  # Cruising
        else:
            regimes[i] = 1  # Gliding
            
    return regimes, speeds, curvature, acc_perp_norm, acc_par_scalar

def grid_search_blending(s47_preds, s52_preds, train_y, regimes):
    N = len(train_y)
    cruising_idx = np.where(regimes == 0)[0]
    gliding_idx = np.where(regimes == 1)[0]
    steering_idx = np.where(regimes == 2)[0]
    
    weights_space = np.linspace(0.0, 1.0, 21)
    
    # Cruising Optimization
    best_w_cruising = 0.0
    best_hr_cruising = 0.0
    for w in weights_space:
        preds = w * s47_preds[cruising_idx] + (1 - w) * s52_preds[cruising_idx]
        dists = np.linalg.norm(preds - train_y[cruising_idx], axis=1)
        hr = np.mean(dists <= 0.01)
        if hr > best_hr_cruising:
            best_hr_cruising = hr
            best_w_cruising = w
            
    # Gliding Optimization
    best_w_gliding = 0.0
    best_hr_gliding = 0.0
    for w in weights_space:
        preds = w * s47_preds[gliding_idx] + (1 - w) * s52_preds[gliding_idx]
        dists = np.linalg.norm(preds - train_y[gliding_idx], axis=1)
        hr = np.mean(dists <= 0.01)
        if hr > best_hr_gliding:
            best_hr_gliding = hr
            best_w_gliding = w
            
    # Steering Optimization
    best_w_steering = 0.0
    best_hr_steering = 0.0
    for w in weights_space:
        preds = w * s47_preds[steering_idx] + (1 - w) * s52_preds[steering_idx]
        dists = np.linalg.norm(preds - train_y[steering_idx], axis=1)
        hr = np.mean(dists <= 0.01)
        if hr > best_hr_steering:
            best_hr_steering = hr
            best_w_steering = w
            
    final_preds = np.zeros_like(train_y)
    final_preds[cruising_idx] = best_w_cruising * s47_preds[cruising_idx] + (1 - best_w_cruising) * s52_preds[cruising_idx]
    final_preds[gliding_idx] = best_w_gliding * s47_preds[gliding_idx] + (1 - best_w_gliding) * s52_preds[gliding_idx]
    final_preds[steering_idx] = best_w_steering * s47_preds[steering_idx] + (1 - best_w_steering) * s52_preds[steering_idx]
    
    final_dists = np.linalg.norm(final_preds - train_y, axis=1)
    overall_hr = np.mean(final_dists <= 0.01)
    
    best_weights = {
        "w_cruising": float(best_w_cruising),
        "w_gliding": float(best_w_gliding),
        "w_steering": float(best_w_steering),
        "hr_cruising": float(best_hr_cruising),
        "hr_gliding": float(best_hr_gliding),
        "hr_steering": float(best_hr_steering),
        "overall_hr": float(overall_hr)
    }
    
    return best_weights, final_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--out-dir", type=str, default="outputs/step52_focal_ode", help="Output directory")
    args = parser.parse_args()
    
    send_discord_notification(
        None,
        "🚀 Started: [Step 52 inference.py] Generating final test predictions and optimizing blending weights..."
    )
    
    try:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"Using device: {device}")
        
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        data_dir = Path("step52_focal_ode/data")
        
        # ==========================================================
        # 1. RUN TEST-SET INFERENCE OF STEP 52 FOCAL ODE MODEL
        # ==========================================================
        print("\n--- Running Test Set Inference for Step 52 Focal ODE ---")
        test_x = np.load(data_dir / "test_x.npy")
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        test_tensor = torch.tensor(test_x, dtype=torch.float32)
        batch_size = 256
        models_dir = Path("step52_focal_ode/models")
        
        available_folds = []
        for f in range(5):
            model_path = models_dir / f"model_fold_{f}.pt"
            stats_path = models_dir / f"stats_fold_{f}.pt"
            if model_path.exists() and stats_path.exists():
                available_folds.append(f)
                
        if not available_folds:
            raise FileNotFoundError("Trained Focal ODE models not found in step52_focal_ode/models.")
            
        fold_predictions = []
        
        for fold in available_folds:
            print(f"  Fold {fold + 1} Inference...")
            stats = torch.load(models_dir / f"stats_fold_{fold}.pt", map_location=device)
            mean_stats = stats["mean"].to(device)
            std_stats = stats["std"].to(device)
            
            model = FrenetNeuralODEModel(input_dim=27, latent_dim=64).to(device)
            model.load_state_dict(torch.load(models_dir / f"model_fold_{fold}.pt", map_location=device))
            model.eval()
            
            preds_list = []
            with torch.no_grad():
                for i in range(0, len(test_tensor), batch_size):
                    Xb = test_tensor[i:i+batch_size].to(device)
                    ft, df, plt_, tht, _, last_a, _, Rt, spt, _, _ = extract_features(Xb, mean_stats, std_stats)
                    pred = model(ft, df, plt_, tht, spt, Rt, last_a)
                    preds_list.append(pred.cpu().numpy())
                    
            fold_pred = np.concatenate(preds_list, axis=0)
            fold_predictions.append(fold_pred)
            
        s52_test_preds = np.mean(fold_predictions, axis=0)
        np.save(data_dir / "step52_test.npy", s52_test_preds)
        print(f"Saved Step 52 test predictions to step52_test.npy | Shape: {s52_test_preds.shape}")
        
        # ==========================================================
        # 2. OPTIMIZE BLENDING WEIGHTS VIA GRID SEARCH (OOF)
        # ==========================================================
        print("\n--- Optimizing Blending Weights via Grid Search (OOF) ---")
        train_x = np.load(data_dir / "train_x.npy")
        train_y = np.load(data_dir / "train_y.npy")
        
        s47_soft = np.load(data_dir / "step47_oof_soft.npy")
        s47_argmax = np.load(data_dir / "step47_oof_argmax.npy")
        s52_oof = np.load(Path("step52_focal_ode/oof_predictions.npy"))
        
        regimes, speeds, curvature, acc_perp, acc_par = classify_regimes(train_x)
        
        soft_weights, soft_blended_oof = grid_search_blending(s47_soft, s52_oof, train_y, regimes)
        argmax_weights, argmax_blended_oof = grid_search_blending(s47_argmax, s52_oof, train_y, regimes)
        
        if soft_weights["overall_hr"] >= argmax_weights["overall_hr"]:
            best_blend_type = "soft"
            optimal_weights = soft_weights
            optimal_blended_oof = soft_blended_oof
            print(f"🏆 Best OOF Blend: Step 47 (Soft) + Step 52 (Focal ODE)")
        else:
            best_blend_type = "argmax"
            optimal_weights = argmax_weights
            optimal_blended_oof = argmax_blended_oof
            print(f"🏆 Best OOF Blend: Step 47 (Argmax) + Step 52 (Focal ODE)")
            
        optimal_weights["blend_type"] = best_blend_type
        print(f"  - w_cruising (Step 47 weight): {optimal_weights['w_cruising']:.2f}")
        print(f"  - w_gliding (Step 47 weight) : {optimal_weights['w_gliding']:.2f}")
        print(f"  - w_steering (Step 47 weight): {optimal_weights['w_steering']:.2f}")
        print(f"  - Optimal Blended OOF HR@1cm : {optimal_weights['overall_hr']*100:.3f}%")
        
        with open(data_dir / "optimal_weights.json", "w") as f:
            json.dump(optimal_weights, f, indent=4)
            
        # ==========================================================
        # 3. BUILD SMART GUIDED POST-CORRECTION MODEL (OOF)
        # ==========================================================
        print("\n--- Training Smart Post-Correction RF Classifier ---")
        p_last_tr = train_x[:, -1]
        pred_disp_tr = optimal_blended_oof - p_last_tr
        pred_disp_norm_tr = np.linalg.norm(pred_disp_tr, axis=1)
        
        # Miss target: error > 1cm
        errors_tr = np.linalg.norm(optimal_blended_oof - train_y, axis=1)
        is_miss_tr = (errors_tr > 0.01).astype(int)
        
        features_tr = np.column_stack([
            speeds,
            curvature,
            acc_perp,
            acc_par,
            pred_disp_norm_tr,
            np.abs(optimal_blended_oof[:, 2] - p_last_tr[:, 2])
        ])
        
        # Fit final classifier using all OOF features
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(features_tr, is_miss_tr)
        print("Successfully trained Random Forest Miss Classifier.")
        
        # ==========================================================
        # 4. BLEND TEST PREDICTIONS & APPLY GUIDED POST-CORRECTION
        # ==========================================================
        print("\n--- Performing Blending & Post-Correction on Test Set ---")
        if best_blend_type == "soft":
            s47_test = np.load(data_dir / "step47_test_soft.npy")
        else:
            s47_test = np.load(data_dir / "step47_test_argmax.npy")
            
        test_regimes, test_speeds, test_curvature, test_acc_perp, test_acc_par = classify_regimes(test_x)
        
        # 4.1 Blend test coordinates
        test_blended = np.zeros_like(s52_test_preds)
        w_cruising = optimal_weights["w_cruising"]
        w_gliding = optimal_weights["w_gliding"]
        w_steering = optimal_weights["w_steering"]
        
        for i in range(len(test_x)):
            r = test_regimes[i]
            if r == 0:
                test_blended[i] = w_cruising * s47_test[i] + (1 - w_cruising) * s52_test_preds[i]
            elif r == 1:
                test_blended[i] = w_gliding * s47_test[i] + (1 - w_gliding) * s52_test_preds[i]
            else:
                test_blended[i] = w_steering * s47_test[i] + (1 - w_steering) * s52_test_preds[i]
                
        # 4.2 Apply Smart Post-Correction
        p_last_te = test_x[:, -1]
        pred_disp_te = test_blended - p_last_te
        pred_disp_norm_te = np.linalg.norm(pred_disp_te, axis=1)
        
        features_te = np.column_stack([
            test_speeds,
            test_curvature,
            test_acc_perp,
            test_acc_par,
            pred_disp_norm_te,
            np.abs(test_blended[:, 2] - p_last_te[:, 2])
        ])
        
        # Predict miss probability on test data
        prob_miss_te = clf.predict_proba(features_te)[:, 1]
        
        # Optimal thresholds found during offline search
        threshold = 0.80
        shrink_factor = 0.93
        
        corrected_test = test_blended.copy()
        corrected_count = 0
        
        for idx in range(len(test_x)):
            p_miss = prob_miss_te[idx]
            if p_miss > threshold and pred_disp_norm_te[idx] > 0.015:
                corrected_test[idx] = p_last_te[idx] + shrink_factor * pred_disp_te[idx]
                corrected_count += 1
                
        print(f"Applied Post-Correction to {corrected_count} test trajectories ({corrected_count/len(test_x)*100:.2f}%)")
        
        # 5. WRITE FINAL SUBMISSION
        sub_path = out_dir / "submission_focal_ode.csv"
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, row in zip(test_ids, corrected_test):
                writer.writerow([sample_id, f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"])
                
        # Displacement comparison
        disp_orig = np.linalg.norm(test_blended - p_last_te, axis=1).mean() * 100
        disp_corr = np.linalg.norm(corrected_test - p_last_te, axis=1).mean() * 100
        
        success_msg = (
            f"✅ Finished: [Step 52 inference.py] Blended & Smart Post-Corrected predictions successfully generated!\n"
            f"Submission file saved at: `{sub_path}`\n"
            f"  - Overall Blended OOF Hit Rate@1cm: **{optimal_weights['overall_hr']*100:.3f}%**\n"
            f"  - Corrected test cases count: **{corrected_count}개**\n"
            f"  - Blended Avg Displacement: **{disp_orig:.4f} cm**\n"
            f"  - Post-Corrected Avg Displacement: **{disp_corr:.4f} cm**"
        )
        print(success_msg)
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 52 inference.py] Blended inference failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
