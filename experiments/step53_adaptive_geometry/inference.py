import os
import sys
import json
import csv
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import torch

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification
from model import extract_features

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

def main():
    send_discord_notification(
        None,
        "🚀 Started: [Step 53 inference.py] Generating final hybrid grid selector predictions..."
    )
    
    try:
        data_dir = Path("step53_adaptive_geometry/data")
        models_dir = Path("step53_adaptive_geometry/models")
        out_dir = Path("outputs/step53_adaptive_geometry")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load test data and candidates
        test_x = np.load(data_dir / "test_x.npy")
        test_candidates = np.load(data_dir / "test_candidates.npy") # (10000, 36, 3)
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        print(f"Loaded test input. Shape: {test_x.shape}, candidates: {test_candidates.shape}")
        
        # 2. Extract 28-dimensional physical features for test data
        print("Extracting 28-dimensional physics features...")
        test_tensor = torch.tensor(test_x, dtype=torch.float32)
        dummy_mean = torch.zeros(28)
        dummy_std = torch.ones(28)
        features_raw_tensor, _, _, _, _, _, _, _, _, _, _ = extract_features(test_tensor, dummy_mean, dummy_std)
        test_features_raw = features_raw_tensor.numpy()
        
        # 3. Perform 5-fold ensemble predictions
        print("Running 5-fold ensemble inference...")
        ensemble_probs = np.zeros((len(test_x), 36))
        
        for fold in range(5):
            print(f"  Predicting with Fold {fold + 1}...")
            # Load normalization stats
            with open(models_dir / f"stats_fold_{fold}.pkl", "rb") as f:
                stats = pickle.load(f)
            tr_mean = stats["mean"]
            tr_std = stats["std"]
            
            # Load LightGBM model
            with open(models_dir / f"lgb_model_fold_{fold}.pkl", "rb") as f:
                clf = pickle.load(f)
                
            test_scaled = (test_features_raw - tr_mean) / tr_std
            fold_probs = clf.predict_proba(test_scaled) # (10000, 36)
            ensemble_probs += fold_probs / 5.0
            
        # 4. Soft Blending test coordinates
        print("Performing Soft coordinate blending...")
        test_preds = np.sum(ensemble_probs[:, :, None] * test_candidates, axis=1) # (10000, 3)
        
        # ==========================================================
        # 5. SMART GUIDED POST-CORRECTION (OOF RF CLASSIFIER)
        # ==========================================================
        print("\n--- Training Smart Post-Correction RF Classifier ---")
        train_x = np.load(data_dir / "train_x.npy")
        train_y = np.load(data_dir / "train_y.npy")
        oof_preds = np.load(models_dir / "oof_preds_soft.npy")
        
        # Calculate OOF classification targets
        p_last_tr = train_x[:, -1]
        pred_disp_tr = oof_preds - p_last_tr
        pred_disp_norm_tr = np.linalg.norm(pred_disp_tr, axis=1)
        
        errors_tr = np.linalg.norm(oof_preds - train_y, axis=1)
        is_miss_tr = (errors_tr > 0.01).astype(int)
        
        # Extract features for RF training
        regimes_tr, speeds_tr, curvature_tr, acc_perp_tr, acc_par_tr = classify_regimes(train_x)
        features_tr = np.column_stack([
            speeds_tr,
            curvature_tr,
            acc_perp_tr,
            acc_par_tr,
            pred_disp_norm_tr,
            np.abs(oof_preds[:, 2] - p_last_tr[:, 2])
        ])
        
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf_rf.fit(features_tr, is_miss_tr)
        
        # 5.2 Apply targeted post-correction on test set
        print("Applying post-correction on test set predictions...")
        test_regimes, test_speeds, test_curvature, test_acc_perp, test_acc_par = classify_regimes(test_x)
        p_last_te = test_x[:, -1]
        pred_disp_te = test_preds - p_last_te
        pred_disp_norm_te = np.linalg.norm(pred_disp_te, axis=1)
        
        features_te = np.column_stack([
            test_speeds,
            test_curvature,
            test_acc_perp,
            test_acc_par,
            pred_disp_norm_te,
            np.abs(test_preds[:, 2] - p_last_te[:, 2])
        ])
        
        prob_miss_te = clf_rf.predict_proba(features_te)[:, 1]
        
        # Best offline parameters
        threshold = 0.80
        shrink_factor = 0.93
        
        corrected_test = test_preds.copy()
        corrected_count = 0
        
        for idx in range(len(test_x)):
            p_miss = prob_miss_te[idx]
            if p_miss > threshold and pred_disp_norm_te[idx] > 0.015:
                corrected_test[idx] = p_last_te[idx] + shrink_factor * pred_disp_te[idx]
                corrected_count += 1
                
        print(f"Applied Post-Correction to {corrected_count} test trajectories ({corrected_count/len(test_x)*100:.2f}%)")
        
        # 6. WRITE FINAL SUBMISSION
        sub_path = out_dir / "submission_adaptive.csv"
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, row in zip(test_ids, corrected_test):
                writer.writerow([sample_id, f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"])
                
        # Calculate statistics
        disp_orig = np.linalg.norm(test_preds - p_last_te, axis=1).mean() * 100
        disp_corr = np.linalg.norm(corrected_test - p_last_te, axis=1).mean() * 100
        
        success_msg = (
            f"✅ Finished: [Step 53 inference.py] Adaptive hybrid selector predictions generated successfully!\n"
            f"Submission file saved at: `{sub_path}`\n"
            f"  - Corrected test cases count: **{corrected_count}개**\n"
            f"  - Original Blended Avg Displacement: **{disp_orig:.4f} cm**\n"
            f"  - Post-Corrected Avg Displacement: **{disp_corr:.4f} cm**"
        )
        print(success_msg)
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 53 inference.py] Adaptive inference failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
