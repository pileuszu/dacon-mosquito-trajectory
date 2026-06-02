import os
import sys
import json
import numpy as np
import lightgbm as lgb
import hashlib
import pickle
import argparse
import traceback
from pathlib import Path
from sklearn.model_selection import KFold
import torch

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification
from model import extract_features

def stable_fold_id(sample_id: str, folds: int) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--fold-limit", type=int, default=None, help="Limit number of folds to train")
    args = parser.parse_args()
    
    send_discord_notification(
        None,
        f"🚀 Started: [Step 53 train_automl.py] Training 5-fold 36-Class LightGBM Grid Selector..."
    )
    
    try:
        data_dir = Path("step53_adaptive_geometry/data")
        models_dir = Path("step53_adaptive_geometry/models")
        models_dir.mkdir(exist_ok=True)
        
        # Load dataset
        X_train = np.load(data_dir / "train_x.npy")
        y_train = np.load(data_dir / "train_y.npy")
        train_candidates = np.load(data_dir / "train_candidates.npy") # (10000, 36, 3)
        
        labels_data = np.load(data_dir / "train_class_labels.npz")
        best_idx = labels_data["best_idx"] # (10000,)
        hit_mask = labels_data["hit_mask"] # (10000,)
        
        with open(data_dir / "train_ids.json", "r") as f:
            train_ids = json.load(f)
            
        print(f"Loaded datasets. X_train: {X_train.shape}, candidates: {train_candidates.shape}")
        
        # Extract 28-dimensional physical features
        print("Extracting 28-dimensional physics features (Z-dynamics & 3D Torsion)...")
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        # Extract features (unscaled first, we will scale per fold to prevent leakage)
        # We pass dummy mean/std to extract raw features
        dummy_mean = torch.zeros(28)
        dummy_std = torch.ones(28)
        features_raw_tensor, _, _, _, _, _, _, _, _, _, _ = extract_features(X_tensor, dummy_mean, dummy_std)
        features_raw = features_raw_tensor.numpy()
        print(f"Features raw shape: {features_raw.shape}")
        
        # Compute stable fold ids
        fold_ids = np.asarray([stable_fold_id(sid, args.folds) for sid in train_ids])
        
        oof_preds_soft = np.zeros_like(y_train)
        oof_preds_argmax = np.zeros_like(y_train)
        
        fold_hr_soft = []
        fold_hr_argmax = []
        
        folds_to_train = args.folds if args.fold_limit is None else args.fold_limit
        
        for fold in range(folds_to_train):
            print(f"\n--- FOLD {fold + 1} / {args.folds} ---")
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            
            X_tr_feat = features_raw[train_mask]
            best_idx_tr = best_idx[train_mask]
            hit_mask_tr = hit_mask[train_mask]
            
            X_val_feat = features_raw[val_mask]
            y_val = y_train[val_mask]
            val_cand = train_candidates[val_mask]
            
            # Apply Normalization scaling using training fold statistics
            tr_mean = X_tr_feat.mean(axis=0)
            tr_std = X_tr_feat.std(axis=0)
            tr_std[tr_std < 1e-6] = 1.0
            
            X_tr_scaled = (X_tr_feat - tr_mean) / tr_std
            X_val_scaled = (X_val_feat - tr_mean) / tr_std
            
            # Save scaling parameters
            stats_path = models_dir / f"stats_fold_{fold}.pkl"
            with open(stats_path, "wb") as f:
                pickle.dump({"mean": tr_mean, "std": tr_std}, f)
                
            # Filter training data to ONLY include resolved hits (hit_mask == 1) to avoid noisy training
            resolved_mask = hit_mask_tr == 1
            X_train_fold = X_tr_scaled[resolved_mask]
            y_train_fold = best_idx_tr[resolved_mask]
            
            print(f"Training on {len(X_train_fold)} resolved samples (excluded {len(resolved_mask) - np.sum(resolved_mask)} lockouts).")
            
            # Build LightGBM Multi-class Classifier
            clf = lgb.LGBMClassifier(
                n_estimators=250,
                learning_rate=0.04,
                num_leaves=31,
                max_depth=6,
                objective="multiclass",
                num_class=36,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            
            clf.fit(X_train_fold, y_train_fold)
            
            # Save model
            model_path = models_dir / f"lgb_model_fold_{fold}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(clf, f)
                
            # Validation Predictions
            probs = clf.predict_proba(X_val_scaled) # (N_val, 36)
            
            # 1. Soft Coordinate Blending: Weighted average of candidates based on class probabilities
            val_preds_soft = np.sum(probs[:, :, None] * val_cand, axis=1) # (N_val, 3)
            
            # 2. Argmax Selection: Choose candidate coordinate with highest probability
            best_val_idx = np.argmax(probs, axis=1)
            val_preds_argmax = val_cand[np.arange(len(X_val_feat)), best_val_idx]
            
            # Evaluate Hit Rate@1cm
            err_soft = np.linalg.norm(val_preds_soft - y_val, axis=1)
            hr_soft = np.mean(err_soft <= 0.01)
            
            err_argmax = np.linalg.norm(val_preds_argmax - y_val, axis=1)
            hr_argmax = np.mean(err_argmax <= 0.01)
            
            print(f"  Val HR@1cm (Soft Blending) : {hr_soft * 100:.3f}%")
            print(f"  Val HR@1cm (Argmax Selection): {hr_argmax * 100:.3f}%")
            
            oof_preds_soft[val_mask] = val_preds_soft
            oof_preds_argmax[val_mask] = val_preds_argmax
            
            fold_hr_soft.append(hr_soft)
            fold_hr_argmax.append(hr_argmax)
            
            if args.folds > 1:
                send_discord_notification(
                    None,
                    f"📢 Fold {fold + 1}/{args.folds} Finished | Soft HR: {hr_soft*100:.3f}% | Argmax HR: {hr_argmax*100:.3f}%"
                )
                
        # Overall OOF metrics
        if args.fold_limit is None:
            overall_soft_err = np.linalg.norm(oof_preds_soft - y_train, axis=1)
            overall_soft_hr = np.mean(overall_soft_err <= 0.01)
            
            overall_argmax_err = np.linalg.norm(oof_preds_argmax - y_train, axis=1)
            overall_argmax_hr = np.mean(overall_argmax_err <= 0.01)
            
            print("\n==========================================")
            print(f"Overall 5-Fold OOF Soft Blended HR  : **{overall_soft_hr * 100:.3f}%**")
            print(f"Overall 5-Fold OOF Argmax Selected HR: **{overall_argmax_hr * 100:.3f}%**")
            print("==========================================")
            
            # Save OOF predictions
            np.save(models_dir / "oof_preds_soft.npy", oof_preds_soft)
            np.save(models_dir / "oof_preds_argmax.npy", oof_preds_argmax)
            
            send_discord_notification(
                None,
                f"✅ Finished: [Step 53 train_automl.py] LightGBM Selector Training Completed!\n"
                f"Overall Soft Blended HR  : **{overall_soft_hr * 100:.3f}%**\n"
                f"Overall Argmax Selected HR: **{overall_argmax_hr * 100:.3f}%**"
            )
        else:
            print("\nDry run finished successfully.")
            send_discord_notification(None, "✅ Finished: [Step 53 train_automl.py] Dry Run Completed successfully!")
            
    except Exception as e:
        error_msg = f"❌ Failed: [Step 53 train_automl.py] Training failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
