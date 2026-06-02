import os
import sys
import json
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

def main():
    src_data_dir = Path("step48_neural_ode/data")
    dest_dir = Path("step49_consensus_ensemble/data")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Preparing Step 49 dataset by aggregating Step 47 and Step 48 data...")
    
    # 1. Copy baseline inputs (train_x, train_y, test_x, train_ids, test_ids)
    files_to_copy = [
        "train_x.npy",
        "train_y.npy",
        "test_x.npy",
        "train_ids.json",
        "test_ids.json"
    ]
    for filename in files_to_copy:
        src = src_data_dir / filename
        dest = dest_dir / filename
        if src.exists():
            print(f"Copying {filename} to {dest_dir}...")
            shutil.copy2(src, dest)
        else:
            print(f"⚠️ Source file not found: {src}")
            
    # Load IDs
    with open(dest_dir / "train_ids.json", "r") as f:
        train_ids = json.load(f)
    with open(dest_dir / "test_ids.json", "r") as f:
        test_ids = json.load(f)
        
    print(f"Loaded {len(train_ids)} train ids and {len(test_ids)} test ids.")
    
    # 2. Aggregate Step 47 Validation OOF predictions from all 5 folds
    step47_soft_dict = {}
    step47_argmax_dict = {}
    
    missing_folds = []
    for fold in range(5):
        val_pred_path = Path(f"outputs/step47_physics_ladder/boundary/fold_{fold}/boundary_val_predictions.npz")
        if val_pred_path.exists():
            data = np.load(val_pred_path, allow_pickle=True)
            val_ids = list(data["val_ids"])
            soft_preds = data["soft"]
            argmax_preds = data["argmax"]
            
            for sid, soft, argmax in zip(val_ids, soft_preds, argmax_preds):
                step47_soft_dict[sid] = soft
                step47_argmax_dict[sid] = argmax
        else:
            missing_folds.append(fold + 1)
            
    if missing_folds:
        print(f"⚠️ Missing Step 47 fold boundary validation files: {missing_folds}")
    else:
        print("Successfully read Step 47 OOF predictions from all 5 folds.")
        
    # Align Step 47 OOF arrays with train_ids
    step47_oof_soft = []
    step47_oof_argmax = []
    
    for sid in train_ids:
        if sid in step47_soft_dict:
            step47_oof_soft.append(step47_soft_dict[sid])
            step47_oof_argmax.append(step47_argmax_dict[sid])
        else:
            # Fallback if somehow missing
            step47_oof_soft.append(np.zeros(3))
            step47_oof_argmax.append(np.zeros(3))
            print(f"⚠️ Missing OOF prediction for id: {sid}")
            
    np.save(dest_dir / "step47_oof_soft.npy", np.array(step47_oof_soft))
    np.save(dest_dir / "step47_oof_argmax.npy", np.array(step47_oof_argmax))
    print("Saved aligned Step 47 OOF arrays.")
    
    # 3. Load Step 48 OOF predictions
    step48_oof_path = Path("step48_neural_ode/oof_predictions.npy")
    if step48_oof_path.exists():
        print(f"Copying Step 48 OOF predictions to {dest_dir}...")
        shutil.copy2(step48_oof_path, dest_dir / "step48_oof.npy")
    else:
        print(f"⚠️ Step 48 OOF file not found at {step48_oof_path}!")
        
    # 4. Parse Step 47 and Step 48 Test predictions
    step47_test_csv = Path("outputs/step47_physics_ladder/submission_soft.csv")
    step48_test_csv = Path("outputs/step48_neural_ode/submission_ode.csv")
    
    if step47_test_csv.exists():
        df_s47 = pd.read_csv(step47_test_csv)
        s47_dict = {row["id"]: np.array([row["x"], row["y"], row["z"]]) for _, row in df_s47.iterrows()}
        s47_test = np.array([s47_dict[sid] for sid in test_ids])
        np.save(dest_dir / "step47_test.npy", s47_test)
        print("Parsed and aligned Step 47 test predictions.")
    else:
        print(f"⚠️ Step 47 test csv not found at {step47_test_csv}!")
        
    if step48_test_csv.exists():
        df_s48 = pd.read_csv(step48_test_csv)
        s48_dict = {row["id"]: np.array([row["x"], row["y"], row["z"]]) for _, row in df_s48.iterrows()}
        s48_test = np.array([s48_dict[sid] for sid in test_ids])
        np.save(dest_dir / "step48_test.npy", s48_test)
        print("Parsed and aligned Step 48 test predictions.")
    else:
        print(f"⚠️ Step 48 test csv not found at {step48_test_csv}!")
        
    print("Step 49 data preparation completed.")

if __name__ == "__main__":
    main()
