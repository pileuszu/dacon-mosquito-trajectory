import os
import shutil
import csv
import json
import numpy as np
from pathlib import Path

def parse_submission_csv(csv_path, test_ids):
    # Map from ID to (x, y, z)
    coords_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if not row:
                continue
            sample_id = row[0]
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            coords_map[sample_id] = [x, y, z]
            
    # Align to test_ids order
    test_preds = []
    for tid in test_ids:
        test_preds.append(coords_map[tid])
        
    return np.asarray(test_preds, dtype=np.float32)

def main():
    print("=== Step 51 Data Preparation Started ===")
    
    src_dir = Path("step49_consensus_ensemble/data")
    dest_dir = Path("step51_consensus_blend_v2/data")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy base datasets and step 47 files
    base_files = [
        "train_x.npy", "train_y.npy", "test_x.npy", 
        "train_ids.json", "test_ids.json",
        "step47_oof_soft.npy", "step47_oof_argmax.npy"
    ]
    
    for filename in base_files:
        src = src_dir / filename
        dest = dest_dir / filename
        if src.exists():
            shutil.copy2(src, dest)
            print(f"Copied {filename} to {dest}")
        else:
            print(f"Warning: Source {src} does not exist.")
            
    # Parse Step 47 Test submissions
    s47_soft_csv = Path("outputs/step47_physics_ladder/submission_soft.csv")
    s47_argmax_csv = Path("outputs/step47_physics_ladder/submission_argmax.csv")
    
    with open(dest_dir / "test_ids.json", "r") as f:
        test_ids = json.load(f)
        
    if s47_soft_csv.exists():
        print(f"Parsing Step 47 soft submission: {s47_soft_csv}")
        s47_soft_preds = parse_submission_csv(s47_soft_csv, test_ids)
        np.save(dest_dir / "step47_test_soft.npy", s47_soft_preds)
        print(f"Saved step47_test_soft.npy | Shape: {s47_soft_preds.shape}")
        
    if s47_argmax_csv.exists():
        print(f"Parsing Step 47 argmax submission: {s47_argmax_csv}")
        s47_argmax_preds = parse_submission_csv(s47_argmax_csv, test_ids)
        np.save(dest_dir / "step47_test_argmax.npy", s47_argmax_preds)
        print(f"Saved step47_test_argmax.npy | Shape: {s47_argmax_preds.shape}")
            
    # 2. Copy Step 50 OOF predictions
    s50_oof_src = Path("step50_frenet_ode/oof_predictions.npy")
    s50_oof_dest = dest_dir / "step50_oof.npy"
    if s50_oof_src.exists():
        shutil.copy2(s50_oof_src, s50_oof_dest)
        print(f"Copied step50_oof.npy to {s50_oof_dest}")
    else:
        raise FileNotFoundError(f"Step 50 OOF predictions not found at {s50_oof_src}. Complete training first.")
        
    # 3. Parse Step 50 Test submission CSV
    s50_sub_csv = Path("outputs/step50_frenet_ode/submission_frenet_ode.csv")
    s50_test_dest = dest_dir / "step50_test.npy"
    
    if s50_sub_csv.exists():
        with open(dest_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        print(f"Parsing test submission file: {s50_sub_csv}")
        s50_test_preds = parse_submission_csv(s50_sub_csv, test_ids)
        np.save(s50_test_dest, s50_test_preds)
        print(f"Successfully processed and saved test predictions to {s50_test_dest} | Shape: {s50_test_preds.shape}")
    else:
        raise FileNotFoundError(f"Step 50 Test submission CSV not found at {s50_sub_csv}. Complete inference first.")
        
    print("=== Step 51 Data Preparation Completed Successfully ===")

if __name__ == "__main__":
    main()
