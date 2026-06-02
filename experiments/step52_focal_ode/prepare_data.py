import os
import shutil
from pathlib import Path

def main():
    src_dir = Path("step50_frenet_ode/data")
    dest_dir = Path("step52_focal_ode/data")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    files = ["train_x.npy", "train_y.npy", "test_x.npy", "train_ids.json", "test_ids.json"]
    
    for filename in files:
        src = src_dir / filename
        dest = dest_dir / filename
        if src.exists():
            shutil.copy2(src, dest)
            print(f"Copied {filename} to {dest}")
        else:
            print(f"Warning: Source {src} does not exist.")
            
    # Copy Step 47 OOF files as well
    oof_files = ["step47_oof_soft.npy", "step47_oof_argmax.npy"]
    for filename in oof_files:
        src = Path("step51_consensus_blend_v2/data") / filename
        dest = dest_dir / filename
        if src.exists():
            shutil.copy2(src, dest)
            print(f"Copied {filename} to {dest}")
            
    # Parse Step 47 test submissions
    import json
    import csv
    import numpy as np
    
    def parse_sub_csv(csv_path, test_ids):
        coords = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if not row: continue
                coords[row[0]] = [float(row[1]), float(row[2]), float(row[3])]
        return np.asarray([coords[tid] for tid in test_ids], dtype=np.float32)
        
    with open(dest_dir / "test_ids.json", "r") as f:
        test_ids = json.load(f)
        
    s47_soft_csv = Path("outputs/step47_physics_ladder/submission_soft.csv")
    s47_argmax_csv = Path("outputs/step47_physics_ladder/submission_argmax.csv")
    
    if s47_soft_csv.exists():
        s47_soft_preds = parse_sub_csv(s47_soft_csv, test_ids)
        np.save(dest_dir / "step47_test_soft.npy", s47_soft_preds)
        print("Successfully parsed and saved step47_test_soft.npy")
        
    if s47_argmax_csv.exists():
        s47_argmax_preds = parse_sub_csv(s47_argmax_csv, test_ids)
        np.save(dest_dir / "step47_test_argmax.npy", s47_argmax_preds)
        print("Successfully parsed and saved step47_test_argmax.npy")
            
    print("Step 52 Data Preparation Completed.")

if __name__ == "__main__":
    main()
