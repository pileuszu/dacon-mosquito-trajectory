import os
import shutil
from pathlib import Path

def main():
    src_dir = Path("step52_focal_ode/data")
    dest_dir = Path("step53_adaptive_geometry/data")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy base inputs
    base_files = ["train_x.npy", "train_y.npy", "test_x.npy", "train_ids.json", "test_ids.json"]
    for filename in base_files:
        src = src_dir / filename
        dest = dest_dir / filename
        if src.exists():
            shutil.copy2(src, dest)
            print(f"Copied {filename} to {dest}")
            
    # 2. Copy Step 52 Focal ODE OOF predictions
    s52_oof_src = Path("step52_focal_ode/oof_predictions.npy")
    s52_oof_dest = dest_dir / "step52_oof.npy"
    if s52_oof_src.exists():
        shutil.copy2(s52_oof_src, s52_oof_dest)
        print(f"Copied step52_oof.npy to {s52_oof_dest}")
        
    # 3. Copy Step 52 Test predictions
    s52_test_src = Path("step52_focal_ode/data/step52_test.npy")
    s52_test_dest = dest_dir / "step52_test.npy"
    if s52_test_src.exists():
        shutil.copy2(s52_test_src, s52_test_dest)
        print(f"Copied step52_test.npy to {s52_test_dest}")
    else:
        # Fallback to copy from top level if needed
        alt_src = Path("step52_focal_ode/step52_test.npy")
        if alt_src.exists():
            shutil.copy2(alt_src, s52_test_dest)
            print(f"Copied step52_test.npy to {s52_test_dest}")
            
    print("Step 53 Data Preparation Completed.")

if __name__ == "__main__":
    main()
