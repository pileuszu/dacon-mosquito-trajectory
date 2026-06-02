import os
import sys
import json
import shutil
from pathlib import Path

def main():
    print("--- Setting up Step 54 Differentiable Physics Workspace ---")
    
    src_dir = Path("step53_adaptive_geometry")
    dest_dir = Path("step54_differentiable_physics")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data directory
    dest_data_dir = dest_dir / "data"
    dest_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy processed datasets from step53
    src_data_dir = src_dir / "data"
    files_to_copy = [
        "train_x.npy",
        "train_y.npy",
        "test_x.npy",
        "train_candidates.npy",
        "test_candidates.npy",
        "train_ids.json",
        "test_ids.json",
        "train_class_labels.npz"
    ]
    
    for f in files_to_copy:
        src_path = src_data_dir / f
        dest_path = dest_data_dir / f
        if src_path.exists():
            print(f"Copying {f}...")
            shutil.copy2(src_path, dest_path)
            
    # Also copy step52 predictions to use as priors
    step52_oof = Path("step53_adaptive_geometry/data/step52_oof.npy")
    step52_test = Path("step53_adaptive_geometry/data/step52_test.npy")
    if step52_oof.exists():
        shutil.copy2(step52_oof, dest_data_dir / "step52_oof.npy")
    if step52_test.exists():
        shutil.copy2(step52_test, dest_data_dir / "step52_test.npy")
        
    print("Data setup complete for Step 54!")

if __name__ == "__main__":
    main()
