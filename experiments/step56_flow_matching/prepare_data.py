import os
import sys
import shutil
from pathlib import Path

def main():
    print("--- Setting up Step 56 Flow Matching Workspace ---")
    
    src_dir = Path("step55_sota_2026")
    dest_dir = Path("step56_flow_matching")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data directory
    dest_data_dir = dest_dir / "data"
    dest_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy processed datasets from step55
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
            
    print("Data setup complete for Step 56 Flow Matching!")

if __name__ == "__main__":
    main()
