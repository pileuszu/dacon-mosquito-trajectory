import os
import sys
import shutil
from pathlib import Path

def main():
    print("--- Setting up Step 63 Spatiotemporal AI Workspace ---")
    
    src_dir = Path("step62_spatiotemporal_ai")
    dest_dir = Path("step63_spatiotemporal_ai")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data directory
    dest_data_dir = dest_dir / "data"
    dest_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy processed datasets from step62
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
            
    print("Data setup complete for Step 63 Spatiotemporal AI!")

if __name__ == "__main__":
    main()
