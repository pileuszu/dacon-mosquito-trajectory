import os
import shutil
import json
from pathlib import Path

def main():
    src_dir = Path("step64_spatiotemporal_ai/data")
    dest_dir = Path("step65_spatiotemporal_ai/data")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_copy = [
        "train_x.npy", "train_y.npy", "train_candidates.npy", "train_ids.json",
        "test_x.npy", "test_candidates.npy", "test_ids.json"
    ]
    
    print("Copying data files from Step 64 to Step 65...")
    for f in files_to_copy:
        src_file = src_dir / f
        dest_file = dest_dir / f
        if src_file.exists():
            shutil.copy2(src_file, dest_file)
            print(f"  Copied: {f}")
        else:
            print(f"  Warning: {f} not found in {src_dir}")
            
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
