import shutil
from pathlib import Path

def main():
    src_dir = Path("step47_physics_ladder/data")
    dest_dir = Path("step48_neural_ode/data")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Preparing Step 48 dataset by copying from Step 47...")
    
    files_to_copy = [
        "train_x.npy",
        "train_y.npy",
        "test_x.npy",
        "train_ids.json",
        "test_ids.json"
    ]
    
    for filename in files_to_copy:
        src = src_dir / filename
        dest = dest_dir / filename
        if src.exists():
            print(f"Copying {filename} to {dest_dir}...")
            shutil.copy2(src, dest)
        else:
            print(f"⚠️ Source file not found: {src}")
            
    print("Step 48 data preparation completed.")

if __name__ == "__main__":
    main()
