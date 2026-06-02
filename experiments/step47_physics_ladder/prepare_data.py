import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv

def read_xyz_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: float(r["timestep_ms"]))
        return np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float32)

def read_labels(path: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    xyz: list[list[float]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id"])
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return ids, np.asarray(xyz, dtype=np.float32)

def read_submission_ids(path: Path) -> list[str]:
    with path.open("r", newline="") as f:
        return [row["id"] for row in csv.DictReader(f)]

def load_stack(folder: Path, ids: list[str]) -> np.ndarray:
    return np.stack([read_xyz_csv(folder / f"{sample_id}.csv") for sample_id in tqdm(ids, desc=f"Loading from {folder.name}")], axis=0).astype(np.float32)

def main():
    root = Path("data/open")
    out_dir = Path("step47_physics_ladder/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Preparing step 47 Physics Ladder pre-compiled dataset...")
    
    # 1. Train
    train_ids, train_y = read_labels(root / "train_labels.csv")
    train_x = load_stack(root / "train", train_ids)
    
    # 2. Test
    test_ids = read_submission_ids(root / "sample_submission.csv")
    test_x = load_stack(root / "test", test_ids)
    
    # Save to out_dir
    print("Saving pre-compiled dataset to step47_physics_ladder/data/...")
    np.save(out_dir / "train_x.npy", train_x)
    np.save(out_dir / "train_y.npy", train_y)
    np.save(out_dir / "test_x.npy", test_x)
    
    # Save ids as json
    import json
    with open(out_dir / "train_ids.json", "w") as f:
        json.dump(train_ids, f)
    with open(out_dir / "test_ids.json", "w") as f:
        json.dump(test_ids, f)
        
    print("Done preparing data.")

if __name__ == "__main__":
    main()
