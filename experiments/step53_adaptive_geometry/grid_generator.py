import os
import sys
import numpy as np
import json
from pathlib import Path

EPS = 1e-8

def make_candidates_physical(x: np.ndarray, end_idx: int) -> np.ndarray:
    p0 = x[:, end_idx]
    d1 = x[:, end_idx] - x[:, end_idx - 1]
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    acc = d1 - d2
    
    prev_acc = d2 - (x[:, end_idx - 2] - x[:, end_idx - 3])
    jerk = acc - prev_acc
    
    speeds = np.linalg.norm(d1, axis=1, keepdims=True)
    tangent = d1 / (speeds + EPS)
    acc_par = np.sum(acc * tangent, axis=1, keepdims=True) * tangent
    acc_perp = acc - acc_par
    
    horizon = 2
    v_scale = horizon / 2.0
    acc_scale = (horizon / 2.0) ** 2
    
    # Candidate specs: (d1, par, perp, d2, jerk, time_scale)
    specs = [
        (2.00, 0.00, 0.00, 0.0, 0.0, 1.0),
        (2.00, 0.40, 0.40, 0.0, 0.0, 1.0),
        (2.00, 0.50, 0.50, 0.0, 0.0, 1.0),
        (1.98, 0.56, 0.56, 0.0, 0.0, 1.0),
        (2.00, 0.60, 0.60, 0.0, 0.0, 1.0),
        (1.98, 0.96, -0.08, 0.0, 0.0, 1.0),
        (1.98, 0.90, 0.00, 0.0, 0.0, 1.0),
        (1.98, 1.00, 0.00, 0.0, 0.0, 1.0),
        (2.00, 1.00, -0.10, 0.0, 0.0, 1.0),
        (1.96, 0.90, 0.20, 0.0, 0.0, 1.0),
        (2.02, 0.80, 0.20, 0.0, 0.0, 1.0),
        (1.94, 1.10, -0.20, 0.0, 0.0, 1.0),
        (2.06, 1.00, -0.08, 0.0, 0.0, 1.0),
        (1.90, 1.00, -0.08, 0.0, 0.0, 1.0),
        (1.98, 0.80, -0.05, 0.0, 0.08, 1.0),
        (1.98, 0.80, -0.05, 0.0, -0.08, 1.0),
        (1.98, 0.70, -0.20, 0.0, 0.0, 1.0),
        (1.98, 1.20, -0.20, 0.0, 0.0, 1.0),
        (1.98, 1.20, 0.20, 0.0, 0.0, 1.0),
        (2.08, 1.20, -0.20, 0.0, 0.0, 1.0),
        (1.86, 0.70, 0.20, 0.0, 0.0, 1.0),
        (1.98, 0.96, -0.08, 0.0, 0.0, 0.85),
        (1.98, 0.96, -0.08, 0.0, 0.0, 0.92),
        (1.98, 0.96, -0.08, 0.0, 0.0, 1.08),
        (1.98, 0.96, -0.08, 0.0, 0.0, 1.15),
        (1.98, 1.10, -0.20, 0.0, 0.0, 1.10),
        (1.96, 0.90, 0.20, 0.0, 0.0, 0.90),
    ]
    
    preds = []
    for spec in specs:
        spec_v_scale = v_scale * spec[5]
        spec_acc_scale = acc_scale * (spec[5] ** 2)
        preds.append(
            p0
            + spec[0] * spec_v_scale * d1
            + spec[3] * spec_v_scale * d2
            + spec[1] * spec_acc_scale * acc_par
            + spec[2] * spec_acc_scale * acc_perp
            + spec[4] * spec_acc_scale * jerk
        )
    return np.stack(preds, axis=1).astype(np.float32)

def generate_hybrid_grid(x, ode_preds):
    N = len(x)
    # 1. Physical 27 Grid
    phys_cand = make_candidates_physical(x, 10) # (N, 27, 3)
    
    # 2. Compute Frenet frame at the end of the history
    d1 = x[:, -1] - x[:, -2]
    speeds = np.linalg.norm(d1, axis=1, keepdims=True)
    t = d1 / (speeds + EPS)
    
    n = np.zeros_like(t)
    axis = np.argmin(np.abs(t), axis=1)
    n[np.arange(N), axis] = 1.0
    n = n - np.sum(n * t, axis=1, keepdims=True) * t
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + EPS)
    b = np.cross(t, n, axis=1)
    
    # 8 local micro-offsets
    offsets = [
        (0.005, 0.005), (0.005, -0.005), (-0.005, 0.005), (-0.005, -0.005),
        (0.005, 0.0), (-0.005, 0.0), (0.0, 0.005), (0.0, -0.005)
    ]
    
    ode_offsets = []
    for dn, db in offsets:
        off_pt = ode_preds + dn * n + db * b
        ode_offsets.append(off_pt)
        
    ode_offsets_arr = np.stack(ode_offsets, axis=1) # (N, 8, 3)
    
    # Combine: 27 Physical + 1 Raw ODE + 8 ODE Offsets = 36 total candidates
    hybrid_grid = np.concatenate([phys_cand, ode_preds[:, None, :], ode_offsets_arr], axis=1) # (N, 36, 3)
    return hybrid_grid

def main():
    print("=== Generating Hybrid Grid Dataset ===")
    
    data_dir = Path("step53_adaptive_geometry/data")
    train_x = np.load(data_dir / "train_x.npy")
    train_y = np.load(data_dir / "train_y.npy")
    test_x = np.load(data_dir / "test_x.npy")
    
    s52_oof = np.load(data_dir / "step52_oof.npy")
    s52_test = np.load(data_dir / "step52_test.npy")
    
    print("Building hybrid grid for train split...")
    train_cand = generate_hybrid_grid(train_x, s52_oof)
    np.save(data_dir / "train_candidates.npy", train_cand)
    print(f"Saved train_candidates.npy | Shape: {train_cand.shape}")
    
    print("Building hybrid grid for test split...")
    test_cand = generate_hybrid_grid(test_x, s52_test)
    np.save(data_dir / "test_candidates.npy", test_cand)
    print(f"Saved test_candidates.npy | Shape: {test_cand.shape}")
    
    # Compute classification targets for train split
    dist = np.linalg.norm(train_cand - train_y[:, None, :], axis=2) # (N, 36)
    best_idx = np.argmin(dist, axis=1) # (N,)
    min_dist = np.min(dist, axis=1) # (N,)
    hit_mask = (min_dist <= 0.01).astype(np.int64) # 1 if lockout resolved, else 0
    
    # Save training targets
    np.savez(data_dir / "train_class_labels.npz", best_idx=best_idx, hit_mask=hit_mask)
    print("Saved classification targets train_class_labels.npz")
    
    lockout_count = np.sum(hit_mask == 0)
    print(f"Hybrid Grid Lockout Rate: {lockout_count/len(train_y)*100:.2f}% ({lockout_count} / {len(train_y)})")
    
if __name__ == "__main__":
    main()
