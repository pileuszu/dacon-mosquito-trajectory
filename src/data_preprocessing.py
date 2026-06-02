import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EPS = 1e-8

def read_xyz_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: float(r["timestep_ms"]))
        return np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float32)

def read_labels(path: Path) -> tuple[list[str], np.ndarray]:
    ids = []
    xyz = []
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

def extract_super_features(X):
    """
    Extracts 180+ dimensional biomechanical and geometric features from Cartesian trajectory history.
    X: numpy array of shape [N, seq_len, 3] (e.g., [N, 11, 3] coordinates)
    Returns: numpy array of shape [N, feature_dim]
    """
    N = X.shape[0]
    seq_len = X.shape[1]
    features_list = []
    
    # 1. Coordinate differences & Derivatives
    v = (X[:, 1:] - X[:, :-1]) / 0.04  # [N, seq_len-1, 3]
    a = (v[:, 1:] - v[:, :-1]) / 0.04  # [N, seq_len-2, 3]
    j = (a[:, 1:] - a[:, :-1]) / 0.04  # [N, seq_len-3, 3]
    s = (j[:, 1:] - j[:, :-1]) / 0.04  # [N, seq_len-4, 3]
    
    # Norms
    speeds = np.linalg.norm(v, axis=2)  
    accs = np.linalg.norm(a, axis=2)    
    jerks = np.linalg.norm(j, axis=2)   
    snaps = np.linalg.norm(s, axis=2)   
    
    # 2. Multi-Scale Sliding Windows (w2, w4, w8)
    features_list.extend([
        np.mean(speeds[:, -2:], axis=1), np.std(speeds[:, -2:], axis=1), np.max(speeds[:, -2:], axis=1),
        np.mean(accs[:, -2:], axis=1), np.std(accs[:, -2:], axis=1), np.max(accs[:, -2:], axis=1)
    ])
    
    features_list.extend([
        np.mean(speeds[:, -4:], axis=1), np.std(speeds[:, -4:], axis=1), np.max(speeds[:, -4:], axis=1), np.min(speeds[:, -4:], axis=1),
        np.mean(accs[:, -4:], axis=1), np.std(accs[:, -4:], axis=1), np.max(accs[:, -4:], axis=1), np.min(accs[:, -4:], axis=1),
        np.mean(jerks[:, -4:], axis=1), np.std(jerks[:, -4:], axis=1), np.max(jerks[:, -4:], axis=1), np.min(jerks[:, -4:], axis=1)
    ])
    
    features_list.extend([
        np.mean(speeds, axis=1), np.std(speeds, axis=1), np.max(speeds, axis=1), np.min(speeds, axis=1),
        np.mean(accs, axis=1), np.std(accs, axis=1), np.max(accs, axis=1), np.min(accs, axis=1),
        np.mean(jerks, axis=1), np.std(jerks, axis=1), np.max(jerks, axis=1), np.min(jerks, axis=1),
        np.mean(snaps, axis=1), np.std(snaps, axis=1), np.max(snaps, axis=1), np.min(snaps, axis=1)
    ])
    
    for t_idx in range(speeds.shape[1]):
        features_list.append(speeds[:, t_idx])
    for t_idx in range(accs.shape[1]):
        features_list.append(accs[:, t_idx])
    for t_idx in range(jerks.shape[1]):
        features_list.append(jerks[:, t_idx])
    for t_idx in range(snaps.shape[1]):
        features_list.append(snaps[:, t_idx])
        
    # 3. High-Order Derivatives on Frenet Frame (last frame)
    v_last = v[:, -1]
    a_last = a[:, -1]
    j_last = j[:, -1]
    s_last = s[:, -1]
    
    speed_last = speeds[:, -1]
    
    T = v_last / (speed_last[:, None] + EPS)
    
    a_par_scalar = np.sum(a_last * T, axis=1)
    a_perp = a_last - a_par_scalar[:, None] * T
    a_perp_norm = np.linalg.norm(a_perp, axis=1)
    
    N = a_perp / (a_perp_norm[:, None] + EPS)
    fallback = np.zeros_like(N)
    axis = np.argmin(np.abs(T), axis=1)
    fallback[np.arange(N.shape[0]), axis] = 1.0
    fallback = fallback - np.sum(fallback * T, axis=1)[:, None] * T
    fallback = fallback / (np.linalg.norm(fallback, axis=1)[:, None] + EPS)
    N = np.where(a_perp_norm[:, None] > 1e-6, N, fallback)
    
    B = np.cross(T, N, axis=1)
    B = B / (np.linalg.norm(B, axis=1)[:, None] + EPS)
    
    a_T = a_par_scalar
    a_N = np.sum(a_last * N, axis=1)
    a_B = np.sum(a_last * B, axis=1)
    
    j_T = np.sum(j_last * T, axis=1)
    j_N = np.sum(j_last * N, axis=1)
    j_B = np.sum(j_last * B, axis=1)
    
    s_T = np.sum(s_last * T, axis=1)
    s_N = np.sum(s_last * N, axis=1)
    s_B = np.sum(s_last * B, axis=1)
    
    features_list.extend([
        a_T, a_N, a_B, a_perp_norm,
        j_T, j_N, j_B, np.linalg.norm(j_last, axis=1),
        s_T, s_N, s_B, np.linalg.norm(s_last, axis=1)
    ])
    
    # 4. Aerodynamics & Inertial Biomechanics
    f_rayleigh = speed_last[:, None]**2 * v_last
    f_rayleigh_norm = np.linalg.norm(f_rayleigh, axis=1)
    f_linear = speed_last[:, None] * v_last
    f_linear_norm = np.linalg.norm(f_linear, axis=1)
    
    ke_last = speed_last**2
    ke_prev = speeds[:, -2]**2
    d_ke = ke_last - ke_prev
    
    cross_prod = np.cross(v_last, a_last, axis=1)
    cross_norm = np.linalg.norm(cross_prod, axis=1)
    curv_last = cross_norm / (speed_last**3 + EPS)
    
    acc_centripetal = curv_last * (speed_last**2)
    
    features_list.extend([
        f_rayleigh[:, 0], f_rayleigh[:, 1], f_rayleigh[:, 2], f_rayleigh_norm,
        f_linear[:, 0], f_linear[:, 1], f_linear[:, 2], f_linear_norm,
        ke_last, d_ke, curv_last, acc_centripetal
    ])
    
    # 5. Curvature & Torsion Profile
    curv_seq = []
    torsion_seq = []
    
    for t_idx in range(2, v.shape[1] - 1):
        vt = v[:, t_idx]
        at = a[:, t_idx-1]
        cp = np.cross(vt, at, axis=1)
        cp_norm = np.linalg.norm(cp, axis=1)
        curv_t = cp_norm / (np.linalg.norm(vt, axis=1)**3 + EPS)
        curv_seq.append(curv_t)
        features_list.append(curv_t)
        
    for t_idx in range(3, v.shape[1] - 1):
        vt = v[:, t_idx]
        at = a[:, t_idx-1]
        jt = j[:, t_idx-2]
        cp = np.cross(vt, at, axis=1)
        cp_norm_sq = np.sum(cp**2, axis=1)
        torsion_t = np.sum(cp * jt, axis=1) / (cp_norm_sq + EPS)
        torsion_seq.append(torsion_t)
        features_list.append(torsion_t)
        
    curv_seq = np.stack(curv_seq, axis=1)  
    torsion_seq = np.stack(torsion_seq, axis=1)  
    
    d_curv = curv_last - curv_seq[:, -1]
    features_list.extend([
        np.mean(curv_seq, axis=1), np.std(curv_seq, axis=1), np.max(curv_seq, axis=1), d_curv,
        np.mean(torsion_seq, axis=1), np.std(torsion_seq, axis=1), np.max(torsion_seq, axis=1)
    ])
    
    # 6. Center of Mass Relative Coordinates
    weights = np.arange(1, seq_len + 1, dtype=float)
    weights /= np.sum(weights)
    p_com = np.sum(X * weights[None, :, None], axis=1)  
    
    p_last = X[:, -1]
    r_com = p_last - p_com  
    r_com_norm = np.linalg.norm(r_com, axis=1)
    
    theta_com = np.arctan2(r_com[:, 1], r_com[:, 0])
    phi_com = np.arccos(np.clip(r_com[:, 2] / (r_com_norm + EPS), -1.0, 1.0))
    
    v_com = v_last - np.mean(v, axis=1)
    a_com = a_last - np.mean(a, axis=1)
    
    features_list.extend([
        r_com[:, 0], r_com[:, 1], r_com[:, 2], r_com_norm,
        theta_com, phi_com,
        v_com[:, 0], v_com[:, 1], v_com[:, 2], np.linalg.norm(v_com, axis=1),
        a_com[:, 0], a_com[:, 1], a_com[:, 2], np.linalg.norm(a_com, axis=1)
    ])
    
    # 7. Volatility, Saccade & Turning Indicators
    volatility = np.std(speeds, axis=1) / (np.mean(speeds, axis=1) + EPS)
    is_steering_soft = curv_last * a_perp_norm
    
    features_list.extend([
        volatility, is_steering_soft
    ])
    
    features_mat = np.column_stack(features_list)
    return features_mat

def classify_4regimes(X):
    """
    Classify into 4 flight regimes:
    0: Slow-Straight
    1: Slow-Turning
    2: Fast-Straight
    3: Fast-Turning
    """
    N = X.shape[0]
    last_v = (X[:, -1] - X[:, -2]) / 0.04
    speeds = np.linalg.norm(last_v, axis=1)
    prev_v = (X[:, -2] - X[:, -3]) / 0.04
    last_a = (last_v - prev_v) / 0.04
    t_dir = last_v / (speeds[:, None] + EPS)
    acc_par_scalar = np.sum(last_a * t_dir, axis=1)
    acc_perp = last_a - acc_par_scalar[:, None] * t_dir
    acc_perp_norm = np.linalg.norm(acc_perp, axis=1)
    cross_prod = np.cross(last_v, last_a, axis=1)
    cross_norm = np.linalg.norm(cross_prod, axis=1)
    curvature = cross_norm / (speeds ** 3 + EPS)
    is_steering = (curvature > 6.0) | (acc_perp_norm > 1.8)
    
    regimes = np.zeros(N, dtype=int)
    for i in range(N):
        if speeds[i] <= 0.50:
            if is_steering[i]:
                regimes[i] = 1
            else:
                regimes[i] = 0
        else:
            if is_steering[i]:
                regimes[i] = 3
            else:
                regimes[i] = 2
                
    return regimes, speeds, curvature, acc_perp_norm, acc_par_scalar

def main():
    root = Path("data/open")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Step 1: Preprocessing Raw Dataset ===")
    
    # Check if raw files exist
    if not (root / "train_labels.csv").exists():
        print(f"Error: {root / 'train_labels.csv'} not found! Make sure data/open/ directory is populated.")
        sys.exit(1)
        
    # Load raw labels and build numpy stacks
    print("Loading train_labels.csv...")
    train_ids, train_y = read_labels(root / "train_labels.csv")
    print("Loading train trajectory files...")
    train_x = load_stack(root / "train", train_ids)
    
    print("Loading sample_submission.csv...")
    test_ids = read_submission_ids(root / "sample_submission.csv")
    print("Loading test trajectory files...")
    test_x = load_stack(root / "test", test_ids)
    
    print(f"Train shapes: x={train_x.shape}, y={train_y.shape}")
    print(f"Test shape: x={test_x.shape}")
    
    # Save base arrays
    np.save(out_dir / "train_x.npy", train_x)
    np.save(out_dir / "train_y.npy", train_y)
    np.save(out_dir / "test_x.npy", test_x)
    
    with open(out_dir / "train_ids.json", "w") as f:
        json.dump(train_ids, f)
    with open(out_dir / "test_ids.json", "w") as f:
        json.dump(test_ids, f)
        
    # Extract super-features
    print("Extracting 180+ dimensional super-features...")
    train_feat = extract_super_features(train_x)
    test_feat = extract_super_features(test_x)
    np.save(out_dir / "train_feat.npy", train_feat)
    np.save(out_dir / "test_feat.npy", test_feat)
    print(f"Features extracted. Shape: {train_feat.shape}")
    
    # Classify 4 regimes
    print("Classifying flight regimes...")
    regimes_tr, speeds_tr, curv_tr, acc_perp_tr, acc_par_tr = classify_4regimes(train_x)
    regimes_te, speeds_te, curv_te, acc_perp_te, acc_par_te = classify_4regimes(test_x)
    
    np.save(out_dir / "regimes_train.npy", regimes_tr)
    np.save(out_dir / "regimes_test.npy", regimes_te)
    np.save(out_dir / "speeds_train.npy", speeds_tr)
    np.save(out_dir / "speeds_test.npy", speeds_te)
    
    print("Done Preprocessing!")

if __name__ == "__main__":
    main()
