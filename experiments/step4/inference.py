import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

from .config import *
from .model import BiomechanicalTransformer

def get_rotation_matrix(vector):
    x, y, _ = vector
    mag_xy = np.sqrt(x**2 + y**2)
    if mag_xy < 1e-6:
        return np.eye(3)
    cos_theta = x / mag_xy
    sin_theta = y / mag_xy
    return np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=np.float32)

def inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    model = BiomechanicalTransformer(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    model_path = 'step4_best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Test Files
    test_files = sorted(TEST_DIR.glob('*.csv'))
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    results = []

    print(f"Starting inference on {len(test_files)} samples...")
    with torch.no_grad():
        for fpath in tqdm(test_files):
            df = pd.read_csv(fpath)
            xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
            
            # Physics & Biomechanical Features
            vel = np.diff(xyz, axis=0, prepend=xyz[0:1])
            accel = np.diff(vel, axis=0, prepend=vel[0:1])
            
            last_move = xyz[-1] - xyz[-2]
            rot_mat = get_rotation_matrix(last_move)
            
            last_xyz = xyz[-1]
            xyz_norm = (xyz - last_xyz) @ rot_mat.T
            vel_norm = vel @ rot_mat.T
            accel_norm = accel @ rot_mat.T
            
            v_mag = np.linalg.norm(vel, axis=1, keepdims=True)
            v_mag_safe = np.where(v_mag < 1e-6, 1e-6, v_mag)
            cross_prod = np.cross(vel, accel)
            curvature = np.linalg.norm(cross_prod, axis=1, keepdims=True) / (v_mag_safe**3)
            curvature = np.clip(curvature, 0, 100)
            
            speed = v_mag
            vel_prev = np.roll(vel, 1, axis=0); vel_prev[0] = vel[0]
            v_mag_prev = np.roll(v_mag, 1, axis=0); v_mag_prev[0] = v_mag[0]
            dot_prod = np.sum(vel * vel_prev, axis=1, keepdims=True)
            cos_sim = dot_prod / (v_mag_safe * (v_mag_prev + 1e-6))
            turn_angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))
            
            features = np.concatenate([xyz_norm, vel_norm, accel_norm, curvature, speed, turn_angle], axis=1)
            cv_prior_rel_rot = (2.0 * last_move) @ rot_mat.T
            
            # To Tensors
            seq_t = torch.tensor(features).unsqueeze(0).to(device)
            cv_prior_t = torch.tensor(cv_prior_rel_rot).unsqueeze(0).to(device)
            last_pos_t = torch.tensor(last_xyz).unsqueeze(0).to(device)
            rot_mat_t = torch.tensor(rot_mat).unsqueeze(0).to(device)
            
            # Predict
            final_pred, _, _ = model(seq_t, cv_prior_t, last_pos_t, rot_mat_t)
            
            pred_xyz = final_pred.cpu().numpy()[0]
            results.append({'id': fpath.stem, 'x': pred_xyz[0], 'y': pred_xyz[1], 'z': pred_xyz[2]})

    # 3. Save
    res_df = pd.DataFrame(results)
    res_df = submission_df[['id']].merge(res_df, on='id', how='left')
    os.makedirs('outputs/step4', exist_ok=True)
    res_df.to_csv('outputs/step4/submission.csv', index=False)
    print(f"Submission saved to outputs/step4/submission.csv")

if __name__ == "__main__":
    inference()
