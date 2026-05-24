import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

from .config import *
from .model import LSTMHybridModel

def inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    model = LSTMHybridModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    model_path = 'step2_best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Prepare Test Data
    test_files = sorted(TEST_DIR.glob('*.csv'))
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    results = []

    print(f"Starting inference on {len(test_files)} samples...")
    with torch.no_grad():
        for fpath in tqdm(test_files):
            df = pd.read_csv(fpath)
            xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
            
            # Physics Features
            vel = np.diff(xyz, axis=0, prepend=xyz[0:1])
            accel = np.diff(vel, axis=0, prepend=vel[0:1])
            features = np.concatenate([xyz, vel, accel], axis=1)
            
            # Normalization
            last_xyz = xyz[-1]
            features[:, 0:3] = features[:, 0:3] - last_xyz
            
            # CV Prior
            cv_prior_rel = 2.0 * (last_xyz - xyz[-2])
            
            # Convert to tensor and add batch dim
            seq_tensor = torch.tensor(features).unsqueeze(0).to(device)
            cv_prior_tensor = torch.tensor(cv_prior_rel).unsqueeze(0).to(device)
            last_pos_tensor = torch.tensor(last_xyz).unsqueeze(0).to(device)
            
            # Predict
            final_pred, _ = model(seq_tensor, cv_prior_tensor, last_pos_tensor)
            
            pred_xyz = final_pred.cpu().numpy()[0]
            results.append({
                'id': fpath.stem,
                'x': pred_xyz[0],
                'y': pred_xyz[1],
                'z': pred_xyz[2]
            })

    # 3. Create Submission
    res_df = pd.DataFrame(results)
    # Ensure ID order matches sample submission
    res_df = submission_df[['id']].merge(res_df, on='id', how='left')
    
    os.makedirs('outputs/step2', exist_ok=True)
    save_path = 'outputs/step2/submission.csv'
    res_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

if __name__ == "__main__":
    inference()
