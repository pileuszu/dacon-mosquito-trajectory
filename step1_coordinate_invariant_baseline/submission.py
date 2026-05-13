import torch
import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm.auto import tqdm
from dataset import MosquitoDataset
from model import BaselineGRU
from torch.utils.data import DataLoader

# Configuration
TEST_DIR = 'data/open/test/'
SUBMISSION_PATH = 'data/open/sample_submission.csv'
MODEL_PATH = 'outputs/step1/best_baseline_model.pth'
OUTPUT_PATH = 'outputs/step1/submission.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_submission():
    # 1. Load Submission Template
    submission = pd.read_csv(SUBMISSION_PATH)
    submission[['x', 'y', 'z']] = submission[['x', 'y', 'z']].astype(float)
    
    # 2. Setup Dataset & Loader
    test_files = sorted(glob(os.path.join(TEST_DIR, '*.csv')))
    test_ds = MosquitoDataset(TEST_DIR, label_path=None, file_list=test_files, mode='test')
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    # 3. Load Model
    model = BaselineGRU().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model.eval()
    
    results = {}
    
    with torch.no_grad():
        for hist, origin, file_ids in tqdm(test_loader, desc="Inference"):
            hist = hist.to(DEVICE)
            output_rel = model(hist).cpu().numpy() # Relative prediction
            origin = origin.numpy() # Original t=0 point
            
            # 4. Convert back to Absolute Coordinates: P_abs = P_origin + P_rel
            output_abs = origin + output_rel
            
            for i, f_id in enumerate(file_ids):
                results[f_id] = output_abs[i]
                
    # 5. Fill Submission DataFrame
    for i, row in tqdm(submission.iterrows(), total=len(submission), desc="Filling CSV"):
        file_id = row['id']
        if file_id in results:
            submission.loc[i, ['x', 'y', 'z']] = results[file_id]
        else:
            print(f"Warning: No prediction for {file_id}")
            
    # 6. Save
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission file saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    if os.path.exists(TEST_DIR):
        generate_submission()
    else:
        print(f"Test directory not found at {TEST_DIR}")
