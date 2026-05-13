import torch
import pandas as pd
import numpy as np
import os
import sys
from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Import from Step 1
sys.path.append('step1_coordinate_invariant_baseline')
from dataset import MosquitoDataset

# Local imports
from model import MultimodalAnchorModel, load_anchors

# Configuration
TEST_DIR = 'data/open/test/'
SUBMISSION_PATH = 'data/open/sample_submission.csv'
MODEL_PATH = 'outputs/step2/best_multimodal_model.pth'
OUTPUT_PATH = 'outputs/step2/submission.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_submission():
    anchors = load_anchors().to(DEVICE)
    submission = pd.read_csv(SUBMISSION_PATH)
    submission[['x', 'y', 'z']] = submission[['x', 'y', 'z']].astype(float)
    
    test_files = sorted(glob(os.path.join(TEST_DIR, '*.csv')))
    test_ds = MosquitoDataset(TEST_DIR, label_path=None, file_list=test_files, mode='test')
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    model = MultimodalAnchorModel(n_anchors=len(anchors)).to(DEVICE)
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
            logits, offsets = model(hist)
            
            # Pick best anchor and its offset
            best_idx = torch.argmax(logits, dim=1)
            # best_idx: [B], selected_anchors: [B, 3]
            selected_anchors = anchors[best_idx]
            # selected_offsets: [B, 3]
            selected_offsets = offsets[torch.arange(hist.size(0)), best_idx]
            
            output_rel = (selected_anchors + selected_offsets).cpu().numpy()
            origin = origin.numpy()
            output_abs = origin + output_rel
            
            for i, f_id in enumerate(file_ids):
                results[f_id] = output_abs[i]
                
    for i, row in tqdm(submission.iterrows(), total=len(submission), desc="Filling CSV"):
        file_id = row['id']
        if file_id in results:
            submission.loc[i, ['x', 'y', 'z']] = results[file_id]
            
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nStep 2 Submission file saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    if os.path.exists(TEST_DIR):
        generate_submission()
    else:
        print("Test directory not found.")
