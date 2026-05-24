import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())

from step4.config import *
from step4.model import BiomechanicalTransformer
from step4.dataset import BiomechanicalDataset

def generate_step4_preds_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = BiomechanicalTransformer(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=0.0
    ).to(device)
    
    model_path = 'step4_best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Dataset
    data_dir = Path("data/open")
    test_files = sorted(list((data_dir / "test").glob("*.csv")))
    
    dataset = BiomechanicalDataset(test_files, is_test=True) 
    
    results = []
    print(f"Generating Step 4 predictions for {len(test_files)} test samples...")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            fid = test_files[i].stem
            
            seq = item['seq'].unsqueeze(0).to(device)
            cv_prior = item['cv_prior'].unsqueeze(0).to(device)
            last_pos = item['last_pos'].unsqueeze(0).to(device)
            rot_mat = item['rot_mat'].unsqueeze(0).to(device)
            
            final_pred, _, _ = model(seq, cv_prior, last_pos, rot_mat)
            pos = final_pred.squeeze().cpu().numpy()
            
            results.append({
                "id": fid,
                "x": pos[0],
                "y": pos[1],
                "z": pos[2]
            })
            
    out_df = pd.DataFrame(results)
    out_path = Path("step12/step4_preds_test.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Step 4 test predictions saved to {out_path}")

if __name__ == "__main__":
    generate_step4_preds_test()
