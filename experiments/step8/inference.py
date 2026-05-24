import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from .config import *
from .dataset import CandidateDataset
from .model import CandidateTransformerSelector

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    test_files = sorted(list(TEST_DIR.glob("*.csv")))
    test_dataset = CandidateDataset(test_files, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. Load Model
    model = CandidateTransformerSelector(
        input_dim=12, 
        num_candidates=NUM_CANDIDATES,
        d_model=D_MODEL,
        dropout=DROPOUT
    ).to(device)
    
    model_save_path = Path(__file__).parent / "models" / "best_model.pth"
    if model_save_path.exists():
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded model from {model_save_path}")
    else:
        print("WARNING: No trained model found. Running with random weights.")

    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            ids = batch['id']
            seq = batch['seq'].to(device)
            candidates = batch['candidates'].to(device)
            cand_feats = batch['cand_feats'].to(device)
            rot_mat = batch['rot_mat'].to(device)
            
            logits, global_correction = model(seq, candidates, cand_feats, rot_mat)
            
            # Select best candidate
            pred_idx = logits.argmax(dim=-1)
            
            # Final prediction = Best Candidate + Global Correction
            for i in range(len(ids)):
                chosen_cand = candidates[i, pred_idx[i]].cpu().numpy()
                corr = global_correction[i].cpu().numpy()
                final_pos = chosen_cand + corr
                
                results.append({
                    "id": ids[i],
                    "x": final_pos[0],
                    "y": final_pos[1],
                    "z": final_pos[2]
                })

    # 3. Save Submission
    submission_df = pd.DataFrame(results)
    submission_path = OUTPUT_DIR / "submission_step8.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    inference()
