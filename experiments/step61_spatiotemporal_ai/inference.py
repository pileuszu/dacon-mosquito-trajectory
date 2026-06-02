import os
import sys
import json
import csv
import torch
import numpy as np
from pathlib import Path
import traceback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import extract_features, SotaSpatiotemporalModel

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_dir = Path("step61_spatiotemporal_ai/data")
        models_dir = Path("step61_spatiotemporal_ai/models")
        out_dir = Path("outputs/step61_spatiotemporal_ai")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        test_x = np.load(data_dir / "test_x.npy")
        test_candidates = np.load(data_dir / "test_candidates.npy")
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        test_tensor = torch.tensor(test_x, dtype=torch.float32)
        ensemble_preds_2step = np.zeros((len(test_x), 3))
        ensemble_preds_1step = np.zeros((len(test_x), 3))
        
        for fold in range(5):
            stats = torch.load(models_dir / f"stats_fold_{fold}.pt")
            tr_mean = stats["mean"]
            tr_std = stats["std"]
            
            model = SotaSpatiotemporalModel(feature_dim=44, latent_dim=128, num_candidates=36, max_norm=0.05, d_mamba_in=6).to(device)
            model.load_state_dict(torch.load(models_dir / f"model_fold_{fold}.pt", map_location=device))
            model.eval()
            
            with torch.no_grad():
                test_tensor_dev = test_tensor.to(device)
                ft_te, df_c, df_s, _, _ = extract_features(test_tensor_dev, tr_mean.to(device), tr_std.to(device))
                candidates_te = torch.tensor(test_candidates, dtype=torch.float32).to(device)
                
                fold_pred_2 = model.predict(test_tensor_dev, ft_te, df_c, df_s, candidates_te, steps=2)
                fold_pred_1 = model.predict(test_tensor_dev, ft_te, df_c, df_s, candidates_te, steps=1)
                
                ensemble_preds_2step += fold_pred_2.cpu().numpy() / 5.0
                ensemble_preds_1step += fold_pred_1.cpu().numpy() / 5.0
                
        np.save(data_dir / "test_preds_cfm_2step.npy", ensemble_preds_2step)
        np.save(data_dir / "test_preds_cfm_1step.npy", ensemble_preds_1step)
        print(f"Step 61 test inference completed successfully.")
        
    except Exception as e:
        print(f"Step 61 inference failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
