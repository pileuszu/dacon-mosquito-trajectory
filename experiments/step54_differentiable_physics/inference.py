import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import DifferentiableJointSelector, extract_features

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = Path("step54_differentiable_physics/data")
    models_dir = Path("step54_differentiable_physics/models")
    
    test_x = np.load(data_dir / "test_x.npy")
    test_candidates = np.load(data_dir / "test_candidates.npy")
    
    test_tensor = torch.tensor(test_x, dtype=torch.float32)
    cand_tensor = torch.tensor(test_candidates, dtype=torch.float32)
    
    batch_size = 256
    fold_predictions = []
    
    for fold in range(5):
        print(f"Running inference for Fold {fold+1}...")
        stats_path = models_dir / f"stats_fold_{fold}.pt"
        model_path = models_dir / f"model_fold_{fold}.pt"
        
        if not stats_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Missing stats/model for fold {fold}")
            
        stats = torch.load(stats_path, map_location=device)
        mean_stats = stats["mean"].to(device)
        std_stats = stats["std"].to(device)
        
        model = DifferentiableJointSelector(feature_dim=38, latent_dim=64, num_candidates=36).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        preds_list = []
        with torch.no_grad():
            for i in range(0, len(test_tensor), batch_size):
                Xb = test_tensor[i:i+batch_size].to(device)
                candb = cand_tensor[i:i+batch_size].to(device)
                
                ft, df, _, _, _, _, _, _, _, _, _ = extract_features(Xb, mean_stats, std_stats)
                pred, _ = model(ft, df, candb)
                preds_list.append(pred.cpu().numpy())
                
        fold_predictions.append(np.concatenate(preds_list, axis=0))
        
    final_test_preds = np.mean(fold_predictions, axis=0)
    out_path = data_dir / "test_preds_soft.npy"
    np.save(out_path, final_test_preds)
    print(f"Saved step54 true test predictions to {out_path} with shape {final_test_preds.shape}")

if __name__ == "__main__":
    main()
