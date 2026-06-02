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

def classify_regimes_np(X):
    EPS = 1e-8
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
        if is_steering[i]:
            regimes[i] = 2
        elif speeds[i] <= 0.50:
            regimes[i] = 0
        else:
            regimes[i] = 1
    return regimes

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_dir = Path("step64_spatiotemporal_ai/data")
        models_dir = Path("step64_spatiotemporal_ai/models")
        out_dir = Path("outputs/step64_spatiotemporal_ai")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        test_x = np.load(data_dir / "test_x.npy")
        test_candidates = np.load(data_dir / "test_candidates.npy")
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        regimes_te = classify_regimes_np(test_x)
        ensemble_preds_2step = np.zeros((len(test_x), 3))
        ensemble_preds_1step = np.zeros((len(test_x), 3))
        
        regime_names = ["cruising", "gliding", "steering"]
        
        for regime in range(3):
            regime_mask = regimes_te == regime
            regime_name = regime_names[regime]
            if np.sum(regime_mask) == 0:
                continue
                
            X_reg = test_x[regime_mask]
            cand_reg = test_candidates[regime_mask]
            test_tensor = torch.tensor(X_reg, dtype=torch.float32)
            
            reg_preds_2 = np.zeros((len(X_reg), 3))
            reg_preds_1 = np.zeros((len(X_reg), 3))
            
            for fold in range(5):
                stats = torch.load(models_dir / f"stats_{regime_name}_fold_{fold}.pt")
                tr_mean = stats["mean"]
                tr_std = stats["std"]
                
                model = SotaSpatiotemporalModel(feature_dim=47, latent_dim=128, num_candidates=36, max_norm=0.05, d_mamba_in=9).to(device)
                model.load_state_dict(torch.load(models_dir / f"model_{regime_name}_fold_{fold}.pt", map_location=device))
                model.eval()
                
                with torch.no_grad():
                    test_tensor_dev = test_tensor.to(device)
                    ft_te, df_c, df_s, _, _, _ = extract_features(test_tensor_dev, tr_mean.to(device), tr_std.to(device))
                    candidates_te = torch.tensor(cand_reg, dtype=torch.float32).to(device)
                    
                    fold_pred_2 = model.predict(test_tensor_dev, ft_te, df_c, df_s, candidates_te, steps=2)
                    fold_pred_1 = model.predict(test_tensor_dev, ft_te, df_c, df_s, candidates_te, steps=1)
                    
                    reg_preds_2 += fold_pred_2.cpu().numpy() / 5.0
                    reg_preds_1 += fold_pred_1.cpu().numpy() / 5.0
            
            ensemble_preds_2step[regime_mask] = reg_preds_2
            ensemble_preds_1step[regime_mask] = reg_preds_1
            
        np.save(data_dir / "test_preds_cfm_2step.npy", ensemble_preds_2step)
        np.save(data_dir / "test_preds_cfm_1step.npy", ensemble_preds_1step)
        print(f"Step 64 Split-CFM test inference completed successfully.")
        
    except Exception as e:
        print(f"Step 64 inference failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
