import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import SotaSpatiotemporalModel, extract_features

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = Path("step65_spatiotemporal_ai/data")
    models_dir = Path("step65_spatiotemporal_ai/models")
    
    # Load test data
    test_x = np.load(data_dir / "test_x.npy")
    test_candidates = np.load(data_dir / "test_candidates.npy")
    
    # Classify regimes for test set
    regimes_te = classify_regimes_np(test_x)
    
    test_preds_1step = np.zeros((len(test_x), 3))
    test_preds_2step = np.zeros((len(test_x), 3))
    
    # Convert numpy inputs to torch tensors
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32).to(device)
    candidates_tensor = torch.tensor(test_candidates, dtype=torch.float32).to(device)
    regimes_tensor = torch.tensor(regimes_te, dtype=torch.long).to(device)
    
    print("Starting 5-Fold Inference on Test Set...")
    
    for fold in range(5):
        print(f"  Processing Fold {fold+1}...")
        
        # Load stats for normalization
        stats = torch.load(models_dir / f"stats_fold_{fold}.pt", map_location=device)
        tr_mean = stats["mean"]
        tr_std = stats["std"]
        
        # Extract normalized features for the test set
        with torch.no_grad():
            ft_te, df_c_te, df_s_te, _, _, _, _ = extract_features(test_x_tensor, tr_mean, tr_std)
            
        # Initialize model architecture
        model = SotaSpatiotemporalModel(feature_dim=47, latent_dim=128, num_candidates=36, max_norm=0.05, d_mamba_in=9).to(device)
        model.load_state_dict(torch.load(models_dir / f"model_fold_{fold}.pt", map_location=device))
        model.eval()
        
        # Predict
        with torch.no_grad():
            pv_1 = model.predict(test_x_tensor, ft_te, df_c_te, df_s_te, candidates_tensor, regimes_tensor, steps=1)
            pv_2 = model.predict(test_x_tensor, ft_te, df_c_te, df_s_te, candidates_tensor, regimes_tensor, steps=2)
            
            test_preds_1step += pv_1.cpu().numpy() / 5.0
            test_preds_2step += pv_2.cpu().numpy() / 5.0
            
    print("Saving test predictions...")
    np.save(data_dir / "test_preds_cfm_1step.npy", test_preds_1step)
    np.save(data_dir / "test_preds_cfm_2step.npy", test_preds_2step)
    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
