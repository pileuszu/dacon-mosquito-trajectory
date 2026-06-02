import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hashlib
import traceback
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification
from step66_super_feature.model_s67 import SotaSpatiotemporalModel
from step65_spatiotemporal_ai.model import extract_features, SotaFocalSoftHitLoss

class MosquitoCandidatesDataset(Dataset):
    def __init__(self, X, y, candidates, regimes):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.candidates = torch.tensor(candidates, dtype=torch.float32)
        self.regimes = torch.tensor(regimes, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.candidates[idx], self.regimes[idx]

def stable_fold_id(sample_id: str, folds: int) -> int:
    import hashlib
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds

def classify_regimes_np(X):
    EPS = 1e-8
    N = X.shape[0]
    last_v = (X[:, -1] - X[:, -2]) / 0.04
    speeds = np.linalg.norm(last_v, axis=1)
    prev_v = (X[:, -2] - X[:, -3]) / 0.04
    last_a = (last_v - prev_v) / 0.04
    t_dir = last_v / (speeds[:, None] + EPS)
    acc_par_scalar = np.sum(last_a * t_dir, axis=1)
    acc_perp = last_a - acc_par_scalar[:, None] * T_tr_dir
    # Note: wait! In train_cfm_steering.py, lines 44-45:
    # t_dir = last_v / (speeds[:, None] + EPS)
    # acc_par_scalar = np.sum(last_a * t_dir, axis=1)
    # acc_perp = last_a - acc_par_scalar[:, None] * t_dir
    # We should match it exactly! Let's define the local variables.
    return None # wait, let's write a clean version of classify_regimes_np

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
        if speeds[i] <= 0.50:
            if is_steering[i]:
                regimes[i] = 1  # Slow-Turning
            else:
                regimes[i] = 0  # Slow-Straight
        else:
            if is_steering[i]:
                regimes[i] = 3  # Fast-Turning
            else:
                regimes[i] = 2  # Fast-Straight
    return regimes

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (fine-tuning)")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent embedding dimension")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Noise scale")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--loss-dist-weight", type=float, default=200.0, help="Weight for L1 distance loss")
    args = parser.parse_args()
    
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    folds = 5
    noise_scale = args.noise_scale
    dropout = args.dropout
    loss_dist_weight = args.loss_dist_weight
    
    send_discord_notification(None, f"🚀 Started: [s67 Fast-Turning Fine-Tuning CFM] lr={lr}, epochs={epochs}")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Paths
        data_dir_s65 = Path("step65_spatiotemporal_ai/data")
        data_dir_s66 = Path("step66_super_feature/data")
        models_dir_steering = Path("step66_super_feature/models/s67_steering")
        models_dir_fast_turn = Path("step66_super_feature/models/s67_fast_turning")
        models_dir_fast_turn.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        X_train_all = np.load(data_dir_s65 / "train_x.npy")
        y_train_all = np.load(data_dir_s65 / "train_y.npy")
        candidates_train_all = np.load(data_dir_s66 / "train_candidates_hybrid_v3.npy")
        with open(data_dir_s65 / "train_ids.json", "r") as f:
            train_ids_all = json.load(f)
            
        test_x_all = np.load(data_dir_s65 / "test_x.npy")
        test_candidates_all = np.load(data_dir_s66 / "test_candidates_hybrid_v3.npy")
        with open(data_dir_s65 / "test_ids.json", "r") as f:
            test_ids_all = json.load(f)
            
        regimes_tr_all = classify_regimes_np(X_train_all)
        regimes_te_all = classify_regimes_np(test_x_all)
        
        fold_ids_all = np.asarray([stable_fold_id(sid, folds) for sid in train_ids_all])
        
        # Filter for Fast-Turning (Regime 3)
        is_fast_tr = regimes_tr_all == 3
        is_fast_te = regimes_te_all == 3
        
        oof_preds_1step = np.zeros_like(y_train_all)
        p_last_tr = X_train_all[:, -1]
        oof_preds_1step[:] = p_last_tr
        
        test_preds_1step = np.zeros((len(test_x_all), 3))
        
        print(f"\n=================== Starting 5-Fold Fast-Turning Expert CFM Fine-Tuning (N_fast={np.sum(is_fast_tr)}) ===================")
        
        for fold in range(folds):
            print(f"\n--- Fold {fold+1} / {folds} ---")
            
            # Split train / val on the entire dataset first, then filter for Fast-Turning
            train_mask_all = fold_ids_all != fold
            val_mask_all = fold_ids_all == fold
            
            train_mask = train_mask_all & is_fast_tr
            val_mask = val_mask_all & is_fast_tr
            
            X_tr, y_tr, cand_tr, reg_tr = X_train_all[train_mask], y_train_all[train_mask], candidates_train_all[train_mask], regimes_tr_all[train_mask]
            X_val, y_val, cand_val, reg_val = X_train_all[val_mask], y_train_all[val_mask], candidates_train_all[val_mask], regimes_tr_all[val_mask]
            
            # Map regimes to 2 (Fast-Turning is regime 3, mapped to 2)
            reg_tr_mapped = np.ones(len(reg_tr), dtype=int) * 2
            reg_val_mapped = np.ones(len(reg_val), dtype=int) * 2
            
            train_loader = DataLoader(MosquitoCandidatesDataset(X_tr, y_tr, cand_tr, reg_tr_mapped), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(MosquitoCandidatesDataset(X_val, y_val, cand_val, reg_val_mapped), batch_size=256, shuffle=False)
            
            # Load pre-trained stats
            stats = torch.load(models_dir_steering / f"stats_fold_{fold}.pt")
            mean_stats = stats["mean"].to(device)
            std_stats = stats["std"].to(device)
            torch.save({"mean": mean_stats.cpu(), "std": std_stats.cpu()}, models_dir_fast_turn / f"stats_fold_{fold}.pt")
            
            # Instantiate model
            model = SotaSpatiotemporalModel(feature_dim=47, latent_dim=args.latent_dim, num_candidates=43, max_norm=0.05, d_mamba_in=9, dropout=dropout).to(device)
            
            # Load pre-trained weights from steering model
            model.load_state_dict(torch.load(models_dir_steering / f"model_fold_{fold}.pt", map_location=device))
            print("Loaded pre-trained weights from Steering model.")
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = SotaFocalSoftHitLoss(delta=0.001026, alpha=200.0, target_dist=0.01).to(device)
            
            best_val_score = 999.0
            best_val_preds_1step = None
            
            # Fine-tuning loop
            for epoch in range(1, epochs + 1):
                model.train()
                train_loss_flow = 0.0
                for Xb, yb, candb, regb in train_loader:
                    Xb, yb, candb, regb = Xb.to(device), yb.to(device), candb.to(device), regb.to(device)
                    optimizer.zero_grad()
                    ft, df_c, df_s, _, _, (t_v, n_v, b_v), raw_curv = extract_features(Xb, mean_stats, std_stats)
                    pred_v, target_v, x_0, h_context = model.forward_flow(Xb, ft, df_c, df_s, candb, yb, regb, noise_scale=noise_scale)
                    
                    loss_cfm = F.huber_loss(pred_v, target_v, delta=0.001)
                    
                    v_N = torch.sum(pred_v * n_v, dim=1)
                    v_B = torch.sum(pred_v * b_v, dim=1)
                    # We are in turning regime, so lambda is lower
                    loss_frenet = (25.0 / (1.0 + 2.0 * raw_curv) * (v_N**2 + v_B**2)).mean()
                    
                    loss_flow = loss_cfm + loss_frenet
                    train_loss_flow += loss_flow.item()
                    
                    # Consistent predictions
                    t0 = torch.zeros(Xb.shape[0], device=device)
                    v0 = model.vector_field(x_0, t0, h_context)
                    pred_1step = x_0 + v0
                    
                    loss_pred = criterion(pred_1step, yb)
                    loss_dist = torch.norm(pred_1step - yb, dim=1).mean()
                    
                    loss = 10.0 * loss_flow + 1.0 * loss_pred + loss_dist_weight * loss_dist
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
                # Val Evaluation
                model.eval()
                val_preds_1step_list = []
                with torch.no_grad():
                    for Xv, yv, candv, regv in val_loader:
                        Xv, yv, candv, regv = Xv.to(device), yv.to(device), candv.to(device), regv.to(device)
                        ft, df_c, df_s, _, _, _, _ = extract_features(Xv, mean_stats, std_stats)
                        pv_1 = model.predict(Xv, ft, df_c, df_s, candv, regv, steps=1)
                        val_preds_1step_list.append(pv_1.cpu().numpy())
                        
                val_preds_1 = np.concatenate(val_preds_1step_list, axis=0)
                val_dists = np.linalg.norm(val_preds_1 - y_val, axis=1)
                
                # Hit Rate OOF val score
                val_score = -np.mean(val_dists <= 0.01) * 100.0
                
                if val_score < best_val_score:
                    best_val_score = val_score
                    torch.save(model.state_dict(), models_dir_fast_turn / f"model_fold_{fold}.pt")
                    best_val_preds_1step = val_preds_1
                    
            val_indices = np.where(val_mask)[0]
            oof_preds_1step[val_indices] = best_val_preds_1step
            print(f"Fold {fold+1} best Fast-Turning val Hit Rate: {-best_val_score:.3f}% (N={len(X_val)})")
            
            # Predict test set using best fold model
            model.load_state_dict(torch.load(models_dir_fast_turn / f"model_fold_{fold}.pt", map_location=device))
            model.eval()
            
            test_x_tensor = torch.tensor(test_x_all, dtype=torch.float32).to(device)
            test_cand_tensor = torch.tensor(test_candidates_all, dtype=torch.float32).to(device)
            test_reg_mapped = np.ones(len(test_x_all), dtype=int) * 2
            test_reg_tensor = torch.tensor(test_reg_mapped, dtype=torch.long).to(device)
            
            with torch.no_grad():
                ft_te, df_c_te, df_s_te, _, _, _, _ = extract_features(test_x_tensor, mean_stats, std_stats)
                tp_1 = model.predict(test_x_tensor, ft_te, df_c_te, df_s_te, test_cand_tensor, test_reg_tensor, steps=1)
                test_preds_1step += tp_1.cpu().numpy() / folds
                
        # Save predictions
        np.save(models_dir_fast_turn / "oof_preds_cfm_1step.npy", oof_preds_1step)
        np.save(models_dir_fast_turn / "test_preds_cfm_1step.npy", test_preds_1step)
        
        # Calculate OOF Hit Rate for Fast-Turning samples
        fast_oof_dists = np.linalg.norm(oof_preds_1step[is_fast_tr] - y_train_all[is_fast_tr], axis=1)
        fast_hr = np.mean(fast_oof_dists <= 0.01) * 100.0
        
        success_msg = f"🏆 Finished fine-tuning Fast-Turning Expert CFM model!\n" \
                      f"Fast-Turning OOF Hit Rate @ 1cm: {fast_hr:.4f}%"
        print(success_msg)
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [s67 Fast-Turning Expert CFM]\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
