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
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--lambda0", type=float, default=15.0, help="Base Frenet-Flow penalty weight")
    parser.add_argument("--eta", type=float, default=2.0, help="Curvature scale decay rate for Frenet loss")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Noise perturbation scale")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--loss-dist-weight", type=float, default=200.0, help="Weight for direct L1 distance loss")
    args = parser.parse_args()
    
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    folds = 5
    noise_scale = args.noise_scale
    lambda0 = args.lambda0
    eta = args.eta
    dropout = args.dropout
    loss_dist_weight = args.loss_dist_weight
    
    send_discord_notification(None, f"🚀 Started: [s67 Gliding Expert CFM] latent_dim={args.latent_dim}, heads={args.num_heads}, lambda0={lambda0}, loss_dist_weight={loss_dist_weight}, dropout={dropout}")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Paths
        data_dir_s65 = Path("step65_spatiotemporal_ai/data")
        data_dir_s66 = Path("step66_super_feature/data")
        models_dir_gliding = Path("step66_super_feature/models/s67_gliding")
        models_dir_gliding.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        X_train_all = np.load(data_dir_s65 / "train_x.npy")
        y_train_all = np.load(data_dir_s65 / "train_y.npy")
        candidates_train_all = np.load(data_dir_s66 / "train_candidates_hybrid_v3.npy") # [10000, 43, 3]
        with open(data_dir_s65 / "train_ids.json", "r") as f:
            train_ids_all = json.load(f)
            
        test_x_all = np.load(data_dir_s65 / "test_x.npy")
        test_candidates_all = np.load(data_dir_s66 / "test_candidates_hybrid_v3.npy") # [10000, 43, 3]
        with open(data_dir_s65 / "test_ids.json", "r") as f:
            test_ids_all = json.load(f)
            
        regimes_tr_all = classify_regimes_np(X_train_all)
        regimes_te_all = classify_regimes_np(test_x_all)
        
        fold_ids_all = np.asarray([stable_fold_id(sid, folds) for sid in train_ids_all])
        
        # Filter for Gliding (Regime 2)
        is_gliding_tr = (regimes_tr_all == 2)
        
        oof_preds_1step = np.zeros_like(y_train_all)
        oof_preds_2step = np.zeros_like(y_train_all)
        
        # Initialize OOF predictions with baseline predictions or copycat from last pos for non-gliding
        p_last_tr = X_train_all[:, -1]
        oof_preds_1step[:] = p_last_tr
        oof_preds_2step[:] = p_last_tr
        
        test_preds_1step = np.zeros((len(test_x_all), 3))
        test_preds_2step = np.zeros((len(test_x_all), 3))
        
        print(f"\n=================== Starting 5-Fold Gliding Expert CFM training (N_gliding={np.sum(is_gliding_tr)}) ===================")
        
        for fold in range(folds):
            print(f"\n--- Fold {fold+1} / {folds} ---")
            
            # Split train / val on the entire dataset first, then filter for gliding
            train_mask_all = fold_ids_all != fold
            val_mask_all = fold_ids_all == fold
            
            train_mask = train_mask_all & is_gliding_tr
            val_mask = val_mask_all & is_gliding_tr
            
            X_tr, y_tr, cand_tr, reg_tr = X_train_all[train_mask], y_train_all[train_mask], candidates_train_all[train_mask], regimes_tr_all[train_mask]
            X_val, y_val, cand_val, reg_val = X_train_all[val_mask], y_train_all[val_mask], candidates_train_all[val_mask], regimes_tr_all[val_mask]
            
            # Map regimes to 0 for the model (single class context)
            reg_tr_mapped = np.zeros(len(reg_tr), dtype=int)
            reg_val_mapped = np.zeros(len(reg_val), dtype=int)
            
            train_loader = DataLoader(MosquitoCandidatesDataset(X_tr, y_tr, cand_tr, reg_tr_mapped), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(MosquitoCandidatesDataset(X_val, y_val, cand_val, reg_val_mapped), batch_size=256, shuffle=False)
            
            # Feature Stats
            with torch.no_grad():
                _, _, _, mean_stats, std_stats, _, _ = extract_features(torch.tensor(X_tr, dtype=torch.float32).to(device))
            torch.save({"mean": mean_stats.cpu(), "std": std_stats.cpu()}, models_dir_gliding / f"stats_fold_{fold}.pt")
            
            # Model with num_candidates=43
            model = SotaSpatiotemporalModel(feature_dim=47, latent_dim=args.latent_dim, num_candidates=43, max_norm=0.05, d_mamba_in=9, dropout=dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = SotaFocalSoftHitLoss(delta=0.001026, alpha=200.0, target_dist=0.01).to(device)
            
            best_val_score = 999.0
            best_val_preds_1step = None
            best_val_preds_2step = None
            
            for epoch in range(1, epochs + 1):
                model.train()
                for Xb, yb, candb, regb in train_loader:
                    Xb, yb, candb, regb = Xb.to(device), yb.to(device), candb.to(device), regb.to(device)
                    optimizer.zero_grad()
                    ft, df_c, df_s, _, _, (t_v, n_v, b_v), raw_curv = extract_features(Xb, mean_stats, std_stats)
                    pred_v, target_v, x_0, h_context = model.forward_flow(Xb, ft, df_c, df_s, candb, yb, regb, noise_scale=noise_scale)
                    
                    loss_cfm = F.huber_loss(pred_v, target_v, delta=0.001)
                    
                    v_N = torch.sum(pred_v * n_v, dim=1)
                    v_B = torch.sum(pred_v * b_v, dim=1)
                    lambda_adaptive = lambda0 / (1.0 + eta * raw_curv)
                    loss_frenet = (lambda_adaptive * (v_N**2 + v_B**2)).mean()
                    
                    loss_flow = loss_cfm + loss_frenet
                    
                    # Consistency & Sigmoid Hit prediction loss
                    t0 = torch.zeros(Xb.shape[0], device=device)
                    v0 = model.vector_field(x_0, t0, h_context)
                    pred_1step = x_0 + v0
                    dt = 0.5
                    x_half = x_0 + v0 * dt
                    t05 = torch.ones(Xb.shape[0], device=device) * 0.5
                    v05 = model.vector_field(x_half, t05, h_context)
                    pred_2step = x_half + v05 * dt
                    
                    loss_pred = criterion(pred_1step, yb) + criterion(pred_2step, yb)
                    loss_cons = F.mse_loss(pred_1step, pred_2step)
                    
                    loss_dist = (torch.norm(pred_1step - yb, dim=1).mean() + torch.norm(pred_2step - yb, dim=1).mean()) * 0.5
                    loss = 10.0 * loss_flow + 1.0 * loss_pred + 50.0 * loss_cons + loss_dist_weight * loss_dist
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
                # Val Evaluation
                model.eval()
                val_preds_1step_list = []
                val_preds_2step_list = []
                with torch.no_grad():
                    for Xv, yv, candv, regv in val_loader:
                        Xv, yv, candv, regv = Xv.to(device), yv.to(device), candv.to(device), regv.to(device)
                        ft, df_c, df_s, _, _, _, _ = extract_features(Xv, mean_stats, std_stats)
                        pv_1 = model.predict(Xv, ft, df_c, df_s, candv, regv, steps=1)
                        pv_2 = model.predict(Xv, ft, df_c, df_s, candv, regv, steps=2)
                        val_preds_1step_list.append(pv_1.cpu().numpy())
                        val_preds_2step_list.append(pv_2.cpu().numpy())
                        
                val_preds_1 = np.concatenate(val_preds_1step_list, axis=0)
                val_preds_2 = np.concatenate(val_preds_2step_list, axis=0)
                
                val_dists = np.linalg.norm(val_preds_2 - y_val, axis=1)
                
                # Choose checkpoint based on Hit Rate
                val_score = -np.mean(val_dists <= 0.01) * 100.0
                
                if val_score < best_val_score:
                    best_val_score = val_score
                    torch.save(model.state_dict(), models_dir_gliding / f"model_fold_{fold}.pt")
                    best_val_preds_1step = val_preds_1
                    best_val_preds_2step = val_preds_2
                    
            val_indices = np.where(val_mask)[0]
            oof_preds_1step[val_indices] = best_val_preds_1step
            oof_preds_2step[val_indices] = best_val_preds_2step
            
            best_oof_hr = -best_val_score
            print(f"Fold {fold+1} best val Hit Rate: {best_oof_hr:.3f}% (N={len(X_val)})")
            
            # Predict test set using best fold model
            model.load_state_dict(torch.load(models_dir_gliding / f"model_fold_{fold}.pt", map_location=device))
            model.eval()
            
            test_x_tensor = torch.tensor(test_x_all, dtype=torch.float32).to(device)
            test_cand_tensor = torch.tensor(test_candidates_all, dtype=torch.float32).to(device)
            test_reg_mapped = np.zeros(len(test_x_all), dtype=int)
            test_reg_tensor = torch.tensor(test_reg_mapped, dtype=torch.long).to(device)
            
            with torch.no_grad():
                ft_te, df_c_te, df_s_te, _, _, _, _ = extract_features(test_x_tensor, mean_stats, std_stats)
                tp_1 = model.predict(test_x_tensor, ft_te, df_c_te, df_s_te, test_cand_tensor, test_reg_tensor, steps=1)
                tp_2 = model.predict(test_x_tensor, ft_te, df_c_te, df_s_te, test_cand_tensor, test_reg_tensor, steps=2)
                test_preds_1step += tp_1.cpu().numpy() / folds
                test_preds_2step += tp_2.cpu().numpy() / folds
                
        # Save OOF and Test outputs
        np.save(models_dir_gliding / "oof_preds_cfm_1step.npy", oof_preds_1step)
        np.save(models_dir_gliding / "oof_preds_cfm_2step.npy", oof_preds_2step)
        np.save(models_dir_gliding / "test_preds_cfm_1step.npy", test_preds_1step)
        np.save(models_dir_gliding / "test_preds_cfm_2step.npy", test_preds_2step)
        
        # Calculate OOF Hit Rate for Gliding samples
        gliding_oof_dists = np.linalg.norm(oof_preds_2step[is_gliding_tr] - y_train_all[is_gliding_tr], axis=1)
        gliding_hr = np.mean(gliding_oof_dists <= 0.01) * 100.0
        
        success_msg = f"🏆 Finished training Gliding Expert CFM model!\n" \
                      f"Gliding OOF Hit Rate @ 1cm: {gliding_hr:.4f}%"
        print(success_msg)
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [s67 Gliding Expert CFM]\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
