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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.notifier import send_discord_notification
from model_s67 import SotaSpatiotemporalModel
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
        if is_steering[i]:
            regimes[i] = 2
        elif speeds[i] <= 0.50:
            regimes[i] = 0
        else:
            regimes[i] = 1
    return regimes

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent embedding dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--lambda0", type=float, default=10.0, help="Base Frenet-Flow penalty weight")
    parser.add_argument("--eta", type=float, default=2.0, help="Curvature scale decay rate for Frenet loss")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Epochs to train with pure Huber loss before focal")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Noise perturbation scale")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
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
    warmup_epochs = args.warmup_epochs
    dropout = args.dropout
    loss_dist_weight = args.loss_dist_weight
    
    send_discord_notification(None, f"🚀 Started: [s67 CFM training] latent_dim={args.latent_dim}, heads={args.num_heads}, lambda0={lambda0}, loss_dist_weight={loss_dist_weight}, dropout={dropout}")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Paths
        data_dir_s65 = Path("step65_spatiotemporal_ai/data")
        models_dir_s67 = Path("step66_super_feature/models/s67")
        models_dir_s67.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        X_train = np.load(data_dir_s65 / "train_x.npy")
        y_train = np.load(data_dir_s65 / "train_y.npy")
        candidates_train = np.load(data_dir_s65 / "train_candidates.npy")
        with open(data_dir_s65 / "train_ids.json", "r") as f:
            train_ids = json.load(f)
            
        test_x = np.load(data_dir_s65 / "test_x.npy")
        test_candidates = np.load(data_dir_s65 / "test_candidates.npy")
        with open(data_dir_s65 / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        regimes_tr = classify_regimes_np(X_train)
        regimes_te = classify_regimes_np(test_x)
        
        fold_ids = np.asarray([stable_fold_id(sid, folds) for sid in train_ids])
        
        oof_preds_1step = np.zeros_like(y_train)
        oof_preds_2step = np.zeros_like(y_train)
        
        test_preds_1step = np.zeros((len(test_x), 3))
        test_preds_2step = np.zeros((len(test_x), 3))
        
        print(f"\n=================== Starting 5-Fold S67 Cross-Attention training ===================")
        
        for fold in range(folds):
            print(f"\n--- Fold {fold+1} / {folds} ---")
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            
            X_tr, y_tr, cand_tr, reg_tr = X_train[train_mask], y_train[train_mask], candidates_train[train_mask], regimes_tr[train_mask]
            X_val, y_val, cand_val, reg_val = X_train[val_mask], y_train[val_mask], candidates_train[val_mask], regimes_tr[val_mask]
            
            train_loader = DataLoader(MosquitoCandidatesDataset(X_tr, y_tr, cand_tr, reg_tr), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(MosquitoCandidatesDataset(X_val, y_val, cand_val, reg_val), batch_size=256, shuffle=False)
            
            # Feature Stats
            with torch.no_grad():
                _, _, _, mean_stats, std_stats, _, _ = extract_features(torch.tensor(X_tr, dtype=torch.float32).to(device))
            torch.save({"mean": mean_stats.cpu(), "std": std_stats.cpu()}, models_dir_s67 / f"stats_fold_{fold}.pt")
            
            # Model with Cross-Attention vector field
            model = SotaSpatiotemporalModel(feature_dim=47, latent_dim=args.latent_dim, num_candidates=36, max_norm=0.05, d_mamba_in=9, dropout=dropout).to(device)
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
                    
                    # CFM Huber Loss
                    loss_cfm = F.huber_loss(pred_v, target_v, delta=0.001)
                    
                    # Frenet Transverse Flow Loss
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
                    if epoch <= warmup_epochs:
                        loss_pred = 0.0
                    else:
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
                val_reg_list = []
                with torch.no_grad():
                    for Xv, yv, candv, regv in val_loader:
                        Xv, yv, candv, regv = Xv.to(device), yv.to(device), candv.to(device), regv.to(device)
                        ft, df_c, df_s, _, _, _, _ = extract_features(Xv, mean_stats, std_stats)
                        pv_1 = model.predict(Xv, ft, df_c, df_s, candv, regv, steps=1)
                        pv_2 = model.predict(Xv, ft, df_c, df_s, candv, regv, steps=2)
                        val_preds_1step_list.append(pv_1.cpu().numpy())
                        val_preds_2step_list.append(pv_2.cpu().numpy())
                        val_reg_list.append(regv.cpu().numpy())
                        
                val_preds_1 = np.concatenate(val_preds_1step_list, axis=0)
                val_preds_2 = np.concatenate(val_preds_2step_list, axis=0)
                val_regs = np.concatenate(val_reg_list, axis=0)
                
                val_dists = np.linalg.norm(val_preds_2 - y_val, axis=1)
                
                # Compute weighted statistics matching test set distribution
                props = {0: 0.1514, 1: 0.2343, 2: 0.6143}
                w_mean = 0.0
                w_std = 0.0
                w_hr_2 = 0.0
                
                for r in [0, 1, 2]:
                    r_mask = val_regs == r
                    if np.sum(r_mask) > 0:
                        r_dists = val_dists[r_mask]
                        w_mean += props[r] * np.mean(r_dists) * 100.0
                        w_std += props[r] * np.std(r_dists) * 100.0
                        w_hr_2 += props[r] * np.mean(r_dists <= 0.01) * 100.0
                        
                val_score = w_mean + 0.2 * w_std
                
                if val_score < best_val_score:
                    best_val_score = val_score
                    torch.save(model.state_dict(), models_dir_s67 / f"model_fold_{fold}.pt")
                    best_val_preds_1step = val_preds_1
                    best_val_preds_2step = val_preds_2
                    
            # Record OOF predictions
            val_indices = np.where(val_mask)[0]
            oof_preds_1step[val_indices] = best_val_preds_1step
            oof_preds_2step[val_indices] = best_val_preds_2step
            best_oof_dists = np.linalg.norm(best_val_preds_2step - y_val, axis=1)
            
            # Compute weighted metrics for fold best predictions
            w_best_mean = 0.0
            w_best_std = 0.0
            w_best_hr = 0.0
            props = {0: 0.1514, 1: 0.2343, 2: 0.6143}
            for r in [0, 1, 2]:
                r_mask = reg_val == r
                if np.sum(r_mask) > 0:
                    r_dists = best_oof_dists[r_mask]
                    w_best_mean += props[r] * np.mean(r_dists) * 100.0
                    w_best_std += props[r] * np.std(r_dists) * 100.0
                    w_best_hr += props[r] * np.mean(r_dists <= 0.01) * 100.0
                    
            print(f"Fold {fold+1} best val Score: {best_val_score:.4f} (Weighted Mean: {w_best_mean:.4f}cm, Std: {w_best_std:.4f}cm, Hit@1cm: {w_best_hr:.3f}%)")
            
            # Predict Test Set using best fold model (accumulate average)
            model.load_state_dict(torch.load(models_dir_s67 / f"model_fold_{fold}.pt", map_location=device))
            model.eval()
            
            test_x_tensor = torch.tensor(test_x, dtype=torch.float32).to(device)
            test_cand_tensor = torch.tensor(test_candidates, dtype=torch.float32).to(device)
            test_reg_tensor = torch.tensor(regimes_te, dtype=torch.long).to(device)
            
            with torch.no_grad():
                ft_te, df_c_te, df_s_te, _, _, _, _ = extract_features(test_x_tensor, mean_stats, std_stats)
                tp_1 = model.predict(test_x_tensor, ft_te, df_c_te, df_s_te, test_cand_tensor, test_reg_tensor, steps=1)
                tp_2 = model.predict(test_x_tensor, ft_te, df_c_te, df_s_te, test_cand_tensor, test_reg_tensor, steps=2)
                test_preds_1step += tp_1.cpu().numpy() / folds
                test_preds_2step += tp_2.cpu().numpy() / folds
                
        # Calculate overall combined OOF scores (weighted by test set distribution)
        oof_dists_1step = np.linalg.norm(oof_preds_1step - y_train, axis=1) * 100.0
        oof_dists_2step = np.linalg.norm(oof_preds_2step - y_train, axis=1) * 100.0
        
        w_mean_1 = 0.0
        w_std_1 = 0.0
        w_hr_1 = 0.0
        
        w_mean_2 = 0.0
        w_std_2 = 0.0
        w_hr_2 = 0.0
        
        for r in [0, 1, 2]:
            r_mask = regimes_tr == r
            if np.sum(r_mask) > 0:
                # 1-step
                d1 = oof_dists_1step[r_mask]
                w_mean_1 += props[r] * np.mean(d1)
                w_std_1 += props[r] * np.std(d1)
                w_hr_1 += props[r] * np.mean(d1 <= 1.0) * 100.0
                
                # 2-step
                d2 = oof_dists_2step[r_mask]
                w_mean_2 += props[r] * np.mean(d2)
                w_std_2 += props[r] * np.std(d2)
                w_hr_2 += props[r] * np.mean(d2 <= 1.0) * 100.0
                
        print(f"\n=================== JOINT-CFM S67 OOF RESULTS (Test-Proportion Weighted) ===================")
        print(f"Overall OOF 1-Step: Weighted Mean: {w_mean_1:.4f}cm, Std: {w_std_1:.4f}cm, Max: {np.max(oof_dists_1step):.4f}cm, Hit@1cm: {w_hr_1:.3f}%")
        print(f"Overall OOF 2-Step: Weighted Mean: {w_mean_2:.4f}cm, Std: {w_std_2:.4f}cm, Max: {np.max(oof_dists_2step):.4f}cm, Hit@1cm: {w_hr_2:.3f}%")
        print(f"=========================================================================================")
        
        # Save OOF and Test outputs
        np.save(models_dir_s67 / "oof_preds_cfm_1step.npy", oof_preds_1step)
        np.save(models_dir_s67 / "oof_preds_cfm_2step.npy", oof_preds_2step)
        np.save(models_dir_s67 / "test_preds_cfm_1step.npy", test_preds_1step)
        np.save(models_dir_s67 / "test_preds_cfm_2step.npy", test_preds_2step)
        
        success_msg = f"✅ Finished: [Step 67 Cross-Attn CFM]\n" \
                      f"1-Step: Weighted Mean={w_mean_1:.3f}cm, Std={w_std_1:.3f}cm, Hit@1cm={w_hr_1:.2f}%\n" \
                      f"2-Step: Weighted Mean={w_mean_2:.3f}cm, Std={w_std_2:.3f}cm, Hit@1cm={w_hr_2:.2f}%"
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 67 Cross-Attn CFM]\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
