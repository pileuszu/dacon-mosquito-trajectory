import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hashlib
import argparse
import traceback
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification
from model import SotaSpatiotemporalModel, extract_features, SotaFocalSoftHitLoss

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs per model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Noise perturbation scale")
    parser.add_argument("--lambda0", type=float, default=10.0, help="Base Frenet-Flow penalty weight")
    parser.add_argument("--eta", type=float, default=2.0, help="Curvature scale decay rate for Frenet loss")
    args = parser.parse_args()
    
    send_discord_notification(None, f"🚀 Started: [Step 65 train_automl.py] Joint-Training CFM models with Conditional Regime Embedding & Adaptive Frenet-Flow loss...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_dir = Path("step65_spatiotemporal_ai/data")
        X_train = np.load(data_dir / "train_x.npy")
        y_train = np.load(data_dir / "train_y.npy")
        candidates_train = np.load(data_dir / "train_candidates.npy")
        with open(data_dir / "train_ids.json", "r") as f:
            train_ids = json.load(f)
        
        regimes_tr = classify_regimes_np(X_train)
        fold_ids = np.asarray([stable_fold_id(sid, args.folds) for sid in train_ids])
        models_dir = Path("step65_spatiotemporal_ai/models")
        models_dir.mkdir(exist_ok=True)
        
        # Out-of-fold array initialization
        oof_preds_1step = np.zeros_like(y_train)
        oof_preds_2step = np.zeros_like(y_train)
        
        print(f"\n=================== Starting 5-Fold Joint Training (N={len(X_train)}) ===================")
        
        for fold in range(args.folds):
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            
            X_tr, y_tr, cand_tr, reg_tr = X_train[train_mask], y_train[train_mask], candidates_train[train_mask], regimes_tr[train_mask]
            X_val, y_val, cand_val, reg_val = X_train[val_mask], y_train[val_mask], candidates_train[val_mask], regimes_tr[val_mask]
            
            train_loader = DataLoader(MosquitoCandidatesDataset(X_tr, y_tr, cand_tr, reg_tr), batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(MosquitoCandidatesDataset(X_val, y_val, cand_val, reg_val), batch_size=256, shuffle=False)
            
            # Extract mean and std stats for features on training subset
            with torch.no_grad():
                _, _, _, mean_stats, std_stats, _, _ = extract_features(torch.tensor(X_tr, dtype=torch.float32).to(device))
            torch.save({"mean": mean_stats.cpu(), "std": std_stats.cpu()}, models_dir / f"stats_fold_{fold}.pt")
            
            # Model initialization (incorporates dense regime projection)
            model = SotaSpatiotemporalModel(feature_dim=47, latent_dim=128, num_candidates=36, max_norm=0.05, d_mamba_in=9).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            criterion = SotaFocalSoftHitLoss(delta=0.001026, alpha=200.0, target_dist=0.01).to(device)
            
            best_val_hr_2step = 0.0
            best_val_preds_1step = None
            best_val_preds_2step = None
            
            for epoch in range(1, args.epochs + 1):
                model.train()
                for Xb, yb, candb, regb in train_loader:
                    Xb, yb, candb, regb = Xb.to(device), yb.to(device), candb.to(device), regb.to(device)
                    optimizer.zero_grad()
                    ft, df_c, df_s, _, _, (t_v, n_v, b_v), raw_curv = extract_features(Xb, mean_stats, std_stats)
                    pred_v, target_v, x_0, h_context = model.forward_flow(Xb, ft, df_c, df_s, candb, yb, regb, noise_scale=args.noise_scale)
                    
                    # CFM Huber Loss
                    loss_cfm = F.huber_loss(pred_v, target_v, delta=0.001)
                    
                    # Frenet Transverse flow velocity
                    v_N = torch.sum(pred_v * n_v, dim=1)
                    v_B = torch.sum(pred_v * b_v, dim=1)
                    
                    # Curvature-Decayed Adaptive Frenet loss weight
                    lambda_adaptive = args.lambda0 / (1.0 + args.eta * raw_curv)
                    loss_frenet = (lambda_adaptive * (v_N**2 + v_B**2)).mean()
                    
                    loss_flow = loss_cfm + loss_frenet
                    
                    # Predictions and Consistency loss
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
                    
                    loss = 10.0 * loss_flow + 1.0 * loss_pred + 50.0 * loss_cons
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
                # Validation
                model.eval()
                val_hits_2step = 0
                val_preds_1step_list = []
                val_preds_2step_list = []
                with torch.no_grad():
                    for Xv, yv, candv, regv in val_loader:
                        Xv, yv, candv, regv = Xv.to(device), yv.to(device), candv.to(device), regv.to(device)
                        ft, df_c, df_s, _, _, _, _ = extract_features(Xv, mean_stats, std_stats)
                        pv_1 = model.predict(Xv, ft, df_c, df_s, candv, regv, steps=1)
                        pv_2 = model.predict(Xv, ft, df_c, df_s, candv, regv, steps=2)
                        val_hits_2step += (torch.norm(pv_2 - yv, dim=1) <= 0.01).sum().item()
                        val_preds_1step_list.append(pv_1.cpu().numpy())
                        val_preds_2step_list.append(pv_2.cpu().numpy())
                
                val_hr_2 = val_hits_2step / len(X_val)
                if val_hr_2 >= best_val_hr_2step:
                    best_val_hr_2step = val_hr_2
                    torch.save(model.state_dict(), models_dir / f"model_fold_{fold}.pt")
                    best_val_preds_1step = np.concatenate(val_preds_1step_list, axis=0)
                    best_val_preds_2step = np.concatenate(val_preds_2step_list, axis=0)
            
            # Save OOF validation coordinates
            val_indices = np.where(val_mask)[0]
            oof_preds_1step[val_indices] = best_val_preds_1step
            oof_preds_2step[val_indices] = best_val_preds_2step
            
            print(f"Fold {fold+1} Finished. Best Hit@1cm: {best_val_hr_2step*100:.3f}%")
            
        # Calculate overall combined OOF scores
        overall_hr_1 = np.mean(np.linalg.norm(oof_preds_1step - y_train, axis=1) <= 0.01)
        overall_hr_2 = np.mean(np.linalg.norm(oof_preds_2step - y_train, axis=1) <= 0.01)
        print(f"\n=================== JOINT-CFM S65 OOF RESULTS ===================")
        print(f"Overall OOF Hit Rate@1cm (1-Step): {overall_hr_1*100:.5f}%")
        print(f"Overall OOF Hit Rate@1cm (2-Step): {overall_hr_2*100:.5f}%")
        print(f"=================================================================")
        
        np.save(data_dir / "oof_preds_cfm_1step.npy", oof_preds_1step)
        np.save(data_dir / "oof_preds_cfm_2step.npy", oof_preds_2step)
        
        success_msg = f"✅ Finished: [Step 65] Joint CFM OOF 1-Step: {overall_hr_1*100:.3f}%, 2-Step: {overall_hr_2*100:.3f}%"
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 65]\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
