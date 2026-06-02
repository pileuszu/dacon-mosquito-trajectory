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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification
from model import FrenetNeuralODEModel, extract_features

class MosquitoSimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def stable_fold_id(sample_id: str, folds: int) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds

def visualize_validation_paths(model, X_val, y_val, mean_stats, std_stats, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    # Calculate errors to find high-curvature / steering cases
    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        ft, df, plt_, tht, is_steering, last_a, _, Rt, spt, _, _ = extract_features(val_tensor, mean_stats, std_stats)
        preds = model(ft, df, plt_, tht, spt, Rt, last_a)
        errors = torch.norm(preds - y_val_tensor, dim=1).cpu().numpy() * 100 # in cm
        is_steering_np = is_steering.cpu().numpy()
        
    # We want to select 5 representative turning (Steering) samples
    steering_indices = np.where(is_steering_np)[0]
    if len(steering_indices) == 0:
        steering_indices = np.arange(len(X_val))
        
    # Select 5 samples spread across the set
    selected_indices = steering_indices[np.linspace(0, len(steering_indices) - 1, min(5, len(steering_indices)), dtype=int)]
    
    print(f"Generating 3D validation visualization plots for indices: {selected_indices}")
    
    for idx in selected_indices:
        history = X_val[idx] # (11, 3)
        target = y_val[idx] # (3,)
        pred = preds[idx].cpu().numpy() # (3,)
        err_cm = errors[idx]
        
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot past history
        ax.plot(history[:, 0], history[:, 1], history[:, 2], 'ko-', label='Past Trajectory (History)', linewidth=2, markersize=5)
        
        # Plot target
        ax.scatter(target[0], target[1], target[2], color='red', s=100, label='True Target', marker='*', zorder=5)
        
        # Plot model prediction
        ax.scatter(pred[0], pred[1], pred[2], color='#ff7f0e', s=80, label=f'Frenet Focal Pred (Err: {err_cm:.2f}cm)', marker='s', zorder=4)
        
        # Draw lines
        last_pt = history[-1]
        ax.plot([last_pt[0], target[0]], [last_pt[1], target[1]], [last_pt[2], target[2]], 'r--', alpha=0.5)
        ax.plot([last_pt[0], pred[0]], [last_pt[1], pred[1]], [last_pt[2], pred[2]], 'g--', alpha=0.5)
        
        # Draw 1.0cm hit sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xs = target[0] + 0.01 * np.cos(u) * np.sin(v)
        ys = target[1] + 0.01 * np.sin(u) * np.sin(v)
        zs = target[2] + 0.01 * np.cos(v)
        ax.plot_wireframe(xs, ys, zs, color='red', alpha=0.08, label='1.0cm Hit Boundary')
        
        ax.set_title(f"3D Path: Frenet Neural ODE (Focal Sample {idx})", fontsize=13, fontweight='bold')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(frameon=True, facecolor='white', shadow=True)
        
        plot_path = out_dir / f"focal_trajectory_sample_{idx}.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved validation 3D plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--fold-limit", type=int, default=None, help="Limit number of folds to train")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    args = parser.parse_args()
    
    send_discord_notification(
        None,
        f"🚀 Started: [Step 52 train_automl.py] Training 5-fold Frenet-Guided Neural ODE (Focal Loss & Z-dynamics)...\n"
        f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, fold_limit={args.fold_limit}"
    )
    
    try:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"Using device: {device}")
        
        # Load dataset
        data_dir = Path("step52_focal_ode/data")
        X_train = np.load(data_dir / "train_x.npy")
        y_train = np.load(data_dir / "train_y.npy")
        
        with open(data_dir / "train_ids.json", "r") as f:
            train_ids = json.load(f)
            
        print(f"Loaded train data. X_train: {X_train.shape}, y_train: {y_train.shape}")
        
        # Compute stable fold ids
        fold_ids = np.asarray([stable_fold_id(sid, args.folds) for sid in train_ids])
        
        oof_preds = np.zeros_like(y_train)
        models_dir = Path("step52_focal_ode/models")
        models_dir.mkdir(exist_ok=True)
        
        fold_hr_scores = []
        folds_to_train = args.folds if args.fold_limit is None else args.fold_limit
        
        for fold in range(folds_to_train):
            print(f"\n--- FOLD {fold + 1} / {args.folds} ---")
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            
            X_tr, y_tr = X_train[train_mask], y_train[train_mask]
            X_val, y_val = X_train[val_mask], y_train[val_mask]
            
            train_ds = MosquitoSimpleDataset(X_tr, y_tr)
            val_ds = MosquitoSimpleDataset(X_val, y_val)
            
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
            
            # Compute feature scaling stats using training set
            train_coords_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, _, _, _, _, _, _, _, _, mean_stats, std_stats = extract_features(train_coords_tensor)
                
            stats_path = models_dir / f"stats_fold_{fold}.pt"
            torch.save({"mean": mean_stats.cpu(), "std": std_stats.cpu()}, stats_path)
            
            model = FrenetNeuralODEModel(input_dim=27, latent_dim=64).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            best_val_hr = 0.0
            best_epoch = 0
            
            for epoch in range(1, args.epochs + 1):
                model.train()
                epoch_loss = 0.0
                
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    
                    ft, df, plt_, tht, is_steering_batch, last_a, _, Rt, spt, _, _ = extract_features(Xb, mean_stats, std_stats)
                    pred = model(ft, df, plt_, tht, spt, Rt, last_a)
                    
                    # 1. Huber Loss + Focal Soft-Hit Loss
                    d = torch.norm(pred - yb, dim=1)
                    
                    # Target exactly 1.0cm (0.01m) with sharper sigmoid scale (400.0)
                    soft_hit_samples = 1 - torch.sigmoid(-(d - 0.01) * 400.0)
                    # Apply 2.5x weight penalty for near-misses (1.0cm < d <= 1.5cm)
                    focal_weight = torch.where((d > 0.01) & (d <= 0.015), 2.5, 1.0)
                    
                    weighted_soft_hit = (focal_weight * soft_hit_samples).mean()
                    huber = F.huber_loss(pred, yb, delta=0.001026)
                    
                    # 2. Dynamic Acceleration Regularization Loss
                    lambda_dynamic = torch.where(is_steering_batch, 1e-5, 1e-3).to(device)
                    
                    reg_loss = 0.0
                    if hasattr(model, "_last_accels") and model._last_accels:
                        accel_reg_samples = sum(a.pow(2).sum(-1) for a in model._last_accels) / len(model._last_accels)
                        reg_loss = (lambda_dynamic * accel_reg_samples).mean()
                        
                    loss = weighted_soft_hit + 126.309 * huber + reg_loss
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * len(Xb)
                    
                epoch_loss /= len(train_loader.dataset)
                
                # Validation evaluation
                model.eval()
                val_hits = 0
                val_preds_list = []
                
                with torch.no_grad():
                    for Xv, yv in val_loader:
                        Xv, yv = Xv.to(device), yv.to(device)
                        ft, df, plt_, tht, _, last_a, _, Rt, spt, _, _ = extract_features(Xv, mean_stats, std_stats)
                        pv = model(ft, df, plt_, tht, spt, Rt, last_a)
                        
                        dist = torch.norm(pv - yv, dim=1)
                        val_hits += (dist <= 0.01).sum().item()
                        val_preds_list.append(pv.cpu().numpy())
                        
                val_hr = val_hits / len(X_val)
                val_preds_arr = np.concatenate(val_preds_list, axis=0)
                
                print(f"  Epoch {epoch:2d}/{args.epochs} | Train Loss: {epoch_loss:.6f} | Val Hit@1cm: {val_hr * 100:.3f}%")
                
                if val_hr >= best_val_hr:
                    best_val_hr = val_hr
                    best_epoch = epoch
                    model_path = models_dir / f"model_fold_{fold}.pt"
                    torch.save(model.state_dict(), model_path)
                    best_val_preds = val_preds_arr
                    
            print(f"Fold {fold + 1} Training Finished. Best Val Hit@1cm: {best_val_hr * 100:.3f}% at Epoch {best_epoch}")
            oof_preds[val_mask] = best_val_preds
            fold_hr_scores.append(best_val_hr)
            
            # Generate 3D visualization plots on Fold 1
            if fold == 0:
                print("\nGenerating 3D validation trajectory plots to verify Focal ODE performance...")
                val_vis_dir = Path("outputs/step52_focal_ode/val_visualizations")
                best_model = FrenetNeuralODEModel(input_dim=27, latent_dim=64).to(device)
                best_model.load_state_dict(torch.load(models_dir / f"model_fold_0.pt", map_location=device))
                visualize_validation_paths(best_model, X_val, y_val, mean_stats, std_stats, device, val_vis_dir)
            
            if args.epochs > 1:
                send_discord_notification(
                    None,
                    f"📢 Fold {fold + 1}/{args.folds} Finished | Best Val Hit@1cm: {best_val_hr * 100:.3f}% (Epoch {best_epoch})"
                )
                
        # Overall OOF calculation
        if args.fold_limit is None:
            overall_hits = np.sum(np.linalg.norm(oof_preds - y_train, axis=1) <= 0.01)
            overall_hr = overall_hits / len(y_train)
            print(f"\n==========================================")
            print(f"Overall 5-Fold OOF Hit Rate@1cm: {overall_hr * 100:.3f}%")
            print(f"==========================================")
            
            np.save("step52_focal_ode/oof_predictions.npy", oof_preds)
            print("Saved OOF predictions to step52_focal_ode/oof_predictions.npy")
            
            send_discord_notification(
                None,
                f"✅ Finished: [Step 52 train_automl.py] Frenet Neural ODE Training Complete!\n"
                f"Overall OOF Hit Rate@1cm: **{overall_hr * 100:.3f}%**\n"
                f"Fold scores: {[round(s*100, 3) for s in fold_hr_scores]}"
            )
        else:
            print(f"\nDry run completed for {args.fold_limit} fold(s).")
            send_discord_notification(
                None,
                f"✅ Finished: [Step 52 train_automl.py] Dry Run Completed Successfully!\n"
                f"Fold scores: {[round(s*100, 3) for s in fold_hr_scores]}"
            )
            
    except Exception as e:
        error_msg = f"❌ Failed: [Step 52 train_automl.py]\nError: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
