import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import BaselineGRU
import os

# Configuration
DATA_DIR = 'data/open/train/'
LABEL_PATH = 'data/open/train_labels.csv'
MODEL_PATH = 'outputs/step1/best_baseline_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_and_visualize(num_samples=3):
    # 1. Load Data
    _, val_loader = get_dataloaders(DATA_DIR, LABEL_PATH, batch_size=1)
    
    # 2. Load Model
    model = BaselineGRU().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}")
        return

    model.eval()
    
    mae_list = []
    cv_mae_list = []
    
    samples_found = 0
    with torch.no_grad():
        for hist, target, origin in val_loader:
            hist, target = hist.to(DEVICE), target.to(DEVICE)
            output = model(hist)
            
            # Metrics (L1 Error)
            mae = torch.abs(output - target).mean().item()
            mae_list.append(mae)
            
            # Constant Velocity (CV) Baseline for comparison
            # CV assumes last displacement continues: P_pred = P_0 + (P_0 - P_{-1}) * gap_timesteps
            # Since DT=40ms and target is at +80ms, gap = 2 timesteps
            v_last = (hist[0, -1] - hist[0, -2]) # last velocity vector (unit: displacement per 40ms)
            cv_pred = hist[0, -1] + v_last * 2 # Predict +80ms
            cv_mae = torch.abs(cv_pred - target[0]).mean().item()
            cv_mae_list.append(cv_mae)
            
            # Visualization for the first few samples
            if samples_found < num_samples:
                visualize_sample(hist[0].cpu(), target[0].cpu(), output[0].cpu(), cv_pred.cpu(), samples_found)
                samples_found += 1
                
    print(f"\nEvaluation Results (Mean over {len(mae_list)} samples):")
    print(f"Model MAE: {np.mean(mae_list):.6f}")
    print(f"CV Baseline MAE: {np.mean(cv_mae_list):.6f}")
    print(f"Improvement: {((np.mean(cv_mae_list) - np.mean(mae_list)) / np.mean(cv_mae_list)) * 100:.2f}%")

def visualize_sample(hist, target, pred, cv_pred, idx):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot History
    ax.plot(hist[:, 0], hist[:, 1], hist[:, 2], 'bo-', label='History', alpha=0.6)
    
    # Plot Target
    ax.scatter(target[0], target[1], target[2], color='red', s=100, label='Ground Truth', marker='X')
    
    # Plot Prediction
    ax.scatter(pred[0], pred[1], pred[2], color='green', s=100, label='GRU Prediction', marker='o')
    
    # Plot CV Prediction
    ax.scatter(cv_pred[0], cv_pred[1], cv_pred[2], color='orange', s=80, label='CV Baseline', marker='^')
    
    ax.set_xlabel('X (Relative)')
    ax.set_ylabel('Y (Relative)')
    ax.set_zlabel('Z (Relative)')
    ax.set_title(f'Sample {idx+1}: Trajectory Prediction Comparison')
    ax.legend()
    
    save_path = f'outputs/step1/visualization_sample_{idx+1}.png'
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()

if __name__ == "__main__":
    evaluate_and_visualize()
